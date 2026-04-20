"""Hybrid Feedback Pipeline — orchestrates grading, concept gap detection,
T5 generative feedback, NLI consistency checking, and template fallback.

Pipeline flow:
1. Grade the student answer (via a GradingModel).
2. Detect concept gaps (via ConceptGapDetector).
3. Generate grounded T5 feedback (via T5GenerativeFeedbackGenerator).
4. Run NLI factual consistency check on the generated feedback.
5. If consistency score < threshold → fall back to template-based feedback.

Validates: Requirements 20, 18 (fallback)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.schema import UnifiedRecord
from src.feedback.concept_gap import ConceptGapDetector, ConceptGapResult
from src.feedback.generative import (
    FactualConsistencyChecker,
    T5GenerativeFeedbackGenerator,
)
from src.feedback.template import FeedbackGenerator, TemplateFeedbackGenerator


@dataclass
class HybridFeedbackResult:
    """Full metadata from the hybrid feedback pipeline."""

    feedback_short: str = ""
    feedback_detailed: str = ""
    predicted_label: str = ""
    gap_result: ConceptGapResult | None = None
    consistency_score: float = 1.0
    used_fallback: bool = False
    contradicting_claims: list[str] = field(default_factory=list)


class HybridFeedbackPipeline(FeedbackGenerator):
    """Orchestrates grade → concept gap → T5 generation → NLI check → fallback.

    Parameters
    ----------
    grading_model : object
        Any object with a ``predict(records)`` method returning a list of
        predicted labels (strings). Follows the ``GradingModel`` interface.
    concept_gap_detector : ConceptGapDetector
        Detects which key concepts are present/missing/contradicted.
    generative_generator : T5GenerativeFeedbackGenerator
        Fine-tuned T5 feedback generator (grounded mode recommended).
    consistency_checker : FactualConsistencyChecker
        NLI-based factual consistency checker.
    template_generator : TemplateFeedbackGenerator | None
        Template-based fallback generator. Created automatically if ``None``.
    consistency_threshold : float
        Minimum consistency score to accept T5 output. Below this value
        the pipeline falls back to template-based feedback.
    """

    def __init__(
        self,
        grading_model: object,
        concept_gap_detector: ConceptGapDetector,
        generative_generator: T5GenerativeFeedbackGenerator,
        consistency_checker: FactualConsistencyChecker,
        template_generator: TemplateFeedbackGenerator | None = None,
        consistency_threshold: float = 0.5,
    ) -> None:
        self._grading_model = grading_model
        self._concept_gap_detector = concept_gap_detector
        self._generative_generator = generative_generator
        self._consistency_checker = consistency_checker
        self._template_generator = template_generator or TemplateFeedbackGenerator()
        self._consistency_threshold = consistency_threshold

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def consistency_threshold(self) -> float:
        """Return the current consistency threshold."""
        return self._consistency_threshold

    @consistency_threshold.setter
    def consistency_threshold(self, value: float) -> None:
        self._consistency_threshold = value

    # ------------------------------------------------------------------
    # FeedbackGenerator interface
    # ------------------------------------------------------------------

    def generate(
        self,
        record: UnifiedRecord,
        gap_result: ConceptGapResult,
        predicted_label: str,
    ) -> tuple[str, str]:
        """Generate feedback via the hybrid pipeline.

        This simplified interface accepts a pre-computed gap result and
        predicted label, skipping the internal grading and gap detection
        steps. Useful when the caller has already performed those steps.

        Returns
        -------
        tuple[str, str]
            ``(feedback_short, feedback_detailed)``
        """
        result = self._run_generation_and_check(
            record, gap_result, predicted_label,
        )
        return result.feedback_short, result.feedback_detailed

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, record: UnifiedRecord) -> HybridFeedbackResult:
        """Execute the full hybrid pipeline on a single record.

        Steps:
        1. Grade the student answer.
        2. Detect concept gaps.
        3. Generate grounded T5 feedback.
        4. NLI consistency check.
        5. Fall back to template if consistency < threshold.

        Parameters
        ----------
        record : UnifiedRecord
            The student answer record.

        Returns
        -------
        HybridFeedbackResult
            Feedback text plus pipeline metadata.
        """
        # Step 1: Grade
        predicted_label = self._grade(record)

        # Step 2: Detect concept gaps
        key_concepts = list(record.key_concepts) if record.key_concepts else None
        gap_result = self._concept_gap_detector.detect(
            question=record.question,
            reference_answer=record.reference_answer,
            student_answer=record.student_answer,
            key_concepts=key_concepts,
        )

        # Steps 3–5: Generate + check + fallback
        result = self._run_generation_and_check(record, gap_result, predicted_label)
        result.predicted_label = predicted_label
        result.gap_result = gap_result
        return result

    def run_batch(
        self, records: list[UnifiedRecord],
    ) -> list[HybridFeedbackResult]:
        """Execute the hybrid pipeline on a batch of records.

        Parameters
        ----------
        records : list[UnifiedRecord]
            Batch of student answer records.

        Returns
        -------
        list[HybridFeedbackResult]
            One result per input record.
        """
        return [self.run(record) for record in records]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grade(self, record: UnifiedRecord) -> str:
        """Obtain a predicted label from the grading model."""
        predictions = self._grading_model.predict([record])
        return str(predictions[0]) if predictions else "incorrect"

    def _run_generation_and_check(
        self,
        record: UnifiedRecord,
        gap_result: ConceptGapResult,
        predicted_label: str,
    ) -> HybridFeedbackResult:
        """Generate T5 feedback, check consistency, fallback if needed."""
        # Step 3: Generate grounded T5 feedback
        gen_result = self._generative_generator.generate_with_metadata(
            record, gap_result, predicted_label,
        )

        # Step 4: NLI consistency check
        consistency_score, contradicting = self._consistency_checker.check(
            generated_feedback=gen_result.feedback_detailed,
            reference_answer=record.reference_answer,
        )

        # Step 5: Fallback decision
        if consistency_score < self._consistency_threshold:
            # Fall back to template-based feedback
            fb_short, fb_detailed = self._template_generator.generate(
                record, gap_result, predicted_label,
            )
            return HybridFeedbackResult(
                feedback_short=fb_short,
                feedback_detailed=fb_detailed,
                predicted_label=predicted_label,
                gap_result=gap_result,
                consistency_score=consistency_score,
                used_fallback=True,
                contradicting_claims=contradicting,
            )

        # T5 output is consistent enough — use it
        return HybridFeedbackResult(
            feedback_short=gen_result.feedback_short,
            feedback_detailed=gen_result.feedback_detailed,
            predicted_label=predicted_label,
            gap_result=gap_result,
            consistency_score=consistency_score,
            used_fallback=False,
            contradicting_claims=contradicting,
        )
