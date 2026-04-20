"""Integration tests for the Hybrid Feedback Pipeline.

Covers:
- 25.1: Full pipeline: grade → concept gaps → T5 feedback → NLI check → fallback
- 25.2: Integration tests for the full pipeline

Validates: Requirements 20, 18 (fallback)
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.data.schema import UnifiedRecord
from src.feedback.concept_gap import ConceptGapResult
from src.feedback.generative import GenerativeFeedbackResult
from src.feedback.hybrid import HybridFeedbackPipeline, HybridFeedbackResult
from src.feedback.template import FeedbackGenerator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_record(**overrides) -> UnifiedRecord:
    defaults = dict(
        sample_id="GEN_0001",
        source_dataset="data_generate",
        original_id="orig_1",
        question_id="q_1",
        domain="biology",
        subdomain="cell_biology",
        difficulty="medium",
        question="What is photosynthesis?",
        reference_answer="Photosynthesis converts light energy into chemical energy.",
        student_answer="Plants make food from sunlight.",
        key_concepts=["photosynthesis", "light energy", "chemical energy"],
        missing_concepts=["chemical energy"],
        label_3way="partially_correct",
    )
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


def _make_mock_grading_model(label: str = "partially_correct"):
    """Mock grading model that always returns the given label."""
    mock = MagicMock()
    mock.predict = MagicMock(return_value=[label])
    return mock


def _make_mock_concept_gap_detector(
    present: list[str] | None = None,
    missing: list[str] | None = None,
    contradicted: list[str] | None = None,
):
    """Mock ConceptGapDetector returning fixed results."""
    mock = MagicMock(spec=["detect"])
    gap = ConceptGapResult(
        present_concepts=present or [],
        missing_concepts=missing or ["chemical energy"],
        contradicted_concepts=contradicted or [],
    )
    mock.detect = MagicMock(return_value=gap)
    return mock


def _make_mock_t5_generator(
    feedback_short: str = "T5 short feedback.",
    feedback_detailed: str = "T5 detailed feedback about the topic.",
    consistency_score: float = 1.0,
    contradicting: list[str] | None = None,
):
    """Mock T5GenerativeFeedbackGenerator."""
    mock = MagicMock(spec=["generate", "generate_with_metadata"])
    result = GenerativeFeedbackResult(
        feedback_short=feedback_short,
        feedback_detailed=feedback_detailed,
        grounded=True,
        is_potential_hallucination=len(contradicting or []) > 0,
        consistency_score=consistency_score,
        contradicting_claims=contradicting or [],
    )
    mock.generate_with_metadata = MagicMock(return_value=result)
    mock.generate = MagicMock(return_value=(feedback_short, feedback_detailed))
    return mock


def _make_mock_consistency_checker(
    consistency_score: float = 0.9,
    contradicting: list[str] | None = None,
):
    """Mock FactualConsistencyChecker."""
    mock = MagicMock(spec=["check"])
    mock.check = MagicMock(return_value=(consistency_score, contradicting or []))
    return mock


# ------------------------------------------------------------------
# Test: HybridFeedbackPipeline is a FeedbackGenerator
# ------------------------------------------------------------------


class TestHybridInterface:
    """Verify the pipeline implements the FeedbackGenerator ABC."""

    def test_is_subclass_of_feedback_generator(self):
        assert issubclass(HybridFeedbackPipeline, FeedbackGenerator)

    def test_generate_returns_tuple(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
        )
        record = _make_record()
        gap = ConceptGapResult(missing_concepts=["chemical energy"])
        result = pipeline.generate(record, gap, "partially_correct")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)


# ------------------------------------------------------------------
# Test: Full pipeline — T5 output accepted (high consistency)
# ------------------------------------------------------------------


class TestPipelineAcceptsT5:
    """When T5 consistency is above threshold, use T5 output."""

    def test_high_consistency_uses_t5_feedback(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model("partially_correct"),
            concept_gap_detector=_make_mock_concept_gap_detector(
                missing=["chemical energy"],
            ),
            generative_generator=_make_mock_t5_generator(
                feedback_short="T5 short.",
                feedback_detailed="T5 detailed about chemical energy.",
            ),
            consistency_checker=_make_mock_consistency_checker(
                consistency_score=0.9,
            ),
            consistency_threshold=0.5,
        )

        record = _make_record()
        result = pipeline.run(record)

        assert isinstance(result, HybridFeedbackResult)
        assert result.used_fallback is False
        assert result.feedback_short == "T5 short."
        assert result.feedback_detailed == "T5 detailed about chemical energy."
        assert result.consistency_score == 0.9
        assert result.predicted_label == "partially_correct"

    def test_exact_threshold_uses_t5(self):
        """Consistency == threshold should NOT trigger fallback."""
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(
                feedback_short="T5.",
                feedback_detailed="T5 detailed.",
            ),
            consistency_checker=_make_mock_consistency_checker(
                consistency_score=0.5,
            ),
            consistency_threshold=0.5,
        )

        result = pipeline.run(_make_record())
        assert result.used_fallback is False
        assert result.feedback_short == "T5."


# ------------------------------------------------------------------
# Test: Full pipeline — fallback to template (low consistency)
# ------------------------------------------------------------------


class TestPipelineFallsBackToTemplate:
    """When T5 consistency is below threshold, fall back to template."""

    def test_low_consistency_triggers_fallback(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model("incorrect"),
            concept_gap_detector=_make_mock_concept_gap_detector(
                missing=["chemical energy", "light energy"],
            ),
            generative_generator=_make_mock_t5_generator(
                feedback_short="Bad T5.",
                feedback_detailed="Bad T5 with hallucinations.",
            ),
            consistency_checker=_make_mock_consistency_checker(
                consistency_score=0.3,
                contradicting=["Bad T5 with hallucinations."],
            ),
            consistency_threshold=0.5,
        )

        record = _make_record()
        result = pipeline.run(record)

        assert result.used_fallback is True
        assert result.consistency_score == 0.3
        assert len(result.contradicting_claims) == 1
        # Template feedback should reference the topic
        assert "photosynthesis" in result.feedback_short.lower() or \
               "what is" in result.feedback_short.lower()

    def test_zero_consistency_triggers_fallback(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model("incorrect"),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(
                consistency_score=0.0,
                contradicting=["claim1", "claim2"],
            ),
            consistency_threshold=0.5,
        )

        result = pipeline.run(_make_record())
        assert result.used_fallback is True
        assert result.consistency_score == 0.0


# ------------------------------------------------------------------
# Test: Pipeline steps are called in order
# ------------------------------------------------------------------


class TestPipelineOrchestration:
    """Verify the pipeline calls components in the correct order."""

    def test_grading_model_called_with_record(self):
        grading_model = _make_mock_grading_model("correct")
        pipeline = HybridFeedbackPipeline(
            grading_model=grading_model,
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
        )

        record = _make_record()
        pipeline.run(record)

        grading_model.predict.assert_called_once_with([record])

    def test_concept_gap_detector_called(self):
        detector = _make_mock_concept_gap_detector()
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=detector,
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
        )

        record = _make_record()
        pipeline.run(record)

        detector.detect.assert_called_once_with(
            question=record.question,
            reference_answer=record.reference_answer,
            student_answer=record.student_answer,
            key_concepts=list(record.key_concepts),
        )

    def test_t5_generator_called_with_gap_and_label(self):
        t5_gen = _make_mock_t5_generator()
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model("partially_correct"),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=t5_gen,
            consistency_checker=_make_mock_consistency_checker(),
        )

        record = _make_record()
        pipeline.run(record)

        t5_gen.generate_with_metadata.assert_called_once()
        call_args = t5_gen.generate_with_metadata.call_args
        assert call_args[0][0] is record  # record
        assert isinstance(call_args[0][1], ConceptGapResult)  # gap_result
        assert call_args[0][2] == "partially_correct"  # predicted_label

    def test_consistency_checker_called_with_feedback(self):
        checker = _make_mock_consistency_checker()
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(
                feedback_detailed="T5 feedback text.",
            ),
            consistency_checker=checker,
        )

        record = _make_record()
        pipeline.run(record)

        checker.check.assert_called_once_with(
            generated_feedback="T5 feedback text.",
            reference_answer=record.reference_answer,
        )


# ------------------------------------------------------------------
# Test: Batch processing
# ------------------------------------------------------------------


class TestBatchProcessing:
    """Verify run_batch processes multiple records."""

    def test_run_batch_returns_list(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
        )

        records = [
            _make_record(sample_id=f"GEN_{i:04d}")
            for i in range(3)
        ]
        results = pipeline.run_batch(records)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, HybridFeedbackResult)

    def test_run_batch_empty(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
        )

        results = pipeline.run_batch([])
        assert results == []


# ------------------------------------------------------------------
# Test: Consistency threshold configuration
# ------------------------------------------------------------------


class TestConsistencyThreshold:
    """Verify threshold is configurable and affects fallback behavior."""

    def test_default_threshold(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
        )
        assert pipeline.consistency_threshold == 0.5

    def test_custom_threshold(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
            consistency_threshold=0.8,
        )
        assert pipeline.consistency_threshold == 0.8

    def test_high_threshold_triggers_more_fallbacks(self):
        """With threshold=0.95, a score of 0.9 should trigger fallback."""
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(
                consistency_score=0.9,
            ),
            consistency_threshold=0.95,
        )

        result = pipeline.run(_make_record())
        assert result.used_fallback is True

    def test_low_threshold_accepts_more(self):
        """With threshold=0.1, a score of 0.3 should NOT trigger fallback."""
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(
                consistency_score=0.3,
            ),
            consistency_threshold=0.1,
        )

        result = pipeline.run(_make_record())
        assert result.used_fallback is False

    def test_threshold_setter(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=_make_mock_concept_gap_detector(),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
        )
        pipeline.consistency_threshold = 0.75
        assert pipeline.consistency_threshold == 0.75


# ------------------------------------------------------------------
# Test: Different grading labels flow through pipeline
# ------------------------------------------------------------------


class TestGradingLabelPropagation:
    """Verify different grading labels produce appropriate feedback."""

    def test_correct_label_flows_through(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model("correct"),
            concept_gap_detector=_make_mock_concept_gap_detector(
                present=["photosynthesis", "light energy", "chemical energy"],
                missing=[],
            ),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
        )

        result = pipeline.run(_make_record())
        assert result.predicted_label == "correct"

    def test_incorrect_label_flows_through(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model("incorrect"),
            concept_gap_detector=_make_mock_concept_gap_detector(
                missing=["photosynthesis", "light energy", "chemical energy"],
            ),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
        )

        result = pipeline.run(_make_record())
        assert result.predicted_label == "incorrect"


# ------------------------------------------------------------------
# Test: Gap result is populated in output
# ------------------------------------------------------------------


class TestGapResultInOutput:
    """Verify the concept gap result is included in the pipeline output."""

    def test_gap_result_present(self):
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=_make_mock_concept_gap_detector(
                present=["photosynthesis"],
                missing=["chemical energy"],
                contradicted=["light energy"],
            ),
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
        )

        result = pipeline.run(_make_record())
        assert result.gap_result is not None
        assert "photosynthesis" in result.gap_result.present_concepts
        assert "chemical energy" in result.gap_result.missing_concepts
        assert "light energy" in result.gap_result.contradicted_concepts


# ------------------------------------------------------------------
# Test: Record with no key_concepts uses fallback extraction
# ------------------------------------------------------------------


class TestNoKeyConcepts:
    """When record has no key_concepts, detector should receive None."""

    def test_empty_key_concepts_passes_none(self):
        detector = _make_mock_concept_gap_detector()
        pipeline = HybridFeedbackPipeline(
            grading_model=_make_mock_grading_model(),
            concept_gap_detector=detector,
            generative_generator=_make_mock_t5_generator(),
            consistency_checker=_make_mock_consistency_checker(),
        )

        record = _make_record(key_concepts=[])
        pipeline.run(record)

        detector.detect.assert_called_once()
        call_kwargs = detector.detect.call_args[1]
        assert call_kwargs["key_concepts"] is None
