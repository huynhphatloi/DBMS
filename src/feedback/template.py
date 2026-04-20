"""Template-based feedback generation — deterministic rule-based baseline.

Correct → praise referencing the question topic.
Partially correct → list present + missing concepts.
Incorrect → list key concepts the student should review.

Produces both ``feedback_short`` (1–2 sentences) and ``feedback_detailed``
(paragraph) outputs for each record.

Validates: Requirement 18
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.schema import UnifiedRecord
from src.feedback.concept_gap import ConceptGapResult


class FeedbackGenerator(ABC):
    """Abstract base class for all feedback strategies."""

    @abstractmethod
    def generate(
        self,
        record: UnifiedRecord,
        gap_result: ConceptGapResult,
        predicted_label: str,
    ) -> tuple[str, str]:
        """Generate feedback for a student answer.

        Parameters
        ----------
        record : UnifiedRecord
            The unified record containing question, reference, and student answer.
        gap_result : ConceptGapResult
            The concept gap analysis result.
        predicted_label : str
            The predicted grading label (``"correct"``, ``"partially_correct"``,
            or ``"incorrect"``).

        Returns
        -------
        tuple[str, str]
            ``(feedback_short, feedback_detailed)``
        """
        ...


class TemplateFeedbackGenerator(FeedbackGenerator):
    """Deterministic template-based feedback generator.

    Rules
    -----
    * **correct** – praise the student and reference the question topic.
    * **partially_correct** – acknowledge present concepts, list missing ones.
    * **incorrect** – list the key concepts the student should review.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        record: UnifiedRecord,
        gap_result: ConceptGapResult,
        predicted_label: str,
    ) -> tuple[str, str]:
        label = predicted_label.lower().strip()

        if label == "correct":
            return self._correct(record, gap_result)
        if label == "partially_correct":
            return self._partially_correct(record, gap_result)
        # Default: treat anything else (including "incorrect") as incorrect.
        return self._incorrect(record, gap_result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _topic(record: UnifiedRecord) -> str:
        """Extract a short topic string from the question."""
        q = record.question.strip()
        # Use the first sentence or the whole question if short.
        first_sentence = q.split(".")[0].split("?")[0].strip()
        return first_sentence if first_sentence else "this topic"

    # -- correct -----------------------------------------------------------

    def _correct(
        self,
        record: UnifiedRecord,
        gap_result: ConceptGapResult,
    ) -> tuple[str, str]:
        topic = self._topic(record)

        feedback_short = (
            f"Great work! Your answer about {topic} is correct."
        )

        present = gap_result.present_concepts
        if present:
            concept_list = ", ".join(present)
            feedback_detailed = (
                f"Excellent job! Your answer correctly addresses {topic}. "
                f"You demonstrated a solid understanding of the following "
                f"key concepts: {concept_list}. Keep up the good work!"
            )
        else:
            feedback_detailed = (
                f"Excellent job! Your answer correctly addresses {topic}. "
                f"You demonstrated a solid understanding of the material. "
                f"Keep up the good work!"
            )

        return feedback_short, feedback_detailed

    # -- partially_correct -------------------------------------------------

    def _partially_correct(
        self,
        record: UnifiedRecord,
        gap_result: ConceptGapResult,
    ) -> tuple[str, str]:
        topic = self._topic(record)
        present = gap_result.present_concepts
        missing = gap_result.missing_concepts
        contradicted = gap_result.contradicted_concepts

        # -- short ----------------------------------------------------------
        if present and missing:
            feedback_short = (
                f"Your answer about {topic} is partially correct. "
                f"You covered some concepts but missed others."
            )
        elif missing:
            feedback_short = (
                f"Your answer about {topic} is partially correct, "
                f"but key concepts are missing."
            )
        else:
            feedback_short = (
                f"Your answer about {topic} is partially correct."
            )

        # -- detailed -------------------------------------------------------
        parts: list[str] = [
            f"Your answer about {topic} is on the right track."
        ]

        if present:
            parts.append(
                f"You correctly addressed: {', '.join(present)}."
            )

        if missing:
            parts.append(
                f"However, you missed the following concepts: "
                f"{', '.join(missing)}. "
                f"Please review these areas to strengthen your answer."
            )

        if contradicted:
            parts.append(
                f"Additionally, your answer contains incorrect claims "
                f"about: {', '.join(contradicted)}. "
                f"Please revisit these concepts carefully."
            )

        feedback_detailed = " ".join(parts)
        return feedback_short, feedback_detailed

    # -- incorrect ---------------------------------------------------------

    def _incorrect(
        self,
        record: UnifiedRecord,
        gap_result: ConceptGapResult,
    ) -> tuple[str, str]:
        topic = self._topic(record)
        missing = gap_result.missing_concepts
        contradicted = gap_result.contradicted_concepts
        # Combine key_concepts from the record as a fallback review list.
        review_concepts = missing + contradicted
        if not review_concepts and record.key_concepts:
            review_concepts = list(record.key_concepts)

        # -- short ----------------------------------------------------------
        if review_concepts:
            feedback_short = (
                f"Your answer about {topic} is incorrect. "
                f"Please review the key concepts for this question."
            )
        else:
            feedback_short = (
                f"Your answer about {topic} is incorrect. "
                f"Please revisit the material and try again."
            )

        # -- detailed -------------------------------------------------------
        parts: list[str] = [
            f"Your answer about {topic} does not correctly address "
            f"the question."
        ]

        if contradicted:
            parts.append(
                f"Your answer contains incorrect claims about: "
                f"{', '.join(contradicted)}."
            )

        if review_concepts:
            parts.append(
                f"You should review the following concepts: "
                f"{', '.join(review_concepts)}. "
                f"Revisiting the reference material on these topics "
                f"will help you build a stronger understanding."
            )
        else:
            parts.append(
                "Please revisit the reference material for this topic "
                "and try again."
            )

        feedback_detailed = " ".join(parts)
        return feedback_short, feedback_detailed
