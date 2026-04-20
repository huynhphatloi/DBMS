"""Tests for the Template-Based Feedback Generator.

Covers:
- Unit tests for all three label cases (correct, partially_correct, incorrect)
- Verification that both feedback_short and feedback_detailed are produced
- Edge cases: empty concept lists, contradicted concepts

Validates: Requirement 18
"""

from __future__ import annotations

from src.data.schema import UnifiedRecord
from src.feedback.concept_gap import ConceptGapResult
from src.feedback.template import FeedbackGenerator, TemplateFeedbackGenerator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_record(**overrides) -> UnifiedRecord:
    """Create a minimal UnifiedRecord with sensible defaults."""
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
    )
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


# ------------------------------------------------------------------
# Basic interface tests
# ------------------------------------------------------------------


class TestTemplateFeedbackGeneratorInterface:
    """Verify the generator implements the abstract interface."""

    def test_is_subclass_of_feedback_generator(self):
        assert issubclass(TemplateFeedbackGenerator, FeedbackGenerator)

    def test_returns_tuple_of_two_strings(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            present_concepts=["photosynthesis"],
            missing_concepts=[],
            contradicted_concepts=[],
        )
        result = gen.generate(record, gap, "correct")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)


# ------------------------------------------------------------------
# Correct label tests
# ------------------------------------------------------------------


class TestCorrectLabel:
    """Requirement 18.1: correct → praise referencing the question topic."""

    def test_short_feedback_mentions_correct(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            present_concepts=["photosynthesis", "light energy"],
            missing_concepts=[],
            contradicted_concepts=[],
        )
        short, _ = gen.generate(record, gap, "correct")
        assert "correct" in short.lower()

    def test_short_feedback_references_topic(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record(question="Explain the water cycle.")
        gap = ConceptGapResult(present_concepts=["evaporation"])
        short, _ = gen.generate(record, gap, "correct")
        assert "water cycle" in short.lower()

    def test_detailed_feedback_lists_present_concepts(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            present_concepts=["photosynthesis", "light energy"],
        )
        _, detailed = gen.generate(record, gap, "correct")
        assert "photosynthesis" in detailed
        assert "light energy" in detailed

    def test_detailed_feedback_without_present_concepts(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult()
        _, detailed = gen.generate(record, gap, "correct")
        # Should still produce a valid paragraph
        assert len(detailed) > 20

    def test_both_outputs_are_nonempty(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(present_concepts=["photosynthesis"])
        short, detailed = gen.generate(record, gap, "correct")
        assert len(short) > 0
        assert len(detailed) > len(short)


# ------------------------------------------------------------------
# Partially correct label tests
# ------------------------------------------------------------------


class TestPartiallyCorrectLabel:
    """Requirement 18.2: partially_correct → list present + missing concepts."""

    def test_short_feedback_mentions_partially(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            present_concepts=["photosynthesis"],
            missing_concepts=["chemical energy"],
        )
        short, _ = gen.generate(record, gap, "partially_correct")
        assert "partially correct" in short.lower()

    def test_detailed_lists_present_concepts(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            present_concepts=["photosynthesis"],
            missing_concepts=["chemical energy"],
        )
        _, detailed = gen.generate(record, gap, "partially_correct")
        assert "photosynthesis" in detailed

    def test_detailed_lists_missing_concepts(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            present_concepts=["photosynthesis"],
            missing_concepts=["chemical energy", "light energy"],
        )
        _, detailed = gen.generate(record, gap, "partially_correct")
        assert "chemical energy" in detailed
        assert "light energy" in detailed

    def test_detailed_lists_contradicted_concepts(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            present_concepts=["photosynthesis"],
            missing_concepts=[],
            contradicted_concepts=["oxygen production"],
        )
        _, detailed = gen.generate(record, gap, "partially_correct")
        assert "oxygen production" in detailed

    def test_only_missing_no_present(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            present_concepts=[],
            missing_concepts=["chemical energy"],
        )
        short, detailed = gen.generate(record, gap, "partially_correct")
        assert "partially correct" in short.lower()
        assert "chemical energy" in detailed

    def test_both_outputs_are_nonempty(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            present_concepts=["photosynthesis"],
            missing_concepts=["chemical energy"],
        )
        short, detailed = gen.generate(record, gap, "partially_correct")
        assert len(short) > 0
        assert len(detailed) > len(short)


# ------------------------------------------------------------------
# Incorrect label tests
# ------------------------------------------------------------------


class TestIncorrectLabel:
    """Requirement 18.3: incorrect → list concepts to review."""

    def test_short_feedback_mentions_incorrect(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            missing_concepts=["photosynthesis", "light energy"],
        )
        short, _ = gen.generate(record, gap, "incorrect")
        assert "incorrect" in short.lower()

    def test_detailed_lists_concepts_to_review(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            missing_concepts=["photosynthesis", "light energy"],
        )
        _, detailed = gen.generate(record, gap, "incorrect")
        assert "photosynthesis" in detailed
        assert "light energy" in detailed

    def test_detailed_mentions_contradicted_concepts(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            missing_concepts=[],
            contradicted_concepts=["oxygen production"],
        )
        _, detailed = gen.generate(record, gap, "incorrect")
        assert "oxygen production" in detailed

    def test_fallback_to_record_key_concepts(self):
        """When gap_result has no missing/contradicted, use record.key_concepts."""
        gen = TemplateFeedbackGenerator()
        record = _make_record(
            key_concepts=["photosynthesis", "chloroplast"],
        )
        gap = ConceptGapResult()  # empty
        _, detailed = gen.generate(record, gap, "incorrect")
        assert "photosynthesis" in detailed
        assert "chloroplast" in detailed

    def test_no_concepts_at_all(self):
        """When neither gap_result nor record has concepts."""
        gen = TemplateFeedbackGenerator()
        record = _make_record(key_concepts=[])
        gap = ConceptGapResult()
        short, detailed = gen.generate(record, gap, "incorrect")
        assert "incorrect" in short.lower()
        assert len(detailed) > 20

    def test_both_outputs_are_nonempty(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            missing_concepts=["photosynthesis"],
        )
        short, detailed = gen.generate(record, gap, "incorrect")
        assert len(short) > 0
        assert len(detailed) > len(short)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and label normalization."""

    def test_label_case_insensitive(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(present_concepts=["photosynthesis"])

        short_lower, _ = gen.generate(record, gap, "correct")
        short_upper, _ = gen.generate(record, gap, "CORRECT")
        short_mixed, _ = gen.generate(record, gap, "Correct")

        assert "correct" in short_lower.lower()
        assert "correct" in short_upper.lower()
        assert "correct" in short_mixed.lower()

    def test_label_with_whitespace(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(present_concepts=["photosynthesis"])
        short, _ = gen.generate(record, gap, "  correct  ")
        assert "correct" in short.lower()

    def test_unknown_label_treated_as_incorrect(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(missing_concepts=["photosynthesis"])
        short, _ = gen.generate(record, gap, "unknown_label")
        assert "incorrect" in short.lower()

    def test_detailed_is_longer_than_short_for_all_labels(self):
        gen = TemplateFeedbackGenerator()
        record = _make_record()
        gap = ConceptGapResult(
            present_concepts=["photosynthesis"],
            missing_concepts=["chemical energy"],
        )
        for label in ("correct", "partially_correct", "incorrect"):
            short, detailed = gen.generate(record, gap, label)
            assert len(detailed) >= len(short), (
                f"For label={label!r}, detailed should be >= short"
            )
