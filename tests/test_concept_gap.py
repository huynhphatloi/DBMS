"""Tests for the Concept Gap Detector module.

Covers:
- Unit tests for all three classification outcomes (present/missing/contradicted)
- Noun-phrase extraction fallback
- Property-based test: union of present + missing + contradicted == input key_concepts
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.feedback.concept_gap import (
    ConceptGapDetector,
    ConceptGapResult,
    extract_noun_phrases,
    _map_nli_label,
)


# ------------------------------------------------------------------
# Helpers — mock NLI pipeline
# ------------------------------------------------------------------


def _make_mock_pipeline(label_map: dict[str, str]):
    """Return a callable that mimics a HuggingFace text-classification pipeline.

    Parameters
    ----------
    label_map : dict[str, str]
        Maps concept substrings to NLI labels.
        E.g. ``{"photosynthesis": "entailment", "mitosis": "neutral"}``
    """

    def _mock_call(inputs, top_k=1):
        hypothesis_text = inputs.get("text_pair", "")
        for keyword, label in label_map.items():
            if keyword.lower() in hypothesis_text.lower():
                return [{"label": label, "score": 0.95}]
        # Default: neutral (missing)
        return [{"label": "neutral", "score": 0.5}]

    mock = MagicMock(side_effect=_mock_call)
    return mock


# ------------------------------------------------------------------
# Unit tests for NLI label mapping
# ------------------------------------------------------------------


class TestNLILabelMapping:
    """Tests for _map_nli_label helper."""

    def test_entailment_labels(self):
        assert _map_nli_label("entailment") == "present"
        assert _map_nli_label("ENTAILMENT") == "present"
        assert _map_nli_label("LABEL_0") == "present"

    def test_contradiction_labels(self):
        assert _map_nli_label("contradiction") == "contradicted"
        assert _map_nli_label("CONTRADICTION") == "contradicted"
        assert _map_nli_label("LABEL_2") == "contradicted"

    def test_neutral_labels(self):
        assert _map_nli_label("neutral") == "missing"
        assert _map_nli_label("NEUTRAL") == "missing"
        assert _map_nli_label("LABEL_1") == "missing"

    def test_unknown_label_defaults_to_missing(self):
        assert _map_nli_label("something_else") == "missing"


# ------------------------------------------------------------------
# Unit tests for noun-phrase extraction fallback
# ------------------------------------------------------------------


class TestExtractNounPhrases:
    """Tests for extract_noun_phrases."""

    def test_empty_string(self):
        assert extract_noun_phrases("") == []

    def test_whitespace_only(self):
        assert extract_noun_phrases("   ") == []

    def test_extracts_content_words(self):
        result = extract_noun_phrases(
            "The process of photosynthesis converts light energy"
        )
        assert len(result) > 0
        # Should contain meaningful phrases, not stop words
        for phrase in result:
            assert len(phrase) >= 3

    def test_deduplication(self):
        result = extract_noun_phrases(
            "energy and energy and more energy"
        )
        assert result.count("energy") <= 1

    def test_returns_lowercased(self):
        result = extract_noun_phrases("Photosynthesis and Respiration")
        for phrase in result:
            assert phrase == phrase.lower()


# ------------------------------------------------------------------
# Unit tests for ConceptGapDetector — all three outcomes
# ------------------------------------------------------------------


class TestConceptGapDetectorPresent:
    """Test that concepts classified as entailed are marked present."""

    def test_single_present_concept(self):
        mock_pipe = _make_mock_pipeline(
            {"photosynthesis": "entailment"}
        )
        detector = ConceptGapDetector(_pipeline=mock_pipe)

        result = detector.detect(
            question="What is photosynthesis?",
            reference_answer="Photosynthesis converts light to energy.",
            student_answer="Photosynthesis uses sunlight to make food.",
            key_concepts=["photosynthesis"],
        )

        assert result.present_concepts == ["photosynthesis"]
        assert result.missing_concepts == []
        assert result.contradicted_concepts == []

    def test_multiple_present_concepts(self):
        mock_pipe = _make_mock_pipeline(
            {"light": "entailment", "energy": "entailment"}
        )
        detector = ConceptGapDetector(_pipeline=mock_pipe)

        result = detector.detect(
            question="Explain energy conversion.",
            reference_answer="Light energy is converted.",
            student_answer="Light produces energy.",
            key_concepts=["light", "energy"],
        )

        assert set(result.present_concepts) == {"light", "energy"}
        assert result.missing_concepts == []
        assert result.contradicted_concepts == []


class TestConceptGapDetectorMissing:
    """Test that concepts classified as neutral are marked missing."""

    def test_single_missing_concept(self):
        mock_pipe = _make_mock_pipeline(
            {"chloroplast": "neutral"}
        )
        detector = ConceptGapDetector(_pipeline=mock_pipe)

        result = detector.detect(
            question="Where does photosynthesis occur?",
            reference_answer="In the chloroplast.",
            student_answer="In the leaf.",
            key_concepts=["chloroplast"],
        )

        assert result.present_concepts == []
        assert result.missing_concepts == ["chloroplast"]
        assert result.contradicted_concepts == []


class TestConceptGapDetectorContradicted:
    """Test that concepts classified as contradiction are marked contradicted."""

    def test_single_contradicted_concept(self):
        mock_pipe = _make_mock_pipeline(
            {"oxygen": "contradiction"}
        )
        detector = ConceptGapDetector(_pipeline=mock_pipe)

        result = detector.detect(
            question="What does photosynthesis produce?",
            reference_answer="Photosynthesis produces oxygen.",
            student_answer="Photosynthesis does not produce oxygen.",
            key_concepts=["oxygen"],
        )

        assert result.present_concepts == []
        assert result.missing_concepts == []
        assert result.contradicted_concepts == ["oxygen"]


class TestConceptGapDetectorMixed:
    """Test mixed classification outcomes."""

    def test_all_three_categories(self):
        mock_pipe = _make_mock_pipeline({
            "light": "entailment",
            "chlorophyll": "neutral",
            "carbon dioxide": "contradiction",
        })
        detector = ConceptGapDetector(_pipeline=mock_pipe)

        result = detector.detect(
            question="Explain photosynthesis.",
            reference_answer="Uses light, chlorophyll, and carbon dioxide.",
            student_answer="Uses light but not carbon dioxide.",
            key_concepts=["light", "chlorophyll", "carbon dioxide"],
        )

        assert result.present_concepts == ["light"]
        assert result.missing_concepts == ["chlorophyll"]
        assert result.contradicted_concepts == ["carbon dioxide"]


# ------------------------------------------------------------------
# Unit tests for fallback behaviour
# ------------------------------------------------------------------


class TestConceptGapDetectorFallback:
    """Test noun-phrase extraction fallback when key_concepts is empty."""

    def test_empty_key_concepts_triggers_fallback(self):
        # The mock will classify everything as neutral (missing)
        mock_pipe = _make_mock_pipeline({})
        detector = ConceptGapDetector(_pipeline=mock_pipe)

        result = detector.detect(
            question="What is photosynthesis?",
            reference_answer="Photosynthesis converts light energy.",
            student_answer="Plants make food.",
            key_concepts=[],
        )

        # Fallback should have extracted concepts from reference_answer
        total = (
            len(result.present_concepts)
            + len(result.missing_concepts)
            + len(result.contradicted_concepts)
        )
        assert total > 0

    def test_none_key_concepts_triggers_fallback(self):
        mock_pipe = _make_mock_pipeline({})
        detector = ConceptGapDetector(_pipeline=mock_pipe)

        result = detector.detect(
            question="What is photosynthesis?",
            reference_answer="Photosynthesis converts light energy.",
            student_answer="Plants make food.",
            key_concepts=None,
        )

        total = (
            len(result.present_concepts)
            + len(result.missing_concepts)
            + len(result.contradicted_concepts)
        )
        assert total > 0

    def test_empty_reference_and_empty_concepts(self):
        """When both key_concepts and reference_answer are empty."""
        mock_pipe = _make_mock_pipeline({})
        detector = ConceptGapDetector(_pipeline=mock_pipe)

        result = detector.detect(
            question="Q",
            reference_answer="",
            student_answer="Some answer.",
            key_concepts=[],
        )

        assert result == ConceptGapResult()


# ------------------------------------------------------------------
# Property-based test — Property 10
# ------------------------------------------------------------------


# Strategy: generate non-empty lists of unique concept strings
_concept_strategy = st.lists(
    st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz "),
        min_size=3,
        max_size=30,
    ).map(str.strip).filter(lambda s: len(s) >= 3),
    min_size=1,
    max_size=10,
    unique=True,
)

# Strategy: generate non-empty text strings
_text_strategy = st.text(
    alphabet=st.sampled_from(
        "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ),
    min_size=1,
    max_size=100,
)


def _make_random_classifier_pipeline():
    """Pipeline mock that randomly assigns entailment/neutral/contradiction."""
    import random

    labels = ["entailment", "neutral", "contradiction"]

    def _call(inputs, top_k=1):
        label = random.choice(labels)
        return [{"label": label, "score": 0.9}]

    return MagicMock(side_effect=_call)


@given(
    question=_text_strategy,
    reference=_text_strategy,
    answer=_text_strategy,
    key_concepts=_concept_strategy,
)
@settings(max_examples=100, deadline=None)
def test_concept_gap_completeness_property(
    question: str,
    reference: str,
    answer: str,
    key_concepts: list[str],
):
    """Property 10: union of present + missing + contradicted == input key_concepts.

    **Validates: Requirements 17.1, 17.3**

    For any (question, reference_answer, student_answer, key_concepts) tuple,
    the union of present_concepts, missing_concepts, and contradicted_concepts
    returned by the Concept_Gap_Detector SHALL equal the input key_concepts
    list (no concept is lost or duplicated).
    """
    # Feature: asag-research-framework, Property 10: Concept Gap Completeness
    mock_pipe = _make_random_classifier_pipeline()
    detector = ConceptGapDetector(_pipeline=mock_pipe)

    result = detector.detect(
        question=question,
        reference_answer=reference,
        student_answer=answer,
        key_concepts=key_concepts,
    )

    # The union of all three lists must equal the input key_concepts
    output_concepts = (
        result.present_concepts
        + result.missing_concepts
        + result.contradicted_concepts
    )

    # Same elements, same count (no loss, no duplication)
    assert sorted(output_concepts) == sorted(key_concepts), (
        f"Concept mismatch!\n"
        f"  Input:  {sorted(key_concepts)}\n"
        f"  Output: {sorted(output_concepts)}"
    )

    # No concept appears in more than one category
    all_sets = [
        set(result.present_concepts),
        set(result.missing_concepts),
        set(result.contradicted_concepts),
    ]
    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            overlap = all_sets[i] & all_sets[j]
            assert not overlap, (
                f"Concept(s) {overlap} appear in multiple categories"
            )
