"""Tests for the Perturbation Engine.

Covers:
  - Unit tests for each perturbation type (28.3)
  - Property-based test: copy-reference → "correct" (Property 7, 28.4)
  - Property-based test: empty-answer → "incorrect" (Property 8, 28.5)
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.schema import UnifiedRecord
from src.grading.baselines.lexical import LexicalThresholdClassifier
from src.robustness.perturbations import (
    VALID_PERTURBATION_TYPES,
    PerturbationEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    sample_id: str = "TEST_001",
    question: str = "What is photosynthesis?",
    reference_answer: str = "Photosynthesis is the process by which plants convert sunlight into energy.",
    student_answer: str = "Plants use sunlight to make food.",
    key_concepts: list[str] | None = None,
) -> UnifiedRecord:
    return UnifiedRecord(
        sample_id=sample_id,
        source_dataset="scientsbank",
        original_id="orig_001",
        question_id="q_001",
        domain="science",
        subdomain="biology",
        difficulty="medium",
        question=question,
        reference_answer=reference_answer,
        student_answer=student_answer,
        key_concepts=key_concepts if key_concepts is not None else ["photosynthesis", "sunlight", "energy"],
    )


# ---------------------------------------------------------------------------
# Unit tests — each perturbation type (28.3)
# ---------------------------------------------------------------------------

class TestKeywordStuffing:
    def test_appends_key_concepts(self):
        rec = _make_record()
        engine = PerturbationEngine(seed=42)
        perturbed = engine.perturb(rec, "keyword_stuffing")

        assert perturbed.student_answer.startswith(rec.student_answer)
        for concept in rec.key_concepts:
            assert concept in perturbed.student_answer

    def test_no_concepts_returns_original(self):
        rec = _make_record(key_concepts=[])
        engine = PerturbationEngine(seed=42)
        perturbed = engine.perturb(rec, "keyword_stuffing")

        assert perturbed.student_answer == rec.student_answer


class TestVerbosityAttack:
    def test_pads_with_filler(self):
        rec = _make_record()
        engine = PerturbationEngine(seed=42)
        perturbed = engine.perturb(rec, "verbosity_attack")

        assert perturbed.student_answer.startswith(rec.student_answer)
        assert len(perturbed.student_answer) > len(rec.student_answer) * 3

    def test_multiplier_controls_length(self):
        rec = _make_record()
        e1 = PerturbationEngine(verbosity_multiplier=1, seed=42)
        e2 = PerturbationEngine(verbosity_multiplier=5, seed=42)
        p1 = e1.perturb(rec, "verbosity_attack")
        p2 = e2.perturb(rec, "verbosity_attack")

        assert len(p2.student_answer) > len(p1.student_answer)


class TestCopyReference:
    def test_replaces_with_reference(self):
        rec = _make_record()
        engine = PerturbationEngine(seed=42)
        perturbed = engine.perturb(rec, "copy_reference")

        assert perturbed.student_answer == rec.reference_answer


class TestEmptyAnswer:
    def test_replaces_with_empty(self):
        rec = _make_record()
        engine = PerturbationEngine(seed=42)
        perturbed = engine.perturb(rec, "empty_answer")

        assert perturbed.student_answer == ""


class TestRandomText:
    def test_replaces_with_random_tokens(self):
        rec = _make_record()
        engine = PerturbationEngine(random_text_tokens=50, seed=42)
        perturbed = engine.perturb(rec, "random_text")

        tokens = perturbed.student_answer.split()
        assert len(tokens) == 50
        assert perturbed.student_answer != rec.student_answer

    def test_seed_reproducibility(self):
        rec = _make_record()
        e1 = PerturbationEngine(seed=123)
        e2 = PerturbationEngine(seed=123)
        p1 = e1.perturb(rec, "random_text")
        p2 = e2.perturb(rec, "random_text")

        assert p1.student_answer == p2.student_answer


# ---------------------------------------------------------------------------
# Metadata tests (28.2)
# ---------------------------------------------------------------------------

class TestPerturbedRecordMetadata:
    @pytest.mark.parametrize("ptype", sorted(VALID_PERTURBATION_TYPES))
    def test_metadata_fields(self, ptype: str):
        rec = _make_record(sample_id="SEB_001")
        engine = PerturbationEngine(seed=42)
        perturbed = engine.perturb(rec, ptype)

        assert perturbed.perturbation_type == ptype
        assert perturbed.adversarial_variant_of == "SEB_001"
        assert perturbed.is_adversarial is True
        assert perturbed.sample_id == f"SEB_001_{ptype}"

    def test_invalid_perturbation_type_raises(self):
        rec = _make_record()
        engine = PerturbationEngine(seed=42)
        with pytest.raises(ValueError, match="perturbation_type must be one of"):
            engine.perturb(rec, "nonexistent")

    def test_perturb_all_returns_all_types(self):
        rec = _make_record()
        engine = PerturbationEngine(seed=42)
        results = engine.perturb_all(rec)

        assert len(results) == len(VALID_PERTURBATION_TYPES)
        types = {r.perturbation_type for r in results}
        assert types == VALID_PERTURBATION_TYPES


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Build multi-word text by joining random ASCII words.
# The lexical tokenizer uses [a-z0-9]+ so we must generate ASCII text.
_word = st.from_regex(r"[a-z]{2,10}", fullmatch=True)

_non_empty_text = st.lists(_word, min_size=3, max_size=15).map(" ".join)


def _record_strategy():
    """Generate random (question, reference_answer, student_answer) tuples."""
    return st.fixed_dictionaries({
        "question": _non_empty_text,
        "reference_answer": _non_empty_text,
        "student_answer": _non_empty_text,
    })


# ---------------------------------------------------------------------------
# Property 7: copy-reference → "correct" (28.4)
# ---------------------------------------------------------------------------

# Feature: asag-research-framework, Property 7: For any grading model and any
# student answer, replacing the student answer with the verbatim
# reference_answer SHALL produce a predicted label of "correct" or the
# maximum possible score.

class TestProperty7CopyReferenceUpperBound:
    """**Validates: Requirements 23.4**"""

    @given(data=_record_strategy())
    @settings(max_examples=100)
    def test_copy_reference_predicts_correct(self, data: dict):
        """Copy-reference perturbation must yield 'correct' from lexical model."""
        rec = UnifiedRecord(
            sample_id="PROP7_001",
            source_dataset="scientsbank",
            original_id="orig",
            question_id="q",
            domain="science",
            subdomain="biology",
            difficulty="medium",
            question=data["question"],
            reference_answer=data["reference_answer"],
            student_answer=data["student_answer"],
        )

        engine = PerturbationEngine(seed=42)
        perturbed = engine.perturb(rec, "copy_reference")

        # The student answer is now the reference answer verbatim, so
        # lexical overlap should be 1.0 → threshold classifier predicts "correct".
        model = LexicalThresholdClassifier(
            metric="rouge_l", threshold=0.5,
            positive_label="correct", negative_label="incorrect",
        )
        [prediction] = model.predict([perturbed])
        assert prediction == "correct"


# ---------------------------------------------------------------------------
# Property 8: empty-answer → "incorrect" (28.5)
# ---------------------------------------------------------------------------

# Feature: asag-research-framework, Property 8: For any grading model and any
# student answer, replacing the student answer with an empty string SHALL
# produce a predicted label of "incorrect" or the minimum possible score.

class TestProperty8EmptyAnswerLowerBound:
    """**Validates: Requirements 23.4**"""

    @given(data=_record_strategy())
    @settings(max_examples=100)
    def test_empty_answer_predicts_incorrect(self, data: dict):
        """Empty-answer perturbation must yield 'incorrect' from lexical model."""
        rec = UnifiedRecord(
            sample_id="PROP8_001",
            source_dataset="scientsbank",
            original_id="orig",
            question_id="q",
            domain="science",
            subdomain="biology",
            difficulty="medium",
            question=data["question"],
            reference_answer=data["reference_answer"],
            student_answer=data["student_answer"],
        )

        engine = PerturbationEngine(seed=42)
        perturbed = engine.perturb(rec, "empty_answer")

        # The student answer is empty → all lexical metrics are 0.0 → "incorrect".
        model = LexicalThresholdClassifier(
            metric="rouge_l", threshold=0.5,
            positive_label="correct", negative_label="incorrect",
        )
        [prediction] = model.predict([perturbed])
        assert prediction == "incorrect"
