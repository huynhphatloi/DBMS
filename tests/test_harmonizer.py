"""Tests for the LabelHarmonizer.

Covers:
- 4.5: Example-based unit tests for SciEntsBank 5-way values
        and MohlerASAG boundary scores (0, 1, 2.5, 4, 5)
- 4.6: Property 2 — score_normalized ∈ [0.0, 1.0]
- 4.7: Property 1 — score_raw=5 → correct, score_raw=0 → incorrect
"""

from __future__ import annotations

import logging

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.harmonizer import LabelHarmonizer
from src.data.schema import UnifiedRecord


# ── Helpers ───────────────────────────────────────────────────────────


def _mohler_record(**overrides) -> UnifiedRecord:
    """Create a minimal MohlerASAG record."""
    defaults = dict(
        sample_id="MOH_00001",
        source_dataset="mohler",
        original_id="m1",
        question_id="q1",
        domain="computer_science",
        subdomain="general",
        difficulty="unknown",
        question="What is a pointer?",
        reference_answer="A variable that stores a memory address.",
        student_answer="It points to memory.",
    )
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


def _scientsbank_record(**overrides) -> UnifiedRecord:
    """Create a minimal SciEntsBank record."""
    defaults = dict(
        sample_id="SEB_00001",
        source_dataset="scientsbank",
        original_id="s1",
        question_id="q1",
        domain="science",
        subdomain="biology",
        difficulty="unknown",
        question="What is photosynthesis?",
        reference_answer="Plants convert light to energy.",
        student_answer="Plants use sunlight.",
    )
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


def _data_generate_record(**overrides) -> UnifiedRecord:
    """Create a minimal Data_Generate record."""
    defaults = dict(
        sample_id="GEN_00001",
        source_dataset="data_generate",
        original_id="g1",
        question_id="q1",
        domain="science",
        subdomain="biology",
        difficulty="unknown",
        question="What is photosynthesis?",
        reference_answer="Plants convert light to energy.",
        student_answer="Plants use sunlight.",
    )
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


# ── 4.5: Example-based unit tests ────────────────────────────────────


class TestMohlerBoundaryScores:
    """MohlerASAG score→label mapping at boundary values."""

    def setup_method(self):
        self.h = LabelHarmonizer()

    def test_score_0(self):
        rec = _mohler_record(score_raw=0.0)
        self.h.harmonize(rec)
        assert rec.score_normalized == 0.0
        assert rec.label_2way == "incorrect"
        assert rec.label_3way == "incorrect"

    def test_score_1(self):
        rec = _mohler_record(score_raw=1.0)
        self.h.harmonize(rec)
        assert rec.score_normalized == pytest.approx(0.2)
        assert rec.label_2way == "incorrect"
        # [1, 4) → partially_correct
        assert rec.label_3way == "partially_correct"

    def test_score_2_5(self):
        rec = _mohler_record(score_raw=2.5)
        self.h.harmonize(rec)
        assert rec.score_normalized == pytest.approx(0.5)
        # threshold=2.5, score >= threshold → correct
        assert rec.label_2way == "correct"
        # [1, 4) → partially_correct
        assert rec.label_3way == "partially_correct"

    def test_score_4(self):
        rec = _mohler_record(score_raw=4.0)
        self.h.harmonize(rec)
        assert rec.score_normalized == pytest.approx(0.8)
        assert rec.label_2way == "correct"
        # [4, 5] → correct
        assert rec.label_3way == "correct"

    def test_score_5(self):
        rec = _mohler_record(score_raw=5.0)
        self.h.harmonize(rec)
        assert rec.score_normalized == pytest.approx(1.0)
        assert rec.label_2way == "correct"
        assert rec.label_3way == "correct"

    def test_no_score_raw_leaves_labels_none(self):
        rec = _mohler_record(score_raw=None)
        self.h.harmonize(rec)
        assert rec.score_normalized is None
        assert rec.label_2way is None
        assert rec.label_3way is None

    def test_custom_threshold(self):
        h = LabelHarmonizer(threshold_2way=3.0)
        rec = _mohler_record(score_raw=2.5)
        h.harmonize(rec)
        # 2.5 < 3.0 → incorrect
        assert rec.label_2way == "incorrect"


class TestSciEntsBankLabelMapping:
    """SciEntsBank 5-way → 3-way → 2-way for all 5 labels."""

    def setup_method(self):
        self.h = LabelHarmonizer()

    def test_correct(self):
        rec = _scientsbank_record(label_5way="correct")
        self.h.harmonize(rec)
        assert rec.label_3way == "correct"
        assert rec.label_2way == "correct"

    def test_partially_correct_incomplete(self):
        rec = _scientsbank_record(
            label_5way="partially_correct_incomplete"
        )
        self.h.harmonize(rec)
        assert rec.label_3way == "partially_correct"
        assert rec.label_2way == "incorrect"

    def test_contradictory(self):
        rec = _scientsbank_record(label_5way="contradictory")
        self.h.harmonize(rec)
        assert rec.label_3way == "incorrect"
        assert rec.label_2way == "incorrect"

    def test_irrelevant(self):
        rec = _scientsbank_record(label_5way="irrelevant")
        self.h.harmonize(rec)
        assert rec.label_3way == "incorrect"
        assert rec.label_2way == "incorrect"

    def test_non_domain(self):
        rec = _scientsbank_record(label_5way="non_domain")
        self.h.harmonize(rec)
        assert rec.label_3way == "incorrect"
        assert rec.label_2way == "incorrect"

    def test_none_label_5way_leaves_labels_none(self):
        rec = _scientsbank_record(label_5way=None)
        self.h.harmonize(rec)
        assert rec.label_3way is None
        assert rec.label_2way is None


class TestDataGenerateRemapping:
    """Data_Generate label_3way='contradictory' → 'incorrect'."""

    def setup_method(self):
        self.h = LabelHarmonizer()

    def test_contradictory_remapped(self):
        rec = _data_generate_record(label_3way="contradictory")
        self.h.harmonize(rec)
        assert rec.label_3way == "incorrect"

    def test_correct_unchanged(self):
        rec = _data_generate_record(label_3way="correct")
        self.h.harmonize(rec)
        assert rec.label_3way == "correct"

    def test_incorrect_unchanged(self):
        rec = _data_generate_record(label_3way="incorrect")
        self.h.harmonize(rec)
        assert rec.label_3way == "incorrect"

    def test_partially_correct_unchanged(self):
        rec = _data_generate_record(
            label_3way="partially_correct"
        )
        self.h.harmonize(rec)
        assert rec.label_3way == "partially_correct"


class TestConsistencyCheck:
    """Log warning when score_normalized > 0.8 but label_2way='incorrect'."""

    def setup_method(self):
        self.h = LabelHarmonizer()

    def test_warning_logged(self, caplog):
        rec = _mohler_record(score_raw=4.5)
        # Manually set label_2way to incorrect to trigger warning
        rec.label_2way = "incorrect"
        rec.score_normalized = 0.9
        with caplog.at_level(logging.WARNING):
            self.h._consistency_check(rec)
        assert "Inconsistency" in caplog.text
        assert rec.sample_id in caplog.text

    def test_no_warning_when_consistent(self, caplog):
        rec = _mohler_record(score_raw=5.0)
        self.h.harmonize(rec)
        # score_normalized=1.0, label_2way="correct" → no warning
        with caplog.at_level(logging.WARNING):
            self.h._consistency_check(rec)
        assert "Inconsistency" not in caplog.text

    def test_no_warning_when_score_normalized_none(self, caplog):
        rec = _mohler_record(score_raw=None)
        with caplog.at_level(logging.WARNING):
            self.h.harmonize(rec)
        assert "Inconsistency" not in caplog.text


# ── 4.6: Property 2 — score_normalized ∈ [0.0, 1.0] ─────────────────
# Feature: asag-research-framework, Property 2: For any score in [0,5],
# score_normalized is in [0.0, 1.0]


@given(
    score=st.floats(
        min_value=0.0, max_value=5.0, allow_nan=False
    )
)
@settings(max_examples=200)
def test_property2_score_normalized_bounds(score):
    """**Validates: Requirements 2.1**

    For any score in [0,5], score_normalized is in [0.0, 1.0].
    """
    # Feature: asag-research-framework, Property 2: For any score
    # in [0,5], score_normalized is in [0.0, 1.0]
    h = LabelHarmonizer()
    rec = _mohler_record(score_raw=score)
    h.harmonize(rec)
    assert rec.score_normalized is not None
    assert 0.0 <= rec.score_normalized <= 1.0


# ── 4.7: Property 1 — score_raw=5→correct, score_raw=0→incorrect ─────
# Feature: asag-research-framework, Property 1: score_raw=5 →
# label_2way="correct", score_raw=0 → label_2way="incorrect"


@given(
    score=st.sampled_from([0.0, 5.0])
)
@settings(max_examples=200)
def test_property1_extreme_scores(score):
    """**Validates: Requirements 2.4**

    score_raw=5 → label_2way="correct",
    score_raw=0 → label_2way="incorrect".
    """
    # Feature: asag-research-framework, Property 1: score_raw=5 →
    # label_2way="correct", score_raw=0 → label_2way="incorrect"
    h = LabelHarmonizer()
    rec = _mohler_record(score_raw=score)
    h.harmonize(rec)
    if score == 5.0:
        assert rec.label_2way == "correct"
    elif score == 0.0:
        assert rec.label_2way == "incorrect"
