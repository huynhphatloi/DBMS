"""Tests for the DataAuditor (Task 6).

Covers:
- 6.1: Label distribution reporting
- 6.2: Low-confidence record identification
- 6.3: Data_Scraping "Not found" reference answer identification
- 6.4: Short student answer detection
- 6.5: Stratified audit sample selection
- 6.6: Numerical/computational question detection
"""

from __future__ import annotations

import pytest

from src.data.audit import DataAuditor, LabelDistribution
from src.data.schema import UnifiedRecord


# ── Helpers ───────────────────────────────────────────────────────────


def _rec(
    source: str = "data_generate",
    sid: str = "GEN_00001",
    label_5way: str | None = "correct",
    label_3way: str | None = "correct",
    label_2way: str | None = "correct",
    student_answer: str = "This is a valid answer.",
    reference_answer: str = "The reference answer.",
    question: str = "What is X?",
    question_id: str = "q1",
    annotation_confidence: float | None = None,
    **kw,
) -> UnifiedRecord:
    defaults = dict(
        sample_id=sid,
        source_dataset=source,
        original_id="o1",
        question_id=question_id,
        domain="science",
        subdomain="biology",
        difficulty="unknown",
        question=question,
        reference_answer=reference_answer,
        student_answer=student_answer,
        label_5way=label_5way,
        label_3way=label_3way,
        label_2way=label_2way,
        annotation_confidence=annotation_confidence,
    )
    defaults.update(kw)
    return UnifiedRecord(**defaults)


# ── 6.1: Label distribution reporting ─────────────────────────────────


class TestLabelDistributions:
    def test_single_source_counts(self):
        records = [
            _rec(sid="GEN_001", label_5way="correct"),
            _rec(sid="GEN_002", label_5way="correct"),
            _rec(sid="GEN_003", label_5way="incorrect"),
        ]
        dists = DataAuditor.label_distributions(records)
        assert len(dists) == 1
        d = dists[0]
        assert d.source_dataset == "data_generate"
        assert d.label_5way["correct"] == 2
        assert d.label_5way["incorrect"] == 1

    def test_multiple_sources(self):
        records = [
            _rec(source="scientsbank", sid="SEB_001", label_5way="correct"),
            _rec(source="data_generate", sid="GEN_001", label_5way="incorrect"),
        ]
        dists = DataAuditor.label_distributions(records)
        assert len(dists) == 2
        sources = {d.source_dataset for d in dists}
        assert sources == {"data_generate", "scientsbank"}

    def test_percentages(self):
        records = [
            _rec(sid="GEN_001", label_2way="correct"),
            _rec(sid="GEN_002", label_2way="correct"),
            _rec(sid="GEN_003", label_2way="incorrect"),
            _rec(sid="GEN_004", label_2way="incorrect"),
        ]
        dists = DataAuditor.label_distributions(records)
        pct = dists[0].label_2way_pct
        assert pct["correct"] == pytest.approx(50.0)
        assert pct["incorrect"] == pytest.approx(50.0)

    def test_none_labels_excluded(self):
        records = [
            _rec(
                sid="GEN_001",
                label_5way=None,
                label_3way="correct",
                label_2way="correct",
            ),
            _rec(
                sid="GEN_002",
                label_5way="correct",
                label_3way=None,
                label_2way=None,
            ),
        ]
        dists = DataAuditor.label_distributions(records)
        d = dists[0]
        assert d.label_5way == {"correct": 1}
        assert d.label_3way == {"correct": 1}
        assert d.label_2way == {"correct": 1}

    def test_empty_records(self):
        dists = DataAuditor.label_distributions([])
        assert dists == []

    def test_empty_percentages(self):
        dist = LabelDistribution(source_dataset="test")
        assert dist.label_5way_pct == {}
        assert dist.label_3way_pct == {}
        assert dist.label_2way_pct == {}


# ── 6.2: Low-confidence record identification ────────────────────────


class TestLowConfidence:
    def test_below_threshold(self):
        records = [
            _rec(sid="GEN_001", annotation_confidence=0.5),
            _rec(sid="GEN_002", annotation_confidence=0.84),
        ]
        result = DataAuditor.low_confidence_records(records)
        assert len(result) == 2
        assert {r.sample_id for r in result} == {"GEN_001", "GEN_002"}

    def test_at_threshold_excluded(self):
        records = [_rec(sid="GEN_001", annotation_confidence=0.85)]
        result = DataAuditor.low_confidence_records(records)
        assert len(result) == 0

    def test_above_threshold_excluded(self):
        records = [_rec(sid="GEN_001", annotation_confidence=0.95)]
        result = DataAuditor.low_confidence_records(records)
        assert len(result) == 0

    def test_none_confidence_excluded(self):
        records = [_rec(sid="GEN_001", annotation_confidence=None)]
        result = DataAuditor.low_confidence_records(records)
        assert len(result) == 0

    def test_custom_threshold(self):
        records = [_rec(sid="GEN_001", annotation_confidence=0.90)]
        result = DataAuditor.low_confidence_records(records, threshold=0.95)
        assert len(result) == 1


# ── 6.3: "Not found" reference answer identification ─────────────────


class TestNotFoundReferences:
    def test_detects_not_found(self):
        records = [
            _rec(
                source="data_scraping",
                sid="SCR_001",
                reference_answer="Not found",
                label_5way=None,
                label_3way=None,
                label_2way=None,
            ),
        ]
        result = DataAuditor.not_found_references(records)
        assert len(result) == 1
        assert result[0].sample_id == "SCR_001"

    def test_ignores_non_scraping(self):
        records = [
            _rec(
                source="data_generate",
                sid="GEN_001",
                reference_answer="Not found",
            ),
        ]
        result = DataAuditor.not_found_references(records)
        assert len(result) == 0

    def test_ignores_valid_reference(self):
        records = [
            _rec(
                source="data_scraping",
                sid="SCR_001",
                reference_answer="A valid answer",
                label_5way=None,
                label_3way=None,
                label_2way=None,
            ),
        ]
        result = DataAuditor.not_found_references(records)
        assert len(result) == 0


# ── 6.4: Short student answer detection ──────────────────────────────


class TestShortStudentAnswers:
    def test_detects_short(self):
        records = [
            _rec(sid="GEN_001", student_answer="yes"),
            _rec(sid="GEN_002", student_answer="I think"),
        ]
        result = DataAuditor.short_student_answers(records)
        assert len(result) == 2

    def test_exactly_three_tokens_excluded(self):
        records = [_rec(sid="GEN_001", student_answer="I think so")]
        result = DataAuditor.short_student_answers(records)
        assert len(result) == 0

    def test_empty_answer(self):
        records = [_rec(sid="GEN_001", student_answer="")]
        result = DataAuditor.short_student_answers(records)
        assert len(result) == 1

    def test_long_answer_excluded(self):
        records = [_rec(sid="GEN_001", student_answer="This is a long student answer")]
        result = DataAuditor.short_student_answers(records)
        assert len(result) == 0

    def test_custom_min_tokens(self):
        records = [_rec(sid="GEN_001", student_answer="one two three")]
        result = DataAuditor.short_student_answers(records, min_tokens=4)
        assert len(result) == 1


# ── 6.5: Stratified audit sample selection ────────────────────────────


class TestStratifiedSample:
    def test_basic_sample(self):
        records = [
            _rec(sid=f"GEN_{i:03d}", label_5way="correct", source="data_generate")
            for i in range(10)
        ] + [
            _rec(sid=f"SEB_{i:03d}", label_5way="incorrect", source="scientsbank")
            for i in range(10)
        ]
        sample = DataAuditor.stratified_sample(records, n=6)
        assert len(sample) == 6

    def test_both_strata_represented(self):
        records = [
            _rec(sid=f"GEN_{i:03d}", label_5way="correct", source="data_generate")
            for i in range(10)
        ] + [
            _rec(sid=f"SEB_{i:03d}", label_5way="incorrect", source="scientsbank")
            for i in range(10)
        ]
        sample = DataAuditor.stratified_sample(records, n=10)
        sources = {r.source_dataset for r in sample}
        assert "data_generate" in sources
        assert "scientsbank" in sources

    def test_n_larger_than_records(self):
        records = [_rec(sid="GEN_001"), _rec(sid="GEN_002")]
        sample = DataAuditor.stratified_sample(records, n=100)
        # Can't return more than available
        assert len(sample) <= len(records)

    def test_empty_records(self):
        sample = DataAuditor.stratified_sample([], n=5)
        assert sample == []

    def test_n_zero(self):
        records = [_rec(sid="GEN_001")]
        sample = DataAuditor.stratified_sample(records, n=0)
        assert sample == []

    def test_deterministic(self):
        records = [
            _rec(sid=f"GEN_{i:03d}", label_5way="correct")
            for i in range(20)
        ]
        s1 = DataAuditor.stratified_sample(records, n=5, seed=42)
        s2 = DataAuditor.stratified_sample(records, n=5, seed=42)
        assert [r.sample_id for r in s1] == [r.sample_id for r in s2]


# ── 6.6: Numerical/computational question detection ───────────────────


class TestNumericalQuestionDetection:
    @pytest.mark.parametrize(
        "question",
        [
            "Calculate the force when mass is 10 kg.",
            "How many meters are in 5 kilometers?",
            "What is the value of 3 + 4?",
            "Compute the area of a circle with radius 7 cm.",
            "Convert 100 celsius to fahrenheit.",
            "Find the value of x if 2x + 3 = 7",
            "Determine the speed in m/s.",
        ],
    )
    def test_numerical_detected(self, question: str):
        assert DataAuditor.is_numerical_question(question) is True

    @pytest.mark.parametrize(
        "question",
        [
            "What is photosynthesis?",
            "Explain the concept of gravity.",
            "Describe the water cycle.",
            "Why do leaves change color in autumn?",
        ],
    )
    def test_conceptual_not_detected(self, question: str):
        assert DataAuditor.is_numerical_question(question) is False

    def test_counts_data_scraping_only(self):
        records = [
            _rec(
                source="data_scraping",
                sid="SCR_001",
                question="Calculate 2+2",
                question_id="q1",
                label_5way=None,
                label_3way=None,
                label_2way=None,
            ),
            _rec(
                source="data_scraping",
                sid="SCR_002",
                question="What is gravity?",
                question_id="q2",
                label_5way=None,
                label_3way=None,
                label_2way=None,
            ),
            _rec(
                source="data_generate",
                sid="GEN_001",
                question="Calculate 5+5",
                question_id="q3",
            ),
        ]
        num, con = DataAuditor.numerical_question_counts(records)
        assert num == 1
        assert con == 1

    def test_deduplicates_by_question_id(self):
        records = [
            _rec(
                source="data_scraping",
                sid="SCR_001",
                question="Calculate 2+2",
                question_id="q1",
                label_5way=None,
                label_3way=None,
                label_2way=None,
            ),
            _rec(
                source="data_scraping",
                sid="SCR_002",
                question="Calculate 2+2",
                question_id="q1",
                label_5way=None,
                label_3way=None,
                label_2way=None,
            ),
        ]
        num, con = DataAuditor.numerical_question_counts(records)
        assert num == 1
        assert con == 0


# ── 6.7: Full audit integration ──────────────────────────────────────


class TestFullAudit:
    def test_full_audit_returns_report(self):
        records = [
            _rec(sid="GEN_001", annotation_confidence=0.5, student_answer="yes"),
            _rec(
                source="data_scraping",
                sid="SCR_001",
                reference_answer="Not found",
                question="Calculate 2+2",
                question_id="q_scr1",
                label_5way=None,
                label_3way=None,
                label_2way=None,
            ),
            _rec(
                source="data_scraping",
                sid="SCR_002",
                reference_answer="Gravity pulls objects.",
                question="What is gravity?",
                question_id="q_scr2",
                label_5way=None,
                label_3way=None,
                label_2way=None,
            ),
        ]
        report = DataAuditor.full_audit(records)

        assert len(report.label_distributions) == 2  # data_generate + data_scraping
        assert len(report.low_confidence_records) == 1
        assert len(report.not_found_reference_records) == 1
        assert len(report.short_answer_records) == 1
        assert report.numerical_question_count == 1
        assert report.conceptual_question_count == 1
