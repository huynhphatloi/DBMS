"""Tests for the SplitManager and SplitIntegrityError.

Covers:
- 5.6: Property 3 — question-level split produces disjoint question_id sets
- 5.7: Property 4 — adversarial co-location invariant holds after split
- 5.8: Edge-case tests — inject split violations, verify SplitIntegrityError
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.schema import UnifiedRecord
from src.data.splitter import SplitIntegrityError, SplitManager


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
        split="train",
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
        split="train",
    )
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


# ── 5.1: SciEntsBank split preservation ──────────────────────────────


class TestSciEntsBankSplitPreservation:
    """SciEntsBank UA/UQ/UD splits are preserved without modification."""

    def test_splits_unchanged(self):
        records = [
            _scientsbank_record(sample_id="SEB_00001", split="train"),
            _scientsbank_record(sample_id="SEB_00002", split="test_ua"),
            _scientsbank_record(sample_id="SEB_00003", split="test_uq"),
            _scientsbank_record(sample_id="SEB_00004", split="test_ud"),
        ]
        sm = SplitManager()
        sm.assign_splits(records)
        assert records[0].split == "train"
        assert records[1].split == "test_ua"
        assert records[2].split == "test_uq"
        assert records[3].split == "test_ud"


# ── 5.2: MohlerASAG question-level split ─────────────────────────────


class TestMohlerSplit:
    """MohlerASAG question-level 60/20/20 split."""

    def test_basic_split_assignment(self):
        """All records get a split assigned."""
        records = [
            _mohler_record(
                sample_id=f"MOH_{i:05d}",
                question_id=f"q{i // 3}",
            )
            for i in range(30)
        ]
        sm = SplitManager(seed=42)
        sm.assign_splits(records)
        for rec in records:
            assert rec.split in ("train", "valid", "test")

    def test_question_ids_disjoint(self):
        """No question_id appears in more than one partition."""
        records = [
            _mohler_record(
                sample_id=f"MOH_{i:05d}",
                question_id=f"q{i // 5}",
            )
            for i in range(50)
        ]
        sm = SplitManager(seed=42)
        sm.assign_splits(records)

        split_qids: dict[str, set[str]] = {
            "train": set(), "valid": set(), "test": set(),
        }
        for rec in records:
            split_qids[rec.split].add(rec.question_id)

        assert split_qids["train"].isdisjoint(split_qids["valid"])
        assert split_qids["train"].isdisjoint(split_qids["test"])
        assert split_qids["valid"].isdisjoint(split_qids["test"])

    def test_deterministic_with_same_seed(self):
        """Same seed produces same splits."""
        records1 = [
            _mohler_record(sample_id=f"MOH_{i:05d}", question_id=f"q{i}")
            for i in range(20)
        ]
        records2 = [
            _mohler_record(sample_id=f"MOH_{i:05d}", question_id=f"q{i}")
            for i in range(20)
        ]
        SplitManager(seed=99).assign_splits(records1)
        SplitManager(seed=99).assign_splits(records2)
        for r1, r2 in zip(records1, records2):
            assert r1.split == r2.split

    def test_approximate_ratio(self):
        """Roughly 60/20/20 distribution with enough questions."""
        records = [
            _mohler_record(
                sample_id=f"MOH_{i:05d}",
                question_id=f"q{i}",
            )
            for i in range(100)
        ]
        sm = SplitManager(seed=42)
        sm.assign_splits(records)

        counts = {"train": 0, "valid": 0, "test": 0}
        for rec in records:
            counts[rec.split] += 1

        # With 100 unique question_ids (1 record each), expect ~60/20/20
        assert counts["train"] == 60
        assert counts["valid"] == 20
        assert counts["test"] == 20


# ── 5.3: Data_Generate split preservation + adversarial co-location ──


class TestDataGenerateSplitPreservation:
    """Data_Generate splits are preserved; adversarial co-location verified."""

    def test_splits_preserved(self):
        records = [
            _data_generate_record(sample_id="GEN_00001", split="train"),
            _data_generate_record(
                sample_id="GEN_00002",
                split="test_unseen_questions",
                question_id="q99",
            ),
        ]
        sm = SplitManager()
        sm.assign_splits(records)
        assert records[0].split == "train"
        assert records[1].split == "test_unseen_questions"

    def test_adversarial_colocation_valid(self):
        """No error when adversarial variant is in same split as original."""
        records = [
            _data_generate_record(
                sample_id="GEN_00001",
                original_id="inst_001",
                split="train",
            ),
            _data_generate_record(
                sample_id="GEN_00002",
                original_id="inst_002",
                split="train",
                adversarial_variant_of="inst_001",
                is_adversarial=True,
            ),
        ]
        sm = SplitManager()
        # Should not raise
        sm.assign_splits(records)


# ── 5.4 & 5.5: Leakage checks and SplitIntegrityError ────────────────


class TestLeakageChecks:
    """Verify unseen-question and unseen-domain leakage detection."""

    def test_unseen_questions_no_leak(self):
        """No error when train and test_unseen_questions have disjoint question_ids."""
        records = [
            _data_generate_record(
                sample_id="GEN_00001",
                question_id="q1",
                split="train",
            ),
            _data_generate_record(
                sample_id="GEN_00002",
                question_id="q99",
                split="test_unseen_questions",
            ),
        ]
        sm = SplitManager()
        sm.assign_splits(records)  # Should not raise

    def test_unseen_domains_no_leak(self):
        """No error when train and test_unseen_domains have disjoint domains."""
        records = [
            _data_generate_record(
                sample_id="GEN_00001",
                domain="science",
                split="train",
            ),
            _data_generate_record(
                sample_id="GEN_00002",
                domain="history",
                split="test_unseen_domains",
            ),
        ]
        sm = SplitManager()
        sm.assign_splits(records)  # Should not raise


# ── 5.8: Edge-case tests — inject violations ─────────────────────────


class TestSplitIntegrityViolations:
    """Inject split violations and verify SplitIntegrityError is raised."""

    def test_adversarial_colocation_violation(self):
        """Error when adversarial variant is in different split than original."""
        records = [
            _data_generate_record(
                sample_id="GEN_00001",
                original_id="inst_001",
                split="train",
            ),
            _data_generate_record(
                sample_id="GEN_00002",
                original_id="inst_002",
                split="test_adversarial",
                adversarial_variant_of="inst_001",
                is_adversarial=True,
            ),
        ]
        sm = SplitManager()
        with pytest.raises(SplitIntegrityError) as exc_info:
            sm.assign_splits(records)
        assert "GEN_00002" in exc_info.value.affected_sample_ids

    def test_unseen_questions_leak(self):
        """Error when a question_id appears in both train and test_unseen_questions."""
        records = [
            _data_generate_record(
                sample_id="GEN_00001",
                question_id="q_shared",
                split="train",
            ),
            _data_generate_record(
                sample_id="GEN_00002",
                question_id="q_shared",
                split="test_unseen_questions",
            ),
        ]
        sm = SplitManager()
        with pytest.raises(SplitIntegrityError) as exc_info:
            sm.assign_splits(records)
        assert "GEN_00001" in exc_info.value.affected_sample_ids
        assert "GEN_00002" in exc_info.value.affected_sample_ids

    def test_unseen_domains_leak(self):
        """Error when a domain appears in both train and test_unseen_domains."""
        records = [
            _data_generate_record(
                sample_id="GEN_00001",
                domain="leaked_domain",
                split="train",
            ),
            _data_generate_record(
                sample_id="GEN_00002",
                domain="leaked_domain",
                split="test_unseen_domains",
            ),
        ]
        sm = SplitManager()
        with pytest.raises(SplitIntegrityError) as exc_info:
            sm.assign_splits(records)
        assert "GEN_00001" in exc_info.value.affected_sample_ids
        assert "GEN_00002" in exc_info.value.affected_sample_ids

    def test_split_integrity_error_has_sample_ids(self):
        """SplitIntegrityError carries affected_sample_ids attribute."""
        err = SplitIntegrityError("test", ["id1", "id2"])
        assert err.affected_sample_ids == ["id1", "id2"]
        assert "id1" in str(err)
        assert "id2" in str(err)

    def test_multiple_adversarial_violations(self):
        """Multiple adversarial violations are all reported."""
        records = [
            _data_generate_record(
                sample_id="GEN_00001",
                original_id="inst_001",
                split="train",
            ),
            _data_generate_record(
                sample_id="GEN_00002",
                original_id="inst_002",
                split="test_adversarial",
                adversarial_variant_of="inst_001",
                is_adversarial=True,
            ),
            _data_generate_record(
                sample_id="GEN_00003",
                original_id="inst_003",
                split="valid",
            ),
            _data_generate_record(
                sample_id="GEN_00004",
                original_id="inst_004",
                split="test_seen",
                adversarial_variant_of="inst_003",
                is_adversarial=True,
            ),
        ]
        sm = SplitManager()
        with pytest.raises(SplitIntegrityError) as exc_info:
            sm.assign_splits(records)
        assert "GEN_00002" in exc_info.value.affected_sample_ids
        assert "GEN_00004" in exc_info.value.affected_sample_ids


# ── 5.6: Property 3 — disjoint question_id sets ──────────────────────
# Feature: asag-research-framework, Property 3: For any question-answer
# set, question-level split produces disjoint question_id sets


@given(
    data=st.data(),
)
@settings(max_examples=100)
def test_property3_disjoint_question_ids(data):
    """**Validates: Requirements 3.2**

    For any question-answer set, question-level split produces
    disjoint question_id sets across train/valid/test.
    """
    # Feature: asag-research-framework, Property 3: For any question-answer
    # set, question-level split produces disjoint question_id sets

    # Generate between 1 and 30 unique question_ids
    num_questions = data.draw(st.integers(min_value=1, max_value=30))
    question_ids = [f"q{i}" for i in range(num_questions)]

    # Generate 1-5 answers per question
    records: list[UnifiedRecord] = []
    counter = 0
    for qid in question_ids:
        num_answers = data.draw(st.integers(min_value=1, max_value=5))
        for _ in range(num_answers):
            counter += 1
            records.append(
                _mohler_record(
                    sample_id=f"MOH_{counter:05d}",
                    question_id=qid,
                )
            )

    seed = data.draw(st.integers(min_value=0, max_value=10000))
    sm = SplitManager(seed=seed)
    sm.assign_splits(records)

    # Collect question_ids per split
    split_qids: dict[str, set[str]] = {"train": set(), "valid": set(), "test": set()}
    for rec in records:
        split_qids[rec.split].add(rec.question_id)

    # Verify disjointness
    assert split_qids["train"].isdisjoint(split_qids["valid"]), (
        f"train ∩ valid = {split_qids['train'] & split_qids['valid']}"
    )
    assert split_qids["train"].isdisjoint(split_qids["test"]), (
        f"train ∩ test = {split_qids['train'] & split_qids['test']}"
    )
    assert split_qids["valid"].isdisjoint(split_qids["test"]), (
        f"valid ∩ test = {split_qids['valid'] & split_qids['test']}"
    )


# ── 5.7: Property 4 — adversarial co-location invariant ──────────────
# Feature: asag-research-framework, Property 4: For any records with
# adversarial links, co-location invariant holds after split assignment


@given(
    data=st.data(),
)
@settings(max_examples=100)
def test_property4_adversarial_colocation(data):
    """**Validates: Requirements 3.4**

    For any records with adversarial links, co-location invariant
    holds after split assignment.
    """
    # Feature: asag-research-framework, Property 4: For any records with
    # adversarial links, co-location invariant holds after split assignment

    valid_splits = [
        "train", "valid", "test_unseen_questions",
        "test_unseen_answers", "test_seen", "test_adversarial",
        "test_unseen_domains",
    ]

    # Generate between 1 and 15 original records
    num_originals = data.draw(st.integers(min_value=1, max_value=15))
    records: list[UnifiedRecord] = []
    counter = 0

    # Use distinct question_ids and domains per split to avoid
    # triggering unseen-question/domain leakage checks
    for i in range(num_originals):
        counter += 1
        split = data.draw(st.sampled_from(valid_splits))
        rec = _data_generate_record(
            sample_id=f"GEN_{counter:05d}",
            original_id=f"inst_{counter:03d}",
            question_id=f"q_{split}_{i}",
            domain=f"domain_{split}_{i}",
            split=split,
        )
        records.append(rec)

        # Optionally add an adversarial variant in the SAME split
        add_variant = data.draw(st.booleans())
        if add_variant:
            counter += 1
            records.append(
                _data_generate_record(
                    sample_id=f"GEN_{counter:05d}",
                    original_id=f"inst_{counter:03d}",
                    question_id=f"q_{split}_{i}",
                    domain=f"domain_{split}_{i}",
                    split=split,
                    adversarial_variant_of=rec.original_id,
                    is_adversarial=True,
                )
            )

    sm = SplitManager()
    # Should not raise — all adversarial variants are co-located
    sm.assign_splits(records)
