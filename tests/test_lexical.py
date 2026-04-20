"""Unit tests for the Lexical Overlap baseline (src/grading/baselines/lexical.py).

Covers:
- BLEU-1 = 1.0 for identical sentences, 0.0 for completely different sentences
- BLEU-4 basic behaviour
- Jaccard = 1.0 for identical word sets, 0.0 for disjoint word sets
- ROUGE-L = 1.0 for identical sentences
- Word overlap ratio for known pairs
- Threshold classifier produces valid labels
- LR mode trains and predicts valid labels
- Integration with EvaluationHarness returns macro_f1, weighted_f1, accuracy
"""

from __future__ import annotations

import pytest

from src.data.schema import UnifiedRecord
from src.grading.baselines.lexical import (
    LexicalLogisticRegression,
    LexicalThresholdClassifier,
    bleu_n,
    compute_lexical_features,
    evaluate_lexical_model,
    jaccard_similarity,
    rouge_l,
    word_overlap_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    reference: str,
    student: str,
    label: str = "correct",
    sample_id: str = "GEN_0001",
) -> UnifiedRecord:
    return UnifiedRecord(
        sample_id=sample_id,
        source_dataset="data_generate",
        original_id=sample_id,
        question_id="Q1",
        domain="science",
        subdomain="physics",
        difficulty="medium",
        question="What is photosynthesis?",
        reference_answer=reference,
        student_answer=student,
        label_2way=label,
        label_3way=label,
    )


# ---------------------------------------------------------------------------
# BLEU-1
# ---------------------------------------------------------------------------

class TestBleu1:
    def test_identical_sentences(self):
        assert bleu_n("the cat sat on the mat", "the cat sat on the mat", 1) == pytest.approx(1.0)

    def test_completely_different(self):
        # No shared unigrams → precision = 0
        score = bleu_n("apple orange banana", "dog cat fish", 1)
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self):
        # "the cat" vs "the dog" → 1 shared unigram out of 2 hypothesis tokens
        score = bleu_n("the cat", "the dog", 1)
        assert 0.0 < score < 1.0

    def test_empty_hypothesis(self):
        assert bleu_n("hello world", "", 1) == pytest.approx(0.0)

    def test_empty_reference(self):
        # No reference → brevity penalty collapses to 0
        assert bleu_n("", "hello world", 1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BLEU-4
# ---------------------------------------------------------------------------

class TestBleu4:
    def test_identical_long_sentence(self):
        sent = "the quick brown fox jumps over the lazy dog"
        assert bleu_n(sent, sent, 4) == pytest.approx(1.0)

    def test_short_hypothesis_below_4_tokens(self):
        # Hypothesis shorter than n → 0
        assert bleu_n("one two three four five", "one two", 4) == pytest.approx(0.0)

    def test_completely_different(self):
        assert bleu_n("apple orange banana grape", "dog cat fish bird", 4) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

class TestRougeL:
    def test_identical_sentences(self):
        assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)

    def test_completely_different(self):
        score = rouge_l("apple orange banana", "dog cat fish")
        assert score == pytest.approx(0.0)

    def test_empty_reference(self):
        assert rouge_l("", "hello world") == pytest.approx(0.0)

    def test_empty_hypothesis(self):
        assert rouge_l("hello world", "") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # "the cat" is a subsequence of "the cat sat on the mat"
        score = rouge_l("the cat sat on the mat", "the cat")
        assert 0.0 < score < 1.0

    def test_symmetry_not_required_but_nonzero(self):
        # ROUGE-L is not symmetric; just verify it's in [0,1]
        s = rouge_l("quick brown fox", "fox brown quick")
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical_word_sets(self):
        assert jaccard_similarity("cat dog bird", "cat dog bird") == pytest.approx(1.0)

    def test_disjoint_word_sets(self):
        assert jaccard_similarity("apple orange", "dog cat") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # {"cat", "dog"} ∩ {"cat", "fish"} = {"cat"}, union = 3
        score = jaccard_similarity("cat dog", "cat fish")
        assert score == pytest.approx(1 / 3)

    def test_both_empty(self):
        assert jaccard_similarity("", "") == pytest.approx(1.0)

    def test_one_empty(self):
        assert jaccard_similarity("hello", "") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Word overlap ratio
# ---------------------------------------------------------------------------

class TestWordOverlapRatio:
    def test_identical(self):
        assert word_overlap_ratio("cat dog bird", "cat dog bird") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert word_overlap_ratio("apple orange", "dog cat") == pytest.approx(0.0)

    def test_partial(self):
        # ref = {"cat", "dog", "bird"}, hyp = {"cat", "dog"} → 2/3
        assert word_overlap_ratio("cat dog bird", "cat dog") == pytest.approx(2 / 3)

    def test_empty_reference(self):
        assert word_overlap_ratio("", "hello world") == pytest.approx(0.0)

    def test_superset_hypothesis(self):
        # All reference words present in hypothesis → 1.0
        assert word_overlap_ratio("cat dog", "cat dog bird fish") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_lexical_features
# ---------------------------------------------------------------------------

class TestComputeLexicalFeatures:
    def test_returns_all_five_keys(self):
        feats = compute_lexical_features("hello world", "hello world")
        assert set(feats.keys()) == {"bleu_1", "bleu_4", "rouge_l", "jaccard", "word_overlap"}

    def test_identical_sentences_all_ones(self):
        feats = compute_lexical_features("the cat sat on the mat", "the cat sat on the mat")
        assert feats["bleu_1"] == pytest.approx(1.0)
        assert feats["rouge_l"] == pytest.approx(1.0)
        assert feats["jaccard"] == pytest.approx(1.0)
        assert feats["word_overlap"] == pytest.approx(1.0)

    def test_all_values_in_unit_interval(self):
        feats = compute_lexical_features("photosynthesis converts light into energy", "plants use sunlight")
        for v in feats.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# LexicalThresholdClassifier
# ---------------------------------------------------------------------------

class TestLexicalThresholdClassifier:
    def _make_records(self):
        return [
            _make_record("the cat sat on the mat", "the cat sat on the mat", "correct", "GEN_0001"),
            _make_record("photosynthesis uses light", "dogs bark loudly", "incorrect", "GEN_0002"),
            _make_record("water boils at 100 degrees", "water boils at 100 degrees celsius", "correct", "GEN_0003"),
        ]

    def test_predict_returns_valid_labels(self):
        clf = LexicalThresholdClassifier(metric="rouge_l", threshold=0.5)
        records = self._make_records()
        preds = clf.predict(records)
        assert len(preds) == len(records)
        assert all(p in {"correct", "incorrect"} for p in preds)

    def test_identical_pair_classified_correct(self):
        clf = LexicalThresholdClassifier(metric="rouge_l", threshold=0.5)
        rec = _make_record("the cat sat on the mat", "the cat sat on the mat")
        assert clf.predict([rec]) == ["correct"]

    def test_disjoint_pair_classified_incorrect(self):
        clf = LexicalThresholdClassifier(metric="jaccard", threshold=0.5)
        rec = _make_record("apple orange banana", "dog cat fish")
        assert clf.predict([rec]) == ["incorrect"]

    def test_predict_proba_shape_and_range(self):
        clf = LexicalThresholdClassifier()
        records = self._make_records()
        proba = clf.predict_proba(records)
        assert len(proba) == len(records)
        for row in proba:
            assert len(row) == 2
            assert abs(sum(row) - 1.0) < 1e-9
            assert all(0.0 <= p <= 1.0 for p in row)

    def test_fit_does_not_crash(self):
        clf = LexicalThresholdClassifier()
        records = self._make_records()
        clf.fit(records, "label_2way")  # should not raise

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            LexicalThresholdClassifier(metric="nonexistent_metric")

    def test_custom_labels(self):
        clf = LexicalThresholdClassifier(
            metric="rouge_l",
            threshold=0.5,
            positive_label="yes",
            negative_label="no",
        )
        rec = _make_record("the cat sat on the mat", "the cat sat on the mat")
        assert clf.predict([rec]) == ["yes"]


# ---------------------------------------------------------------------------
# LexicalLogisticRegression
# ---------------------------------------------------------------------------

class TestLexicalLogisticRegression:
    def _make_training_records(self):
        pairs = [
            ("the cat sat on the mat", "the cat sat on the mat", "correct"),
            ("photosynthesis converts light to energy", "photosynthesis uses sunlight to make food", "correct"),
            ("water boils at 100 degrees", "water freezes at 0 degrees", "incorrect"),
            ("the earth orbits the sun", "dogs bark at night", "incorrect"),
            ("gravity pulls objects downward", "gravity is a force that attracts masses", "correct"),
            ("cells divide during mitosis", "mitosis is cell division", "correct"),
            ("the sky is blue", "the ocean is deep", "incorrect"),
            ("plants need sunlight to grow", "plants require light for photosynthesis", "correct"),
        ]
        return [
            _make_record(ref, stu, lbl, f"GEN_{i:04d}")
            for i, (ref, stu, lbl) in enumerate(pairs)
        ]

    def test_fit_and_predict_valid_labels(self):
        clf = LexicalLogisticRegression()
        records = self._make_training_records()
        clf.fit(records, "label_2way")
        preds = clf.predict(records)
        assert len(preds) == len(records)
        assert all(p in {"correct", "incorrect"} for p in preds)

    def test_predict_proba_shape_and_range(self):
        clf = LexicalLogisticRegression()
        records = self._make_training_records()
        clf.fit(records, "label_2way")
        proba = clf.predict_proba(records)
        assert len(proba) == len(records)
        for row in proba:
            assert len(row) == 2
            assert abs(sum(row) - 1.0) < 1e-9
            assert all(0.0 <= p <= 1.0 for p in row)

    def test_identical_pair_likely_correct(self):
        clf = LexicalLogisticRegression()
        records = self._make_training_records()
        clf.fit(records, "label_2way")
        rec = _make_record("the cat sat on the mat", "the cat sat on the mat", "correct", "GEN_9999")
        pred = clf.predict([rec])
        assert pred[0] == "correct"

    def test_three_way_classification(self):
        pairs = [
            ("photosynthesis converts light", "photosynthesis converts light", "correct"),
            ("water boils at 100 degrees", "water boils at 100 degrees", "correct"),
            ("gravity pulls objects down", "gravity is a weak force", "partially_correct"),
            ("cells divide in mitosis", "cells sometimes split", "partially_correct"),
            ("the earth orbits the sun", "dogs bark at night", "incorrect"),
            ("plants need sunlight", "rocks are hard", "incorrect"),
        ]
        records = [
            _make_record(ref, stu, lbl, f"GEN_{i:04d}")
            for i, (ref, stu, lbl) in enumerate(pairs)
        ]
        clf = LexicalLogisticRegression()
        clf.fit(records, "label_3way")
        preds = clf.predict(records)
        assert all(p in {"correct", "partially_correct", "incorrect"} for p in preds)


# ---------------------------------------------------------------------------
# Integration with EvaluationHarness (task 10.4)
# ---------------------------------------------------------------------------

class TestEvaluationHarnessIntegration:
    def _make_eval_records(self):
        pairs = [
            ("the cat sat on the mat", "the cat sat on the mat", "correct"),
            ("photosynthesis converts light to energy", "photosynthesis uses sunlight", "correct"),
            ("water boils at 100 degrees", "water freezes at 0 degrees", "incorrect"),
            ("the earth orbits the sun", "dogs bark at night", "incorrect"),
            ("gravity pulls objects downward", "gravity attracts masses", "correct"),
            ("cells divide during mitosis", "mitosis is cell division", "correct"),
            ("the sky is blue", "the ocean is deep", "incorrect"),
            ("plants need sunlight to grow", "plants require light", "correct"),
        ]
        return [
            _make_record(ref, stu, lbl, f"GEN_{i:04d}")
            for i, (ref, stu, lbl) in enumerate(pairs)
        ]

    def test_threshold_model_returns_required_keys(self):
        records = self._make_eval_records()
        clf = LexicalThresholdClassifier(metric="rouge_l", threshold=0.3)
        result = evaluate_lexical_model(clf, records, "label_2way", bootstrap_n=50)
        assert "macro_f1" in result
        assert "weighted_f1" in result
        assert "accuracy" in result

    def test_lr_model_returns_required_keys(self):
        records = self._make_eval_records()
        clf = LexicalLogisticRegression()
        clf.fit(records, "label_2way")
        result = evaluate_lexical_model(clf, records, "label_2way", bootstrap_n=50)
        assert "macro_f1" in result
        assert "weighted_f1" in result
        assert "accuracy" in result

    def test_metrics_have_ci_fields(self):
        records = self._make_eval_records()
        clf = LexicalThresholdClassifier(metric="jaccard", threshold=0.3)
        result = evaluate_lexical_model(clf, records, "label_2way", bootstrap_n=50)
        for key in ("macro_f1", "weighted_f1", "accuracy"):
            assert "value" in result[key]
            assert "ci_lower" in result[key]
            assert "ci_upper" in result[key]

    def test_metric_values_in_unit_interval(self):
        records = self._make_eval_records()
        clf = LexicalThresholdClassifier(metric="rouge_l", threshold=0.3)
        result = evaluate_lexical_model(clf, records, "label_2way", bootstrap_n=50)
        for key in ("macro_f1", "weighted_f1", "accuracy"):
            v = result[key]["value"]
            assert 0.0 <= v <= 1.0, f"{key} value {v} out of [0,1]"

    def test_ci_lower_le_value_le_upper(self):
        records = self._make_eval_records()
        clf = LexicalThresholdClassifier(metric="rouge_l", threshold=0.3)
        result = evaluate_lexical_model(clf, records, "label_2way", bootstrap_n=50)
        for key in ("macro_f1", "weighted_f1", "accuracy"):
            lo = result[key]["ci_lower"]
            val = result[key]["value"]
            hi = result[key]["ci_upper"]
            assert lo <= val <= hi, f"{key}: CI [{lo}, {hi}] does not contain {val}"
