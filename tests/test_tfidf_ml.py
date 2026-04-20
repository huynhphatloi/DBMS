"""Integration tests for TF-IDF + Traditional ML baseline (src/grading/baselines/tfidf_ml.py).

Covers:
- LR classifier: train on small dataset, verify predictions are valid labels
- SVM (linear): train on small dataset, verify predictions are valid labels
- Random Forest: train on small dataset, verify predictions are valid labels
- Gradient Boosting: train on small dataset, verify predictions are valid labels
- Regression model: train on small dataset, verify regression metrics are computed
- evaluate_tfidf_model() returns macro_f1, weighted_f1, accuracy with CI fields
- Regression evaluation returns pearson_r, rmse with CI fields
"""

from __future__ import annotations

import pytest

from src.data.schema import UnifiedRecord
from src.grading.baselines.tfidf_ml import (
    TfidfMLClassifier,
    TfidfMLRegressor,
    evaluate_tfidf_model,
    evaluate_tfidf_regressor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    reference: str,
    student: str,
    label_2way: str = "correct",
    score_normalized: float | None = None,
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
        label_2way=label_2way,
        score_normalized=score_normalized,
    )


# 16 records with varied text pairs for robust training
_TRAINING_PAIRS = [
    ("photosynthesis converts light energy into chemical energy stored in glucose",
     "photosynthesis uses sunlight to produce glucose and oxygen", "correct", 0.9),
    ("the mitochondria is the powerhouse of the cell producing ATP",
     "mitochondria generate ATP through cellular respiration", "correct", 0.85),
    ("water boils at 100 degrees celsius at standard atmospheric pressure",
     "water boils at 100 degrees celsius", "correct", 0.8),
    ("gravity is a force that attracts objects with mass toward each other",
     "gravity pulls objects with mass together", "correct", 0.75),
    ("the earth orbits the sun once every 365 days completing one year",
     "earth takes 365 days to orbit the sun", "correct", 0.9),
    ("cells divide through mitosis producing two identical daughter cells",
     "mitosis produces two genetically identical cells", "correct", 0.85),
    ("the speed of light in a vacuum is approximately 3 times 10 to the 8 meters per second",
     "light travels at about 300 million meters per second in vacuum", "correct", 0.8),
    ("DNA carries genetic information encoded in sequences of nucleotide bases",
     "DNA stores genetic information using nucleotide sequences", "correct", 0.9),
    ("photosynthesis converts light energy into chemical energy stored in glucose",
     "dogs bark loudly at strangers in the park", "incorrect", 0.1),
    ("the mitochondria is the powerhouse of the cell producing ATP",
     "the sky is blue because of rayleigh scattering", "incorrect", 0.05),
    ("water boils at 100 degrees celsius at standard atmospheric pressure",
     "rocks are formed by geological processes over millions of years", "incorrect", 0.1),
    ("gravity is a force that attracts objects with mass toward each other",
     "plants need sunlight water and carbon dioxide to grow", "incorrect", 0.05),
    ("the earth orbits the sun once every 365 days completing one year",
     "the ocean is very deep and contains many species of fish", "incorrect", 0.1),
    ("cells divide through mitosis producing two identical daughter cells",
     "the moon orbits the earth and causes tides", "incorrect", 0.05),
    ("the speed of light in a vacuum is approximately 3 times 10 to the 8 meters per second",
     "volcanoes erupt due to pressure from magma beneath the surface", "incorrect", 0.1),
    ("DNA carries genetic information encoded in sequences of nucleotide bases",
     "the weather changes due to atmospheric pressure differences", "incorrect", 0.05),
]


def _make_training_records() -> list[UnifiedRecord]:
    return [
        _make_record(ref, stu, lbl, score, f"GEN_{i:04d}")
        for i, (ref, stu, lbl, score) in enumerate(_TRAINING_PAIRS)
    ]


VALID_LABELS = {"correct", "incorrect"}


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

class TestTfidfMLClassifierLR:
    def test_fit_and_predict_valid_labels(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="logistic_regression")
        clf.fit(records, "label_2way")
        preds = clf.predict(records)
        assert len(preds) == len(records)
        assert all(p in VALID_LABELS for p in preds)

    def test_predict_proba_shape_and_range(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="logistic_regression")
        clf.fit(records, "label_2way")
        proba = clf.predict_proba(records)
        assert len(proba) == len(records)
        for row in proba:
            assert len(row) == 2
            assert abs(sum(row) - 1.0) < 1e-6
            assert all(0.0 <= p <= 1.0 for p in row)


class TestTfidfMLClassifierSVMLinear:
    def test_fit_and_predict_valid_labels(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="svm_linear")
        clf.fit(records, "label_2way")
        preds = clf.predict(records)
        assert len(preds) == len(records)
        assert all(p in VALID_LABELS for p in preds)

    def test_predict_proba_shape_and_range(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="svm_linear")
        clf.fit(records, "label_2way")
        proba = clf.predict_proba(records)
        assert len(proba) == len(records)
        for row in proba:
            assert len(row) == 2
            assert abs(sum(row) - 1.0) < 1e-6


class TestTfidfMLClassifierSVMRBF:
    def test_fit_and_predict_valid_labels(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="svm_rbf")
        clf.fit(records, "label_2way")
        preds = clf.predict(records)
        assert len(preds) == len(records)
        assert all(p in VALID_LABELS for p in preds)


class TestTfidfMLClassifierRandomForest:
    def test_fit_and_predict_valid_labels(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="random_forest")
        clf.fit(records, "label_2way")
        preds = clf.predict(records)
        assert len(preds) == len(records)
        assert all(p in VALID_LABELS for p in preds)

    def test_predict_proba_shape_and_range(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="random_forest")
        clf.fit(records, "label_2way")
        proba = clf.predict_proba(records)
        assert len(proba) == len(records)
        for row in proba:
            assert len(row) == 2
            assert abs(sum(row) - 1.0) < 1e-6


class TestTfidfMLClassifierGradientBoosting:
    def test_fit_and_predict_valid_labels(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="gradient_boosting")
        clf.fit(records, "label_2way")
        preds = clf.predict(records)
        assert len(preds) == len(records)
        assert all(p in VALID_LABELS for p in preds)


class TestTfidfMLClassifierInvalidName:
    def test_invalid_classifier_raises(self):
        with pytest.raises(ValueError, match="classifier must be one of"):
            TfidfMLClassifier(classifier="nonexistent")


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

class TestTfidfMLRegressor:
    def test_ridge_fit_and_predict(self):
        records = _make_training_records()
        reg = TfidfMLRegressor(regressor="ridge")
        reg.fit(records, "score_normalized")
        preds = reg.predict(records)
        assert len(preds) == len(records)
        assert all(isinstance(p, float) for p in preds)

    def test_svr_fit_and_predict(self):
        records = _make_training_records()
        reg = TfidfMLRegressor(regressor="svr")
        reg.fit(records, "score_normalized")
        preds = reg.predict(records)
        assert len(preds) == len(records)
        assert all(isinstance(p, float) for p in preds)

    def test_invalid_regressor_raises(self):
        with pytest.raises(ValueError, match="regressor must be one of"):
            TfidfMLRegressor(regressor="nonexistent")

    def test_regression_metrics_computed(self):
        records = _make_training_records()
        reg = TfidfMLRegressor(regressor="ridge")
        reg.fit(records, "score_normalized")
        result = evaluate_tfidf_regressor(reg, records, "score_normalized", bootstrap_n=50)
        assert "pearson_r" in result
        assert "spearman_rho" in result
        assert "rmse" in result
        assert "mae" in result
        assert "qwk" in result

    def test_regression_metrics_have_ci_fields(self):
        records = _make_training_records()
        reg = TfidfMLRegressor(regressor="ridge")
        reg.fit(records, "score_normalized")
        result = evaluate_tfidf_regressor(reg, records, "score_normalized", bootstrap_n=50)
        for key in ("pearson_r", "rmse"):
            assert "value" in result[key]
            assert "ci_lower" in result[key]
            assert "ci_upper" in result[key]

    def test_regression_ci_contains_point_estimate(self):
        records = _make_training_records()
        reg = TfidfMLRegressor(regressor="ridge")
        reg.fit(records, "score_normalized")
        result = evaluate_tfidf_regressor(reg, records, "score_normalized", bootstrap_n=50)
        for key in ("pearson_r", "rmse"):
            lo = result[key]["ci_lower"]
            val = result[key]["value"]
            hi = result[key]["ci_upper"]
            assert lo <= val <= hi, f"{key}: CI [{lo}, {hi}] does not contain {val}"


# ---------------------------------------------------------------------------
# evaluate_tfidf_model integration tests
# ---------------------------------------------------------------------------

class TestEvaluateTfidfModel:
    def test_returns_required_keys(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="logistic_regression")
        clf.fit(records, "label_2way")
        result = evaluate_tfidf_model(clf, records, "label_2way", bootstrap_n=50)
        assert "macro_f1" in result
        assert "weighted_f1" in result
        assert "accuracy" in result

    def test_metrics_have_ci_fields(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="logistic_regression")
        clf.fit(records, "label_2way")
        result = evaluate_tfidf_model(clf, records, "label_2way", bootstrap_n=50)
        for key in ("macro_f1", "weighted_f1", "accuracy"):
            assert "value" in result[key]
            assert "ci_lower" in result[key]
            assert "ci_upper" in result[key]

    def test_metric_values_in_unit_interval(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="logistic_regression")
        clf.fit(records, "label_2way")
        result = evaluate_tfidf_model(clf, records, "label_2way", bootstrap_n=50)
        for key in ("macro_f1", "weighted_f1", "accuracy"):
            v = result[key]["value"]
            assert 0.0 <= v <= 1.0, f"{key} value {v} out of [0,1]"

    def test_ci_lower_le_value_le_upper(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="logistic_regression")
        clf.fit(records, "label_2way")
        result = evaluate_tfidf_model(clf, records, "label_2way", bootstrap_n=50)
        for key in ("macro_f1", "weighted_f1", "accuracy"):
            lo = result[key]["ci_lower"]
            val = result[key]["value"]
            hi = result[key]["ci_upper"]
            assert lo <= val <= hi, f"{key}: CI [{lo}, {hi}] does not contain {val}"

    def test_svm_linear_evaluate(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="svm_linear")
        clf.fit(records, "label_2way")
        result = evaluate_tfidf_model(clf, records, "label_2way", bootstrap_n=50)
        assert "macro_f1" in result
        assert "weighted_f1" in result
        assert "accuracy" in result

    def test_random_forest_evaluate(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="random_forest")
        clf.fit(records, "label_2way")
        result = evaluate_tfidf_model(clf, records, "label_2way", bootstrap_n=50)
        assert "macro_f1" in result

    def test_gradient_boosting_evaluate(self):
        records = _make_training_records()
        clf = TfidfMLClassifier(classifier="gradient_boosting")
        clf.fit(records, "label_2way")
        result = evaluate_tfidf_model(clf, records, "label_2way", bootstrap_n=50)
        assert "macro_f1" in result
