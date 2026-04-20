"""Tests for the shared evaluation harness (metrics, bootstrap CI, reporting).

Covers:
- Perfect classifier: accuracy=1.0, macro_f1=1.0
- All-same-class classifier: accuracy varies, macro_f1 low
- Known regression values: Pearson r, RMSE against manually computed values
- Bootstrap CI contains point estimate (property-based test, Property 9)
- McNemar's test returns a p-value
- Paired t-test returns a p-value
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.metrics import EvaluationHarness
from src.evaluation.reporting import load_results, save_results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def harness():
    return EvaluationHarness()


# ---------------------------------------------------------------------------
# 9.6 Unit tests — classification metrics
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    def test_perfect_classifier(self, harness):
        y_true = ["correct", "incorrect", "partially_correct", "correct", "incorrect"]
        y_pred = y_true[:]
        result = harness.classification_metrics(y_true, y_pred, bootstrap_n=100)

        assert result["accuracy"]["value"] == pytest.approx(1.0)
        assert result["macro_f1"]["value"] == pytest.approx(1.0)
        assert result["weighted_f1"]["value"] == pytest.approx(1.0)
        for cls, metrics in result["per_class_f1"].items():
            assert metrics["value"] == pytest.approx(1.0), f"F1 for {cls} should be 1.0"

    def test_perfect_classifier_confusion_matrix(self, harness):
        y_true = ["a", "b", "a", "b"]
        y_pred = ["a", "b", "a", "b"]
        result = harness.classification_metrics(y_true, y_pred, bootstrap_n=50)
        cm = result["confusion_matrix"]["matrix"]
        # Should be identity-like: [[2,0],[0,2]]
        assert cm[0][1] == 0
        assert cm[1][0] == 0

    def test_all_same_class_classifier(self, harness):
        """Predicting all 'correct' when truth is mixed → low macro_f1."""
        y_true = ["correct", "incorrect", "incorrect", "partially_correct"]
        y_pred = ["correct", "correct", "correct", "correct"]
        result = harness.classification_metrics(y_true, y_pred, bootstrap_n=100)

        # accuracy = 1/4 = 0.25
        assert result["accuracy"]["value"] == pytest.approx(0.25)
        # macro_f1 should be well below 1.0
        assert result["macro_f1"]["value"] < 0.5

    def test_binary_known_values(self, harness):
        """2-class case with known accuracy."""
        y_true = ["pos", "pos", "neg", "neg"]
        y_pred = ["pos", "neg", "neg", "neg"]
        result = harness.classification_metrics(y_true, y_pred, bootstrap_n=100)
        # 3 correct out of 4
        assert result["accuracy"]["value"] == pytest.approx(0.75)

    def test_ci_bounds_are_valid(self, harness):
        """CI lower <= value <= upper for all metrics."""
        y_true = ["a", "b", "a", "b", "a"]
        y_pred = ["a", "b", "b", "b", "a"]
        result = harness.classification_metrics(y_true, y_pred, bootstrap_n=200)

        for key in ("accuracy", "macro_f1", "weighted_f1"):
            v = result[key]["value"]
            lo = result[key]["ci_lower"]
            hi = result[key]["ci_upper"]
            assert lo <= v <= hi, f"{key}: CI [{lo}, {hi}] does not contain {v}"

    def test_confusion_matrix_labels(self, harness):
        y_true = ["a", "b", "c"]
        y_pred = ["a", "b", "c"]
        result = harness.classification_metrics(y_true, y_pred, bootstrap_n=50)
        assert result["confusion_matrix"]["labels"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 9.6 Unit tests — regression metrics
# ---------------------------------------------------------------------------

class TestRegressionMetrics:
    def test_perfect_regression(self, harness):
        # Use integer values so QWK works (it requires ordinal labels)
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = harness.regression_metrics(y, y, bootstrap_n=100)

        assert result["pearson_r"]["value"] == pytest.approx(1.0)
        assert result["spearman_rho"]["value"] == pytest.approx(1.0)
        assert result["rmse"]["value"] == pytest.approx(0.0, abs=1e-10)
        assert result["mae"]["value"] == pytest.approx(0.0, abs=1e-10)
        assert result["qwk"]["value"] == pytest.approx(1.0)

    def test_known_rmse(self, harness):
        """RMSE of [0,0,0] vs [1,2,3] = sqrt((1+4+9)/3) = sqrt(14/3)."""
        y_true = [0.0, 0.0, 0.0]
        y_pred = [1.0, 2.0, 3.0]
        expected_rmse = math.sqrt((1 + 4 + 9) / 3)
        result = harness.regression_metrics(y_true, y_pred, bootstrap_n=100)
        assert result["rmse"]["value"] == pytest.approx(expected_rmse, rel=1e-5)

    def test_known_mae(self, harness):
        """MAE of [0,0,0] vs [1,2,3] = (1+2+3)/3 = 2.0."""
        y_true = [0.0, 0.0, 0.0]
        y_pred = [1.0, 2.0, 3.0]
        result = harness.regression_metrics(y_true, y_pred, bootstrap_n=100)
        assert result["mae"]["value"] == pytest.approx(2.0, rel=1e-5)

    def test_known_pearson(self, harness):
        """Perfectly anti-correlated → Pearson r = -1."""
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [4.0, 3.0, 2.0, 1.0]
        result = harness.regression_metrics(y_true, y_pred, bootstrap_n=100)
        assert result["pearson_r"]["value"] == pytest.approx(-1.0, abs=1e-5)

    def test_regression_ci_bounds_valid(self, harness):
        """CI lower <= value <= upper for all regression metrics."""
        rng = np.random.default_rng(0)
        # Use integer-valued arrays so QWK works (it requires ordinal labels)
        y_true = rng.integers(0, 5, 30).astype(float).tolist()
        y_pred = np.clip(
            np.array(y_true) + rng.integers(-1, 2, 30), 0, 4
        ).astype(float).tolist()
        result = harness.regression_metrics(y_true, y_pred, bootstrap_n=200)

        for key in ("pearson_r", "spearman_rho", "rmse", "mae", "qwk"):
            v = result[key]["value"]
            lo = result[key]["ci_lower"]
            hi = result[key]["ci_upper"]
            assert lo <= v <= hi, f"{key}: CI [{lo}, {hi}] does not contain {v}"


# ---------------------------------------------------------------------------
# 9.6 Unit tests — compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_mcnemar_returns_pvalue(self, harness):
        y_true = ["a", "b", "a", "b", "a", "b", "a", "b"]
        y_pred_a = ["a", "b", "a", "a", "a", "b", "b", "b"]
        y_pred_b = ["a", "b", "b", "b", "a", "a", "a", "b"]
        result = harness.compare_models(y_true, y_pred_a, y_pred_b, task="classification")
        assert "mcnemar_p" in result
        p = result["mcnemar_p"]
        assert 0.0 <= p <= 1.0

    def test_paired_t_returns_pvalue(self, harness):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred_a = [1.1, 2.1, 3.1, 4.1, 5.1]
        y_pred_b = [1.5, 2.5, 3.5, 4.5, 5.5]
        result = harness.compare_models(y_true, y_pred_a, y_pred_b, task="regression")
        assert "paired_t_p" in result
        p = result["paired_t_p"]
        assert 0.0 <= p <= 1.0

    def test_identical_models_mcnemar_high_p(self, harness):
        """Identical predictions → McNemar p-value should be 1.0 (no difference)."""
        y_true = ["a", "b", "a", "b", "a"]
        y_pred = ["a", "b", "a", "b", "a"]
        result = harness.compare_models(y_true, y_pred, y_pred, task="classification")
        # Both models identical → no discordant pairs → p = 1.0
        assert result["mcnemar_p"] == pytest.approx(1.0)

    def test_invalid_task_raises(self, harness):
        with pytest.raises(ValueError, match="task must be"):
            harness.compare_models([1], [1], [1], task="invalid")


# ---------------------------------------------------------------------------
# 9.6 Unit tests — reporting
# ---------------------------------------------------------------------------

class TestReporting:
    def test_save_and_load_results(self, tmp_path):
        results = {"accuracy": {"value": 0.9, "ci_lower": 0.85, "ci_upper": 0.95}}
        path = save_results(results, "test_output", results_dir=tmp_path)
        assert path.exists()
        loaded = load_results(path)
        assert loaded["accuracy"]["value"] == pytest.approx(0.9)

    def test_save_creates_directory(self, tmp_path):
        subdir = tmp_path / "nested" / "dir"
        results = {"x": 1}
        path = save_results(results, "out", results_dir=subdir)
        assert path.exists()

    def test_save_adds_json_extension(self, tmp_path):
        results = {"x": 1}
        path = save_results(results, "no_ext", results_dir=tmp_path)
        assert path.suffix == ".json"

    def test_saved_file_is_valid_json(self, tmp_path):
        results = {"macro_f1": {"value": 0.8}}
        path = save_results(results, "valid", results_dir=tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert "results" in data
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# 9.7 Property-based test — Property 9: bootstrap CI contains point estimate
# Feature: asag-research-framework, Property 9: for any prediction array,
# bootstrap CI contains the point estimate
# ---------------------------------------------------------------------------

# Strategy: lists of floats for regression bootstrap CI test
_float_lists = st.lists(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=5,
    max_size=50,
)


@given(y_true=_float_lists, noise=_float_lists)
@settings(max_examples=100, deadline=None)
def test_bootstrap_ci_contains_point_estimate_regression(y_true, noise):
    """Property 9: bootstrap CI contains the point estimate.

    **Validates: Requirements 14.3**
    # Feature: asag-research-framework, Property 9: for any prediction array,
    # bootstrap CI contains the point estimate
    """
    # Pad noise to same length as y_true
    n = len(y_true)
    noise_padded = (noise * ((n // len(noise)) + 1))[:n] if noise else [0.0] * n
    y_pred = [yt + nd for yt, nd in zip(y_true, noise_padded)]

    def mae(yt, yp):
        return float(np.mean(np.abs(np.array(yt) - np.array(yp))))

    point_estimate = mae(y_true, y_pred)
    lo, hi = bootstrap_ci(y_true, y_pred, mae, n=200)

    assert lo <= point_estimate <= hi, (
        f"Bootstrap CI [{lo}, {hi}] does not contain point estimate {point_estimate}"
    )


# Strategy: lists of class labels for classification bootstrap CI test
_label_lists = st.lists(
    st.sampled_from(["correct", "incorrect", "partially_correct"]),
    min_size=5,
    max_size=40,
)


@given(y_true=_label_lists, y_pred=_label_lists)
@settings(max_examples=100, deadline=None)
def test_bootstrap_ci_contains_point_estimate_classification(y_true, y_pred):
    """Property 9: bootstrap CI contains the point estimate (classification).

    **Validates: Requirements 14.3**
    # Feature: asag-research-framework, Property 9: for any prediction array,
    # bootstrap CI contains the point estimate
    """
    from sklearn.metrics import accuracy_score

    # Pad y_pred to same length as y_true
    n = len(y_true)
    y_pred_padded = (y_pred * ((n // len(y_pred)) + 1))[:n] if y_pred else ["incorrect"] * n

    def acc(yt, yp):
        return accuracy_score(yt, yp)

    point_estimate = acc(y_true, y_pred_padded)
    lo, hi = bootstrap_ci(y_true, y_pred_padded, acc, n=200)

    assert lo <= point_estimate <= hi, (
        f"Bootstrap CI [{lo}, {hi}] does not contain point estimate {point_estimate}"
    )
