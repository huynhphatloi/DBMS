"""Unit tests for the CalibrationAnalyzer.

Verifies:
- ECE and MCE computation on known probability outputs
- ECE is in [0, 1]
- Reliability diagram generation (10 equal-width bins)
- AUROC of confidence vs. correctness
- ΔECE = ECE_adversarial - ECE_clean
- Accuracy-coverage curve (threshold sweep 0.0→1.0, step 0.05)
- Optimal abstention threshold (lowest threshold achieving 95% accuracy)

Validates: Requirements 24, 25
"""

from __future__ import annotations

import numpy as np
import pytest

from src.robustness.calibration import (
    CalibrationAnalyzer,
    CalibrationResult,
    ReliabilityDiagram,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_analyzer(**kwargs) -> CalibrationAnalyzer:
    """Create a CalibrationAnalyzer with default settings."""
    defaults = dict(n_bins=10, threshold_step=0.05, target_accuracy=0.95)
    defaults.update(kwargs)
    return CalibrationAnalyzer(**defaults)


# ── Tests: ECE and MCE computation (Task 30.1) ───────────────────────

class TestECEandMCE:
    def test_perfectly_calibrated_model(self):
        """A perfectly calibrated model has ECE = 0."""
        analyzer = _make_analyzer()
        # All predictions correct with confidence 1.0
        correctness = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        confidences = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        ece = analyzer.compute_ece(correctness, confidences)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_completely_miscalibrated_model(self):
        """All wrong predictions with confidence 1.0 → ECE = 1.0."""
        analyzer = _make_analyzer()
        correctness = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        confidences = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        ece = analyzer.compute_ece(correctness, confidences)
        assert ece == pytest.approx(1.0, abs=1e-9)

    def test_mce_equals_max_bin_gap(self):
        """MCE is the maximum |acc - conf| across bins."""
        analyzer = _make_analyzer()
        correctness = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        confidences = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        mce = analyzer.compute_mce(correctness, confidences)
        assert mce == pytest.approx(1.0, abs=1e-9)

    def test_ece_in_unit_interval(self):
        """ECE should always be in [0, 1]."""
        analyzer = _make_analyzer()
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(10, 100)
            correctness = rng.choice([0.0, 1.0], size=n)
            confidences = rng.uniform(0.0, 1.0, size=n)
            ece = analyzer.compute_ece(correctness, confidences)
            assert 0.0 <= ece <= 1.0, f"ECE={ece} out of [0,1]"

    def test_mce_in_unit_interval(self):
        """MCE should always be in [0, 1]."""
        analyzer = _make_analyzer()
        rng = np.random.default_rng(123)
        for _ in range(20):
            n = rng.integers(10, 100)
            correctness = rng.choice([0.0, 1.0], size=n)
            confidences = rng.uniform(0.0, 1.0, size=n)
            mce = analyzer.compute_mce(correctness, confidences)
            assert 0.0 <= mce <= 1.0, f"MCE={mce} out of [0,1]"

    def test_ece_empty_input(self):
        """ECE of empty input is 0."""
        analyzer = _make_analyzer()
        ece = analyzer.compute_ece(np.array([]), np.array([]))
        assert ece == 0.0

    def test_mce_empty_input(self):
        """MCE of empty input is 0."""
        analyzer = _make_analyzer()
        mce = analyzer.compute_mce(np.array([]), np.array([]))
        assert mce == 0.0

    def test_ece_known_value(self):
        """ECE on a known distribution matches hand-computed value."""
        # 10 samples, all in the [0.9, 1.0] bin:
        # 8 correct, 2 wrong → acc=0.8, conf≈0.95 → gap=0.15
        # weight = 10/10 = 1.0 → ECE = 0.15
        analyzer = _make_analyzer()
        correctness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=float)
        confidences = np.array([0.95] * 10)
        ece = analyzer.compute_ece(correctness, confidences)
        assert ece == pytest.approx(0.15, abs=1e-9)

    def test_mce_known_value(self):
        """MCE on same known distribution."""
        analyzer = _make_analyzer()
        correctness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=float)
        confidences = np.array([0.95] * 10)
        mce = analyzer.compute_mce(correctness, confidences)
        assert mce == pytest.approx(0.15, abs=1e-9)

    def test_mce_geq_ece(self):
        """MCE >= ECE always (MCE is the max, ECE is the weighted average)."""
        analyzer = _make_analyzer()
        rng = np.random.default_rng(99)
        for _ in range(20):
            n = rng.integers(10, 100)
            correctness = rng.choice([0.0, 1.0], size=n)
            confidences = rng.uniform(0.0, 1.0, size=n)
            ece = analyzer.compute_ece(correctness, confidences)
            mce = analyzer.compute_mce(correctness, confidences)
            assert mce >= ece - 1e-9


# ── Tests: Reliability diagram (Task 30.2) ───────────────────────────

class TestReliabilityDiagram:
    def test_diagram_has_10_bins(self):
        """Reliability diagram should have exactly 10 bins."""
        analyzer = _make_analyzer()
        correctness = np.array([1.0, 0.0, 1.0, 0.0])
        confidences = np.array([0.9, 0.1, 0.8, 0.2])
        diagram = analyzer.compute_reliability_diagram(correctness, confidences)
        assert len(diagram.bin_midpoints) == 10
        assert len(diagram.bin_accuracies) == 10
        assert len(diagram.bin_confidences) == 10
        assert len(diagram.bin_counts) == 10

    def test_bin_midpoints_are_correct(self):
        """Bin midpoints should be 0.05, 0.15, ..., 0.95."""
        analyzer = _make_analyzer()
        correctness = np.array([1.0])
        confidences = np.array([0.5])
        diagram = analyzer.compute_reliability_diagram(correctness, confidences)
        expected = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        for actual, exp in zip(diagram.bin_midpoints, expected):
            assert actual == pytest.approx(exp, abs=1e-9)

    def test_bin_counts_sum_to_n(self):
        """Total bin counts should equal the number of samples."""
        analyzer = _make_analyzer()
        correctness = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        confidences = np.array([0.95, 0.05, 0.75, 0.25, 0.55])
        diagram = analyzer.compute_reliability_diagram(correctness, confidences)
        assert sum(diagram.bin_counts) == 5

    def test_empty_bins_have_zero_accuracy(self):
        """Empty bins should report accuracy=0 and confidence=0."""
        analyzer = _make_analyzer()
        # All samples in the last bin
        correctness = np.array([1.0, 1.0])
        confidences = np.array([0.95, 0.99])
        diagram = analyzer.compute_reliability_diagram(correctness, confidences)
        # First 9 bins should be empty
        for i in range(9):
            assert diagram.bin_counts[i] == 0
            assert diagram.bin_accuracies[i] == 0.0
        assert diagram.bin_counts[9] == 2
        assert diagram.bin_accuracies[9] == pytest.approx(1.0)


# ── Tests: AUROC (Task 30.3) ─────────────────────────────────────────

class TestAUROC:
    def test_perfect_auroc(self):
        """Perfect separation → AUROC = 1.0."""
        analyzer = _make_analyzer()
        # Correct predictions have high confidence, wrong have low
        correctness = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        confidences = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        auroc = analyzer.compute_auroc(correctness, confidences)
        assert auroc == pytest.approx(1.0, abs=1e-9)

    def test_random_auroc(self):
        """All same correctness → AUROC = 0.5."""
        analyzer = _make_analyzer()
        correctness = np.array([1.0, 1.0, 1.0, 1.0])
        confidences = np.array([0.9, 0.8, 0.7, 0.6])
        auroc = analyzer.compute_auroc(correctness, confidences)
        assert auroc == pytest.approx(0.5)

    def test_worst_auroc(self):
        """Inverted separation → AUROC = 0.0."""
        analyzer = _make_analyzer()
        # Wrong predictions have high confidence, correct have low
        correctness = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        confidences = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        auroc = analyzer.compute_auroc(correctness, confidences)
        assert auroc == pytest.approx(0.0, abs=1e-9)

    def test_auroc_empty(self):
        """Empty input → AUROC = 0."""
        analyzer = _make_analyzer()
        auroc = analyzer.compute_auroc(np.array([]), np.array([]))
        assert auroc == 0.0

    def test_auroc_in_unit_interval(self):
        """AUROC should be in [0, 1]."""
        analyzer = _make_analyzer()
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(10, 50)
            correctness = rng.choice([0.0, 1.0], size=n)
            confidences = rng.uniform(0.0, 1.0, size=n)
            # Need at least one of each class
            if correctness.sum() == 0 or correctness.sum() == n:
                correctness[0] = 1.0
                correctness[-1] = 0.0
            auroc = analyzer.compute_auroc(correctness, confidences)
            assert 0.0 <= auroc <= 1.0


# ── Tests: ΔECE (Task 30.4) ──────────────────────────────────────────

class TestDeltaECE:
    def test_delta_ece_zero_when_same(self):
        """ΔECE = 0 when clean and adversarial are identical."""
        analyzer = _make_analyzer()
        y_true = ["correct", "incorrect", "correct"]
        y_pred = ["correct", "incorrect", "correct"]
        confs = [0.9, 0.8, 0.95]
        delta = analyzer.compute_delta_ece(
            y_true, y_pred, confs, y_true, y_pred, confs,
        )
        assert delta == pytest.approx(0.0, abs=1e-9)

    def test_delta_ece_positive_when_adversarial_worse(self):
        """ΔECE > 0 when adversarial calibration is worse."""
        analyzer = _make_analyzer()
        # Clean: well-calibrated
        y_true_c = ["correct"] * 10
        y_pred_c = ["correct"] * 10
        confs_c = [1.0] * 10
        # Adversarial: miscalibrated (all wrong but high confidence)
        y_true_a = ["correct"] * 10
        y_pred_a = ["incorrect"] * 10
        confs_a = [0.95] * 10
        delta = analyzer.compute_delta_ece(
            y_true_c, y_pred_c, confs_c,
            y_true_a, y_pred_a, confs_a,
        )
        assert delta > 0

    def test_delta_ece_multi_model(self):
        """ΔECE per model returns correct structure."""
        analyzer = _make_analyzer()
        clean = {
            "model_a": (
                ["correct", "incorrect"],
                ["correct", "incorrect"],
                [0.9, 0.8],
            ),
            "model_b": (
                ["correct", "incorrect"],
                ["correct", "incorrect"],
                [0.95, 0.85],
            ),
        }
        adv = {
            "model_a": (
                ["correct", "incorrect"],
                ["incorrect", "incorrect"],
                [0.9, 0.8],
            ),
            "model_b": (
                ["correct", "incorrect"],
                ["incorrect", "correct"],
                [0.95, 0.85],
            ),
        }
        result = analyzer.compute_delta_ece_multi_model(clean, adv)
        assert "model_a" in result
        assert "model_b" in result
        assert isinstance(result["model_a"], float)
        assert isinstance(result["model_b"], float)


# ── Tests: Accuracy-coverage curve (Task 30.5) ───────────────────────

class TestAccuracyCoverageCurve:
    def test_curve_has_correct_thresholds(self):
        """Curve should have thresholds from 0.0 to 1.0 in steps of 0.05."""
        analyzer = _make_analyzer()
        correctness = np.array([1.0, 0.0, 1.0, 0.0])
        confidences = np.array([0.9, 0.1, 0.8, 0.2])
        curve = analyzer.compute_accuracy_coverage_curve(correctness, confidences)
        thresholds = [p.threshold for p in curve]
        expected = [round(t, 2) for t in np.arange(0.0, 1.05, 0.05)]
        assert len(thresholds) == len(expected)
        for actual, exp in zip(thresholds, expected):
            assert actual == pytest.approx(exp, abs=0.01)

    def test_coverage_at_zero_threshold_is_one(self):
        """At threshold=0.0, all predictions are covered → coverage=1.0."""
        analyzer = _make_analyzer()
        correctness = np.array([1.0, 0.0, 1.0])
        confidences = np.array([0.9, 0.5, 0.7])
        curve = analyzer.compute_accuracy_coverage_curve(correctness, confidences)
        assert curve[0].threshold == pytest.approx(0.0)
        assert curve[0].coverage == pytest.approx(1.0)

    def test_coverage_decreases_with_threshold(self):
        """Coverage should be non-increasing as threshold increases."""
        analyzer = _make_analyzer()
        rng = np.random.default_rng(42)
        correctness = rng.choice([0.0, 1.0], size=50)
        confidences = rng.uniform(0.0, 1.0, size=50)
        curve = analyzer.compute_accuracy_coverage_curve(correctness, confidences)
        for i in range(1, len(curve)):
            assert curve[i].coverage <= curve[i - 1].coverage + 1e-9

    def test_empty_input_returns_empty_curve(self):
        """Empty input → empty curve."""
        analyzer = _make_analyzer()
        curve = analyzer.compute_accuracy_coverage_curve(np.array([]), np.array([]))
        assert curve == []

    def test_accuracy_at_high_threshold(self):
        """At high threshold, only high-confidence predictions are covered."""
        analyzer = _make_analyzer()
        # 3 correct with high conf, 3 wrong with low conf
        correctness = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        confidences = np.array([0.95, 0.90, 0.85, 0.15, 0.10, 0.05])
        curve = analyzer.compute_accuracy_coverage_curve(correctness, confidences)
        # At threshold 0.8, only the 3 correct predictions are covered
        point_80 = [p for p in curve if p.threshold == pytest.approx(0.8)][0]
        assert point_80.accuracy == pytest.approx(1.0)
        assert point_80.coverage == pytest.approx(0.5)


# ── Tests: Optimal abstention threshold (Task 30.6) ──────────────────

class TestOptimalAbstentionThreshold:
    def test_finds_lowest_threshold_for_95_accuracy(self):
        """Should find the lowest threshold achieving 95% accuracy."""
        analyzer = _make_analyzer()
        # 8 correct with high conf, 2 wrong with low conf
        correctness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=float)
        confidences = np.array([
            0.99, 0.95, 0.90, 0.85, 0.80,
            0.75, 0.70, 0.65, 0.15, 0.05,
        ])
        threshold = analyzer.find_optimal_abstention_threshold(
            correctness, confidences,
        )
        assert threshold is not None
        # At threshold 0.0, accuracy = 8/10 = 0.8 < 0.95
        # Need to find where accuracy on covered >= 0.95
        # Verify the threshold achieves >= 95% accuracy
        mask = confidences >= threshold
        if mask.sum() > 0:
            acc = correctness[mask].mean()
            assert acc >= 0.95

    def test_returns_none_when_impossible(self):
        """Returns None when no threshold achieves target accuracy."""
        analyzer = _make_analyzer()
        # All wrong predictions with uniform confidence
        correctness = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        threshold = analyzer.find_optimal_abstention_threshold(
            correctness, confidences,
        )
        assert threshold is None

    def test_threshold_zero_when_already_accurate(self):
        """If accuracy is already >= 95% at threshold 0, return 0.0."""
        analyzer = _make_analyzer()
        # All correct
        correctness = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        threshold = analyzer.find_optimal_abstention_threshold(
            correctness, confidences,
        )
        assert threshold == pytest.approx(0.0)


# ── Tests: Full analyze() integration (Task 30.7) ────────────────────

class TestAnalyzeIntegration:
    def test_analyze_returns_calibration_result(self):
        """Full analyze() returns a CalibrationResult with all fields."""
        analyzer = _make_analyzer()
        y_true = ["correct", "incorrect", "correct", "incorrect", "correct"]
        y_pred = ["correct", "incorrect", "correct", "correct", "incorrect"]
        confs = [0.9, 0.8, 0.95, 0.6, 0.4]
        result = analyzer.analyze(y_true, y_pred, confs)

        assert isinstance(result, CalibrationResult)
        assert 0.0 <= result.ece <= 1.0
        assert 0.0 <= result.mce <= 1.0
        assert isinstance(result.reliability_diagram, ReliabilityDiagram)
        assert len(result.reliability_diagram.bin_midpoints) == 10
        assert isinstance(result.auroc, float)
        assert len(result.accuracy_coverage_curve) > 0

    def test_analyze_with_string_labels(self):
        """analyze() works with string labels for y_true and y_pred."""
        analyzer = _make_analyzer()
        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "B", "B", "A"]
        confs = [0.9, 0.85, 0.6, 0.55]
        result = analyzer.analyze(y_true, y_pred, confs)
        assert 0.0 <= result.ece <= 1.0

    def test_ece_bounded_on_random_data(self):
        """ECE is always in [0, 1] on random data."""
        analyzer = _make_analyzer()
        rng = np.random.default_rng(42)
        labels = ["correct", "incorrect", "partially_correct"]
        for _ in range(10):
            n = rng.integers(20, 100)
            y_true = [labels[i] for i in rng.integers(0, 3, size=n)]
            y_pred = [labels[i] for i in rng.integers(0, 3, size=n)]
            confs = rng.uniform(0.0, 1.0, size=n).tolist()
            result = analyzer.analyze(y_true, y_pred, confs)
            assert 0.0 <= result.ece <= 1.0, f"ECE={result.ece} out of [0,1]"
            assert 0.0 <= result.mce <= 1.0, f"MCE={result.mce} out of [0,1]"
