"""Calibration Analyzer for grading model confidence evaluation.

Measures how well a model's confidence scores reflect its actual accuracy.

Supports:
- ECE (Expected Calibration Error) and MCE (Maximum Calibration Error)
- Reliability diagrams (10 equal-width bins)
- AUROC of confidence vs. correctness
- ΔECE = ECE_adversarial - ECE_clean per model
- Accuracy-coverage curves (threshold sweep 0.0→1.0, step 0.05)
- Optimal abstention threshold (lowest threshold achieving 95% accuracy)

Validates: Requirements 24, 25
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityDiagram:
    """Data for a reliability diagram with equal-width bins."""

    bin_midpoints: list[float]
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]


@dataclass
class AccuracyCoveragePoint:
    """A single point on the accuracy-coverage curve."""

    threshold: float
    accuracy: float
    coverage: float


@dataclass
class CalibrationResult:
    """Full calibration analysis result for a single model/dataset."""

    ece: float
    mce: float
    reliability_diagram: ReliabilityDiagram
    auroc: float
    accuracy_coverage_curve: list[AccuracyCoveragePoint]
    optimal_abstention_threshold: float | None


class CalibrationAnalyzer:
    """Analyze calibration of grading model confidence scores.

    Parameters
    ----------
    n_bins : int
        Number of equal-width bins for ECE/MCE and reliability diagrams.
        Default is 10.
    threshold_min : float
        Minimum confidence threshold for accuracy-coverage sweep.
    threshold_max : float
        Maximum confidence threshold for accuracy-coverage sweep.
    threshold_step : float
        Step size for the threshold sweep.
    target_accuracy : float
        Target accuracy for optimal abstention threshold.
    """

    def __init__(
        self,
        n_bins: int = 10,
        threshold_min: float = 0.0,
        threshold_max: float = 1.0,
        threshold_step: float = 0.05,
        target_accuracy: float = 0.95,
    ) -> None:
        self.n_bins = n_bins
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_step = threshold_step
        self.target_accuracy = target_accuracy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        y_true: list[str],
        y_pred: list[str],
        confidences: list[float],
    ) -> CalibrationResult:
        """Run full calibration analysis.

        Parameters
        ----------
        y_true : list[str]
            True labels.
        y_pred : list[str]
            Predicted labels.
        confidences : list[float]
            Confidence scores for each prediction (in [0, 1]).

        Returns
        -------
        CalibrationResult
        """
        correctness = [
            1.0 if yt == yp else 0.0
            for yt, yp in zip(y_true, y_pred)
        ]
        conf_arr = np.array(confidences, dtype=np.float64)
        correct_arr = np.array(correctness, dtype=np.float64)

        ece = self.compute_ece(correct_arr, conf_arr)
        mce = self.compute_mce(correct_arr, conf_arr)
        diagram = self.compute_reliability_diagram(correct_arr, conf_arr)
        auroc = self.compute_auroc(correct_arr, conf_arr)
        acc_cov = self.compute_accuracy_coverage_curve(correct_arr, conf_arr)
        opt_threshold = self.find_optimal_abstention_threshold(
            correct_arr, conf_arr,
        )

        return CalibrationResult(
            ece=ece,
            mce=mce,
            reliability_diagram=diagram,
            auroc=auroc,
            accuracy_coverage_curve=acc_cov,
            optimal_abstention_threshold=opt_threshold,
        )

    def compute_ece(
        self,
        correctness: np.ndarray,
        confidences: np.ndarray,
    ) -> float:
        """Compute Expected Calibration Error.

        ECE = Σ (|B_i|/n) * |acc(B_i) - conf(B_i)|

        Uses ``n_bins`` equal-width bins over [0, 1].
        """
        n = len(correctness)
        if n == 0:
            return 0.0

        bin_boundaries = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            if i == self.n_bins - 1:
                # Last bin includes the right boundary
                mask = (confidences >= lower) & (confidences <= upper)
            else:
                mask = (confidences >= lower) & (confidences < upper)

            bin_size = mask.sum()
            if bin_size == 0:
                continue

            bin_acc = correctness[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

        return float(ece)

    def compute_mce(
        self,
        correctness: np.ndarray,
        confidences: np.ndarray,
    ) -> float:
        """Compute Maximum Calibration Error.

        MCE = max over bins of |acc(B_i) - conf(B_i)|
        """
        n = len(correctness)
        if n == 0:
            return 0.0

        bin_boundaries = np.linspace(0.0, 1.0, self.n_bins + 1)
        mce = 0.0

        for i in range(self.n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            if i == self.n_bins - 1:
                mask = (confidences >= lower) & (confidences <= upper)
            else:
                mask = (confidences >= lower) & (confidences < upper)

            bin_size = mask.sum()
            if bin_size == 0:
                continue

            bin_acc = correctness[mask].mean()
            bin_conf = confidences[mask].mean()
            mce = max(mce, abs(bin_acc - bin_conf))

        return float(mce)

    def compute_reliability_diagram(
        self,
        correctness: np.ndarray,
        confidences: np.ndarray,
    ) -> ReliabilityDiagram:
        """Generate reliability diagram data with equal-width bins.

        Returns bin midpoints, accuracies, mean confidences, and counts.
        """
        bin_boundaries = np.linspace(0.0, 1.0, self.n_bins + 1)
        midpoints: list[float] = []
        accuracies: list[float] = []
        mean_confs: list[float] = []
        counts: list[int] = []

        for i in range(self.n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            midpoints.append(float((lower + upper) / 2.0))

            if i == self.n_bins - 1:
                mask = (confidences >= lower) & (confidences <= upper)
            else:
                mask = (confidences >= lower) & (confidences < upper)

            bin_size = int(mask.sum())
            counts.append(bin_size)

            if bin_size == 0:
                accuracies.append(0.0)
                mean_confs.append(0.0)
            else:
                accuracies.append(float(correctness[mask].mean()))
                mean_confs.append(float(confidences[mask].mean()))

        return ReliabilityDiagram(
            bin_midpoints=midpoints,
            bin_accuracies=accuracies,
            bin_confidences=mean_confs,
            bin_counts=counts,
        )

    def compute_auroc(
        self,
        correctness: np.ndarray,
        confidences: np.ndarray,
    ) -> float:
        """Compute AUROC of confidence vs. correctness.

        Positive class = model was correct (correctness == 1).
        Uses the trapezoidal rule on sorted thresholds.
        """
        if len(correctness) == 0:
            return 0.0

        # If all predictions are correct or all incorrect, AUROC is undefined
        # but we return 0.5 (random) for all-same and handle edge cases.
        unique_correct = np.unique(correctness)
        if len(unique_correct) == 1:
            return 0.5

        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(correctness, confidences))
        except (ImportError, ValueError):
            # Manual computation if sklearn unavailable or error
            return self._manual_auroc(correctness, confidences)

    @staticmethod
    def _manual_auroc(
        correctness: np.ndarray,
        confidences: np.ndarray,
    ) -> float:
        """Manual AUROC via trapezoidal rule."""
        sorted_indices = np.argsort(-confidences)
        sorted_correct = correctness[sorted_indices]

        n_pos = correctness.sum()
        n_neg = len(correctness) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_prev = 0.0
        fpr_prev = 0.0
        auc = 0.0
        tp = 0.0
        fp = 0.0

        for c in sorted_correct:
            if c == 1.0:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
            tpr_prev = tpr
            fpr_prev = fpr

        return float(auc)

    def compute_accuracy_coverage_curve(
        self,
        correctness: np.ndarray,
        confidences: np.ndarray,
    ) -> list[AccuracyCoveragePoint]:
        """Compute accuracy-coverage curve by sweeping confidence thresholds.

        For each threshold t in [threshold_min, threshold_max] with step
        threshold_step, compute:
        - accuracy: fraction correct among predictions with confidence >= t
        - coverage: fraction of predictions with confidence >= t
        """
        n = len(correctness)
        if n == 0:
            return []

        points: list[AccuracyCoveragePoint] = []
        # Generate thresholds: 0.0, 0.05, 0.10, ..., 1.0
        thresholds = np.arange(
            self.threshold_min,
            self.threshold_max + self.threshold_step / 2,
            self.threshold_step,
        )

        for t in thresholds:
            t_val = float(round(t, 10))
            mask = confidences >= t_val
            covered = int(mask.sum())
            coverage = covered / n

            if covered == 0:
                accuracy = 0.0
            else:
                accuracy = float(correctness[mask].mean())

            points.append(AccuracyCoveragePoint(
                threshold=round(t_val, 2),
                accuracy=accuracy,
                coverage=coverage,
            ))

        return points

    def find_optimal_abstention_threshold(
        self,
        correctness: np.ndarray,
        confidences: np.ndarray,
    ) -> float | None:
        """Find the lowest threshold achieving target accuracy on covered predictions.

        Returns None if no threshold achieves the target accuracy.
        """
        curve = self.compute_accuracy_coverage_curve(correctness, confidences)

        for point in curve:
            # Need non-zero coverage and accuracy >= target
            if point.coverage > 0 and point.accuracy >= self.target_accuracy:
                return point.threshold

        return None

    # ------------------------------------------------------------------
    # ΔECE computation
    # ------------------------------------------------------------------

    def compute_delta_ece(
        self,
        y_true_clean: list[str],
        y_pred_clean: list[str],
        confidences_clean: list[float],
        y_true_adv: list[str],
        y_pred_adv: list[str],
        confidences_adv: list[float],
    ) -> float:
        """Compute ΔECE = ECE_adversarial - ECE_clean.

        Parameters
        ----------
        y_true_clean, y_pred_clean, confidences_clean
            Clean test set data.
        y_true_adv, y_pred_adv, confidences_adv
            Adversarial test set data.

        Returns
        -------
        float
            ΔECE value. Positive means calibration worsened under attack.
        """
        correct_clean = np.array([
            1.0 if yt == yp else 0.0
            for yt, yp in zip(y_true_clean, y_pred_clean)
        ])
        conf_clean = np.array(confidences_clean, dtype=np.float64)

        correct_adv = np.array([
            1.0 if yt == yp else 0.0
            for yt, yp in zip(y_true_adv, y_pred_adv)
        ])
        conf_adv = np.array(confidences_adv, dtype=np.float64)

        ece_clean = self.compute_ece(correct_clean, conf_clean)
        ece_adv = self.compute_ece(correct_adv, conf_adv)

        return ece_adv - ece_clean

    def compute_delta_ece_multi_model(
        self,
        clean_results: dict[str, tuple[list[str], list[str], list[float]]],
        adv_results: dict[str, tuple[list[str], list[str], list[float]]],
    ) -> dict[str, float]:
        """Compute ΔECE per model.

        Parameters
        ----------
        clean_results : dict
            ``{model_name: (y_true, y_pred, confidences)}`` for clean data.
        adv_results : dict
            ``{model_name: (y_true, y_pred, confidences)}`` for adversarial data.

        Returns
        -------
        dict
            ``{model_name: ΔECE}``
        """
        delta_ece: dict[str, float] = {}
        for model_name in clean_results:
            if model_name not in adv_results:
                continue
            yt_c, yp_c, conf_c = clean_results[model_name]
            yt_a, yp_a, conf_a = adv_results[model_name]
            delta_ece[model_name] = self.compute_delta_ece(
                yt_c, yp_c, conf_c, yt_a, yp_a, conf_a,
            )
        return delta_ece
