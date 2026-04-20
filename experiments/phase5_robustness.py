"""Phase 5: Robustness Evaluation Experiments

Evaluates grading model robustness via adversarial perturbations and
calibration analysis. Supports:
  - Programmatic perturbations on SciEntsBank test answers (5 types)
  - Adversarial evaluation on Data_Generate test_adversarial split (12 types)
  - Vulnerability matrix (model × perturbation_type → Δ_F1)
  - Calibration analysis (ECE, MCE, reliability diagrams, AUROC)
  - ΔECE per model (clean vs. adversarial)
  - Abstention threshold analysis (accuracy-coverage curves)

All results are saved to results/phase5/ as JSON.

Usage:
    python experiments/phase5_robustness.py
    python experiments/phase5_robustness.py --config configs/robustness.yaml
    python experiments/phase5_robustness.py --experiments perturbations adversarial calibration abstention

Config is read from configs/robustness.yaml.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DataLoader
from src.data.schema import UnifiedRecord
from src.evaluation.metrics import EvaluationHarness
from src.evaluation.reporting import save_results
from src.robustness.adversarial_eval import AdversarialEvaluator
from src.robustness.calibration import CalibrationAnalyzer
from src.robustness.perturbations import PerturbationEngine, VALID_PERTURBATION_TYPES
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase5_robustness")

CONFIG_PATH = PROJECT_ROOT / "configs" / "robustness.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase5"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_unified_records(unified_dir: Path) -> list[UnifiedRecord]:
    """Load all unified JSONL files into UnifiedRecord instances."""
    all_records: list[UnifiedRecord] = []
    for jsonl_file in sorted(unified_dir.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skipping malformed JSON in %s line %d: %s",
                        jsonl_file.name, line_num, e,
                    )
                    continue
                valid_fields = {
                    fld.name for fld in dataclasses.fields(UnifiedRecord)
                }
                filtered = {k: v for k, v in data.items() if k in valid_fields}
                try:
                    rec = UnifiedRecord(**filtered)
                    all_records.append(rec)
                except (TypeError, ValueError) as e:
                    logger.warning("Skipping record: %s", e)
    return all_records


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_trained_grading_models(cfg: dict) -> dict[str, Any]:
    """Load or instantiate trained grading models for robustness evaluation.

    Attempts to load models from saved checkpoints. Falls back to
    creating and training lightweight models on available data.

    Returns a dict of model_name → model instance.
    """
    models: dict[str, Any] = {}

    # Try TF-IDF models (lightweight, can be retrained quickly)
    try:
        from src.grading.baselines.tfidf_ml import TfidfMLClassifier

        for clf_name in ["logistic_regression", "svm_linear", "random_forest"]:
            models[f"tfidf_{clf_name}"] = TfidfMLClassifier(classifier=clf_name)
    except ImportError:
        logger.warning("Could not import TfidfMLClassifier")

    # Try lexical models
    try:
        from src.grading.baselines.lexical import (
            LexicalLogisticRegression,
            LexicalThresholdClassifier,
        )

        models["lexical_threshold"] = LexicalThresholdClassifier()
        models["lexical_logreg"] = LexicalLogisticRegression()
    except ImportError:
        logger.warning("Could not import lexical baselines")

    # Try SBERT models
    try:
        from src.grading.baselines.sbert_sim import (
            SBERTLogisticRegression,
            SBERTThresholdClassifier,
        )

        models["sbert_threshold"] = SBERTThresholdClassifier()
        models["sbert_logreg"] = SBERTLogisticRegression()
    except ImportError:
        logger.warning("Could not import SBERT baselines")

    # Try cross-encoder
    try:
        from src.grading.baselines.cross_encoder import CrossEncoderClassifier

        models["cross_encoder_3way"] = CrossEncoderClassifier(num_labels=3)
    except ImportError:
        logger.warning("Could not import CrossEncoderClassifier")

    # Try ref-aware DeBERTa
    try:
        from src.grading.models.ref_aware import RefAwareClassifier

        models["ref_aware_3way"] = RefAwareClassifier(num_labels=3)
    except ImportError:
        logger.warning("Could not import RefAwareClassifier")

    return models


def train_models_on_data(
    models: dict[str, Any],
    train_records: list[UnifiedRecord],
    label_field: str = "label_3way",
) -> dict[str, Any]:
    """Train all models on the provided training records.

    Returns only the models that trained successfully.
    """
    trained: dict[str, Any] = {}
    train_filtered = [
        r for r in train_records if getattr(r, label_field) is not None
    ]

    if not train_filtered:
        logger.warning("No training records with %s; cannot train models", label_field)
        return trained

    for model_name, model in models.items():
        try:
            logger.info("  Training %s on %d records...", model_name, len(train_filtered))
            model.fit(train_filtered, label_field)
            trained[model_name] = model
            logger.info("  %s trained successfully", model_name)
        except Exception as e:
            logger.warning("  Failed to train %s: %s", model_name, e)

    return trained


# ---------------------------------------------------------------------------
# Clean baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_clean_baseline(
    models: dict[str, Any],
    clean_records: list[UnifiedRecord],
    label_field: str = "label_3way",
) -> dict[str, Any]:
    """Evaluate all models on clean (unperturbed) test data.

    Returns dict with:
      - clean_f1: {model_name: macro_f1}
      - clean_predictions: {model_name: (y_true, y_pred, confidences)}
      - clean_metrics: {model_name: full classification metrics}
    """
    harness = EvaluationHarness()
    clean_f1: dict[str, float] = {}
    clean_predictions: dict[str, tuple[list[str], list[str], list[float]]] = {}
    clean_metrics: dict[str, Any] = {}

    records_with_labels = [
        r for r in clean_records if getattr(r, label_field) is not None
    ]

    if not records_with_labels:
        logger.warning("No clean records with %s for evaluation", label_field)
        return {
            "clean_f1": clean_f1,
            "clean_predictions": clean_predictions,
            "clean_metrics": clean_metrics,
        }

    y_true = [str(getattr(r, label_field)) for r in records_with_labels]

    for model_name, model in models.items():
        try:
            logger.info("  Evaluating %s on %d clean records...",
                        model_name, len(records_with_labels))
            y_pred = [str(p) for p in model.predict(records_with_labels)]

            # Get confidence scores if available
            confidences = _get_confidences(model, records_with_labels)

            metrics = harness.classification_metrics(
                y_true, y_pred, bootstrap_n=100,
            )
            macro_f1 = metrics["macro_f1"]["value"]
            clean_f1[model_name] = macro_f1
            clean_predictions[model_name] = (y_true, y_pred, confidences)
            clean_metrics[model_name] = metrics

            logger.info("  %s clean Macro-F1: %.4f", model_name, macro_f1)
        except Exception as e:
            logger.warning("  Failed to evaluate %s on clean data: %s",
                           model_name, e)

    return {
        "clean_f1": clean_f1,
        "clean_predictions": clean_predictions,
        "clean_metrics": clean_metrics,
    }


def _get_confidences(
    model: Any,
    records: list[UnifiedRecord],
) -> list[float]:
    """Extract confidence scores from a model's predictions.

    Falls back to uniform 1.0 if predict_proba is not available.
    """
    try:
        probas = model.predict_proba(records)
        # Confidence = max probability per prediction
        return [float(max(p)) for p in probas]
    except (AttributeError, NotImplementedError, Exception):
        return [1.0] * len(records)


# ---------------------------------------------------------------------------
# Experiment 1: Programmatic perturbations (sub-task 31.2)
# ---------------------------------------------------------------------------

def run_programmatic_perturbations(
    clean_records: list[UnifiedRecord],
    cfg: dict,
) -> dict[str, list[UnifiedRecord]]:
    """Apply programmatic perturbations to SciEntsBank test answers.

    Returns a dict of perturbation_type → list of perturbed records.
    """
    logger.info("=" * 70)
    logger.info("  PROGRAMMATIC PERTURBATIONS ON SCIENTSBANK TEST ANSWERS")
    logger.info("=" * 70)

    pert_cfg = cfg.get("perturbations", {})
    perturbation_types = pert_cfg.get("types", list(VALID_PERTURBATION_TYPES))
    verbosity_multiplier = pert_cfg.get("verbosity_multiplier", 3)
    random_text_tokens = pert_cfg.get("random_text_tokens", 50)
    seed = cfg.get("seed", 42)

    engine = PerturbationEngine(
        verbosity_multiplier=verbosity_multiplier,
        random_text_tokens=random_text_tokens,
        seed=seed,
    )

    perturbed_by_type: dict[str, list[UnifiedRecord]] = {}

    for pt in perturbation_types:
        logger.info("  Applying perturbation: %s to %d records...",
                     pt, len(clean_records))
        perturbed = []
        for rec in clean_records:
            try:
                perturbed_rec = engine.perturb(rec, pt)
                perturbed.append(perturbed_rec)
            except Exception as e:
                logger.warning(
                    "  Failed to perturb %s with %s: %s",
                    rec.sample_id, pt, e,
                )
        perturbed_by_type[pt] = perturbed
        logger.info("  %s: %d perturbed records", pt, len(perturbed))

    total = sum(len(recs) for recs in perturbed_by_type.values())
    logger.info("  Total perturbed records: %d", total)
    return perturbed_by_type


# ---------------------------------------------------------------------------
# Experiment 2: Adversarial evaluation (sub-task 31.3)
# ---------------------------------------------------------------------------

def run_adversarial_evaluation(
    models: dict[str, Any],
    data_loader: DataLoader,
    perturbed_by_type: dict[str, list[UnifiedRecord]],
    clean_f1: dict[str, float],
    cfg: dict,
    label_field: str = "label_3way",
) -> dict[str, Any]:
    """Evaluate all grading models on adversarial and perturbed test sets.

    Returns vulnerability matrix and per-model per-perturbation F1 results.
    """
    logger.info("=" * 70)
    logger.info("  ADVERSARIAL EVALUATION")
    logger.info("=" * 70)

    adv_cfg = cfg.get("adversarial", {})
    adversarial_split = adv_cfg.get("adversarial_split", "test_adversarial")

    evaluator = AdversarialEvaluator(
        models=models,
        label_field=label_field,
    )

    all_results: dict[str, Any] = {}

    # --- Part A: Data_Generate test_adversarial split ---
    logger.info("-" * 50)
    logger.info("  Part A: Data_Generate %s split", adversarial_split)
    logger.info("-" * 50)

    try:
        adv_records = data_loader.get_split("data_generate", adversarial_split)
        adv_records_with_pt = [
            r for r in adv_records if r.perturbation_type is not None
        ]
        logger.info("  Loaded %d adversarial records (%d with perturbation_type)",
                     len(adv_records), len(adv_records_with_pt))

        if adv_records_with_pt:
            dg_results = evaluator.evaluate(
                adv_records_with_pt, clean_f1=clean_f1,
            )
            all_results["data_generate_adversarial"] = dg_results
            logger.info("  Data_Generate adversarial evaluation complete")

            # Log vulnerability summary
            if "vulnerability_matrix" in dg_results:
                for model_name, pt_deltas in dg_results["vulnerability_matrix"].items():
                    mean_delta = (
                        sum(pt_deltas.values()) / len(pt_deltas)
                        if pt_deltas else 0.0
                    )
                    logger.info("    %s: mean Δ_F1 = %.4f", model_name, mean_delta)
        else:
            logger.warning("  No adversarial records with perturbation_type found")
    except ValueError as e:
        logger.warning("  Could not load %s split: %s", adversarial_split, e)

    # --- Part B: Programmatically perturbed SciEntsBank ---
    logger.info("-" * 50)
    logger.info("  Part B: Programmatically perturbed SciEntsBank")
    logger.info("-" * 50)

    if perturbed_by_type:
        # Combine all perturbed records into a single list
        all_perturbed: list[UnifiedRecord] = []
        for pt_records in perturbed_by_type.values():
            all_perturbed.extend(pt_records)

        if all_perturbed:
            seb_results = evaluator.evaluate(
                all_perturbed, clean_f1=clean_f1,
            )
            all_results["scientsbank_perturbed"] = seb_results
            logger.info("  SciEntsBank perturbed evaluation complete (%d records)",
                         len(all_perturbed))

            if "vulnerability_matrix" in seb_results:
                for model_name, pt_deltas in seb_results["vulnerability_matrix"].items():
                    mean_delta = (
                        sum(pt_deltas.values()) / len(pt_deltas)
                        if pt_deltas else 0.0
                    )
                    logger.info("    %s: mean Δ_F1 = %.4f", model_name, mean_delta)
    else:
        logger.warning("  No perturbed SciEntsBank records available")

    # --- Combined vulnerability matrix ---
    all_results["combined_vulnerability_matrix"] = _build_combined_vulnerability_matrix(
        all_results,
    )

    return all_results


def _build_combined_vulnerability_matrix(
    results: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Merge vulnerability matrices from Data_Generate and SciEntsBank evaluations."""
    combined: dict[str, dict[str, float]] = {}

    for source_key in ["data_generate_adversarial", "scientsbank_perturbed"]:
        source_results = results.get(source_key, {})
        vm = source_results.get("vulnerability_matrix", {})
        for model_name, pt_deltas in vm.items():
            if model_name not in combined:
                combined[model_name] = {}
            for pt, delta in pt_deltas.items():
                combined[model_name][pt] = delta

    return combined


# ---------------------------------------------------------------------------
# Experiment 3: Calibration analysis (sub-task 31.4)
# ---------------------------------------------------------------------------

def run_calibration_analysis(
    models: dict[str, Any],
    clean_predictions: dict[str, tuple[list[str], list[str], list[float]]],
    perturbed_by_type: dict[str, list[UnifiedRecord]],
    cfg: dict,
    label_field: str = "label_3way",
) -> dict[str, Any]:
    """Run calibration analysis on clean and adversarial test sets.

    Computes ECE, MCE, reliability diagrams, AUROC, and ΔECE per model.
    """
    logger.info("=" * 70)
    logger.info("  CALIBRATION ANALYSIS")
    logger.info("=" * 70)

    cal_cfg = cfg.get("calibration", {})
    analyzer = CalibrationAnalyzer(
        n_bins=cal_cfg.get("n_bins", 10),
        threshold_min=cal_cfg.get("threshold_min", 0.0),
        threshold_max=cal_cfg.get("threshold_max", 1.0),
        threshold_step=cal_cfg.get("threshold_step", 0.05),
        target_accuracy=cal_cfg.get("target_accuracy", 0.95),
    )

    all_results: dict[str, Any] = {}

    # --- Clean calibration ---
    logger.info("-" * 50)
    logger.info("  Clean test set calibration")
    logger.info("-" * 50)

    clean_calibration: dict[str, Any] = {}
    for model_name, (y_true, y_pred, confidences) in clean_predictions.items():
        try:
            cal_result = analyzer.analyze(y_true, y_pred, confidences)
            clean_calibration[model_name] = _calibration_result_to_dict(cal_result)
            logger.info("  %s: ECE=%.4f, MCE=%.4f, AUROC=%.4f",
                         model_name, cal_result.ece, cal_result.mce,
                         cal_result.auroc)
        except Exception as e:
            logger.warning("  Calibration failed for %s: %s", model_name, e)

    all_results["clean_calibration"] = clean_calibration

    # --- Adversarial calibration (on combined perturbed records) ---
    logger.info("-" * 50)
    logger.info("  Adversarial test set calibration")
    logger.info("-" * 50)

    adv_calibration: dict[str, Any] = {}
    adv_predictions: dict[str, tuple[list[str], list[str], list[float]]] = {}

    if perturbed_by_type:
        # Combine all perturbed records
        all_perturbed: list[UnifiedRecord] = []
        for pt_records in perturbed_by_type.values():
            all_perturbed.extend(pt_records)

        perturbed_with_labels = [
            r for r in all_perturbed if getattr(r, label_field) is not None
        ]

        if perturbed_with_labels:
            y_true_adv = [
                str(getattr(r, label_field)) for r in perturbed_with_labels
            ]

            for model_name, model in models.items():
                try:
                    y_pred_adv = [
                        str(p) for p in model.predict(perturbed_with_labels)
                    ]
                    confidences_adv = _get_confidences(model, perturbed_with_labels)

                    cal_result = analyzer.analyze(
                        y_true_adv, y_pred_adv, confidences_adv,
                    )
                    adv_calibration[model_name] = _calibration_result_to_dict(
                        cal_result,
                    )
                    adv_predictions[model_name] = (
                        y_true_adv, y_pred_adv, confidences_adv,
                    )
                    logger.info(
                        "  %s (adv): ECE=%.4f, MCE=%.4f, AUROC=%.4f",
                        model_name, cal_result.ece, cal_result.mce,
                        cal_result.auroc,
                    )
                except Exception as e:
                    logger.warning(
                        "  Adversarial calibration failed for %s: %s",
                        model_name, e,
                    )

    all_results["adversarial_calibration"] = adv_calibration

    # --- ΔECE per model ---
    logger.info("-" * 50)
    logger.info("  ΔECE (adversarial - clean)")
    logger.info("-" * 50)

    delta_ece: dict[str, float] = {}
    if clean_predictions and adv_predictions:
        delta_ece = analyzer.compute_delta_ece_multi_model(
            clean_predictions, adv_predictions,
        )
        for model_name, d_ece in delta_ece.items():
            logger.info("  %s: ΔECE = %.4f", model_name, d_ece)

    all_results["delta_ece"] = delta_ece

    return all_results


def _calibration_result_to_dict(cal_result: Any) -> dict[str, Any]:
    """Convert a CalibrationResult dataclass to a JSON-serializable dict."""
    return {
        "ece": cal_result.ece,
        "mce": cal_result.mce,
        "reliability_diagram": {
            "bin_midpoints": cal_result.reliability_diagram.bin_midpoints,
            "bin_accuracies": cal_result.reliability_diagram.bin_accuracies,
            "bin_confidences": cal_result.reliability_diagram.bin_confidences,
            "bin_counts": cal_result.reliability_diagram.bin_counts,
        },
        "auroc": cal_result.auroc,
        "accuracy_coverage_curve": [
            {
                "threshold": pt.threshold,
                "accuracy": pt.accuracy,
                "coverage": pt.coverage,
            }
            for pt in cal_result.accuracy_coverage_curve
        ],
        "optimal_abstention_threshold": cal_result.optimal_abstention_threshold,
    }


# ---------------------------------------------------------------------------
# Experiment 4: Abstention threshold analysis (sub-task 31.5)
# ---------------------------------------------------------------------------

def run_abstention_analysis(
    clean_predictions: dict[str, tuple[list[str], list[str], list[float]]],
    models: dict[str, Any],
    perturbed_by_type: dict[str, list[UnifiedRecord]],
    cfg: dict,
    label_field: str = "label_3way",
) -> dict[str, Any]:
    """Run abstention threshold analysis on clean and adversarial test sets.

    Computes accuracy-coverage curves and optimal thresholds per model.
    """
    logger.info("=" * 70)
    logger.info("  ABSTENTION THRESHOLD ANALYSIS")
    logger.info("=" * 70)

    cal_cfg = cfg.get("calibration", {})
    analyzer = CalibrationAnalyzer(
        n_bins=cal_cfg.get("n_bins", 10),
        threshold_min=cal_cfg.get("threshold_min", 0.0),
        threshold_max=cal_cfg.get("threshold_max", 1.0),
        threshold_step=cal_cfg.get("threshold_step", 0.05),
        target_accuracy=cal_cfg.get("target_accuracy", 0.95),
    )

    all_results: dict[str, Any] = {}

    # --- Clean abstention analysis ---
    logger.info("-" * 50)
    logger.info("  Clean test set abstention analysis")
    logger.info("-" * 50)

    clean_abstention: dict[str, Any] = {}
    for model_name, (y_true, y_pred, confidences) in clean_predictions.items():
        try:
            cal_result = analyzer.analyze(y_true, y_pred, confidences)
            opt_threshold = cal_result.optimal_abstention_threshold

            # Find coverage at optimal threshold
            coverage_at_opt = None
            if opt_threshold is not None:
                for pt in cal_result.accuracy_coverage_curve:
                    if abs(pt.threshold - opt_threshold) < 1e-6:
                        coverage_at_opt = pt.coverage
                        break

            clean_abstention[model_name] = {
                "optimal_threshold": opt_threshold,
                "coverage_at_optimal": coverage_at_opt,
                "target_accuracy": analyzer.target_accuracy,
                "accuracy_coverage_curve": [
                    {
                        "threshold": pt.threshold,
                        "accuracy": pt.accuracy,
                        "coverage": pt.coverage,
                    }
                    for pt in cal_result.accuracy_coverage_curve
                ],
            }
            logger.info(
                "  %s: optimal_threshold=%.2f, coverage=%.4f",
                model_name,
                opt_threshold if opt_threshold is not None else -1.0,
                coverage_at_opt if coverage_at_opt is not None else 0.0,
            )
        except Exception as e:
            logger.warning("  Abstention analysis failed for %s: %s",
                           model_name, e)

    all_results["clean_abstention"] = clean_abstention

    # --- Adversarial abstention analysis ---
    logger.info("-" * 50)
    logger.info("  Adversarial test set abstention analysis")
    logger.info("-" * 50)

    adv_abstention: dict[str, Any] = {}

    if perturbed_by_type:
        all_perturbed: list[UnifiedRecord] = []
        for pt_records in perturbed_by_type.values():
            all_perturbed.extend(pt_records)

        perturbed_with_labels = [
            r for r in all_perturbed if getattr(r, label_field) is not None
        ]

        if perturbed_with_labels:
            y_true_adv = [
                str(getattr(r, label_field)) for r in perturbed_with_labels
            ]

            for model_name, model in models.items():
                try:
                    y_pred_adv = [
                        str(p) for p in model.predict(perturbed_with_labels)
                    ]
                    confidences_adv = _get_confidences(
                        model, perturbed_with_labels,
                    )

                    cal_result = analyzer.analyze(
                        y_true_adv, y_pred_adv, confidences_adv,
                    )
                    opt_threshold = cal_result.optimal_abstention_threshold

                    coverage_at_opt = None
                    if opt_threshold is not None:
                        for pt in cal_result.accuracy_coverage_curve:
                            if abs(pt.threshold - opt_threshold) < 1e-6:
                                coverage_at_opt = pt.coverage
                                break

                    adv_abstention[model_name] = {
                        "optimal_threshold": opt_threshold,
                        "coverage_at_optimal": coverage_at_opt,
                        "target_accuracy": analyzer.target_accuracy,
                        "accuracy_coverage_curve": [
                            {
                                "threshold": pt.threshold,
                                "accuracy": pt.accuracy,
                                "coverage": pt.coverage,
                            }
                            for pt in cal_result.accuracy_coverage_curve
                        ],
                    }
                    logger.info(
                        "  %s (adv): optimal_threshold=%.2f, coverage=%.4f",
                        model_name,
                        opt_threshold if opt_threshold is not None else -1.0,
                        coverage_at_opt if coverage_at_opt is not None else 0.0,
                    )
                except Exception as e:
                    logger.warning(
                        "  Adversarial abstention failed for %s: %s",
                        model_name, e,
                    )

    all_results["adversarial_abstention"] = adv_abstention

    # --- Cross-model comparison ---
    all_results["model_comparison"] = _compare_abstention_across_models(
        clean_abstention, adv_abstention,
    )

    return all_results


def _compare_abstention_across_models(
    clean: dict[str, Any],
    adversarial: dict[str, Any],
) -> dict[str, Any]:
    """Compare abstention performance across model families."""
    comparison: dict[str, Any] = {}

    for model_name in clean:
        clean_data = clean[model_name]
        adv_data = adversarial.get(model_name, {})

        entry: dict[str, Any] = {
            "clean_optimal_threshold": clean_data.get("optimal_threshold"),
            "clean_coverage": clean_data.get("coverage_at_optimal"),
        }

        if adv_data:
            entry["adversarial_optimal_threshold"] = adv_data.get(
                "optimal_threshold",
            )
            entry["adversarial_coverage"] = adv_data.get("coverage_at_optimal")

            # Compute coverage drop
            clean_cov = clean_data.get("coverage_at_optimal")
            adv_cov = adv_data.get("coverage_at_optimal")
            if clean_cov is not None and adv_cov is not None:
                entry["coverage_drop"] = clean_cov - adv_cov

        comparison[model_name] = entry

    return comparison


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(all_results: dict[str, Any]) -> dict[str, Any]:
    """Build a summary of all Phase 5 experiment results."""
    summary: dict[str, Any] = {}

    # Vulnerability matrix summary
    adv_results = all_results.get("adversarial_evaluation", {})
    combined_vm = adv_results.get("combined_vulnerability_matrix", {})
    if combined_vm:
        vm_summary: dict[str, Any] = {}
        for model_name, pt_deltas in combined_vm.items():
            if pt_deltas:
                vm_summary[model_name] = {
                    "mean_delta_f1": sum(pt_deltas.values()) / len(pt_deltas),
                    "max_delta_f1": max(pt_deltas.values()),
                    "most_vulnerable_perturbation": max(
                        pt_deltas, key=pt_deltas.get,
                    ),
                    "n_perturbation_types": len(pt_deltas),
                }
        summary["vulnerability_summary"] = vm_summary

    # Calibration summary
    cal_results = all_results.get("calibration_analysis", {})
    delta_ece = cal_results.get("delta_ece", {})
    if delta_ece:
        summary["calibration_summary"] = {
            "delta_ece_per_model": delta_ece,
            "mean_delta_ece": (
                sum(delta_ece.values()) / len(delta_ece)
                if delta_ece else 0.0
            ),
            "worst_calibration_model": (
                max(delta_ece, key=delta_ece.get) if delta_ece else None
            ),
        }

    # Abstention summary
    abstention_results = all_results.get("abstention_analysis", {})
    model_comparison = abstention_results.get("model_comparison", {})
    if model_comparison:
        summary["abstention_summary"] = model_comparison

    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all Phase 5 robustness evaluation experiments."""
    parser = argparse.ArgumentParser(
        description="Phase 5: Robustness Evaluation Experiments",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to robustness config YAML (default: configs/robustness.yaml)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["perturbations", "adversarial", "calibration", "abstention"],
        default=["perturbations", "adversarial", "calibration", "abstention"],
        help="Which experiments to run (default: all)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info("Loaded config from %s (seed=%d)", config_path, seed)

    # Load unified data
    unified_dir = PROJECT_ROOT / "data" / "unified"
    logger.info("Loading unified records from %s ...", unified_dir)
    all_records = load_unified_records(unified_dir)
    logger.info("Loaded %d total records", len(all_records))

    if not all_records:
        logger.error("No records loaded. Run phase1_data_audit.py first.")
        sys.exit(1)

    # Create DataLoader
    data_loader = DataLoader(all_records)

    # Load and train grading models
    logger.info("Loading grading models...")
    models = load_trained_grading_models(cfg)
    logger.info("Loaded %d model types", len(models))

    # Train models on available training data
    label_field = "label_3way"
    train_records: list[UnifiedRecord] = []
    for source, split in [("scientsbank", "train"), ("data_generate", "train")]:
        try:
            recs = data_loader.get_split(source, split)
            train_records.extend(recs)
        except ValueError:
            logger.warning("%s/%s not available for training", source, split)

    if train_records:
        logger.info("Training models on %d records...", len(train_records))
        models = train_models_on_data(models, train_records, label_field)
    else:
        logger.warning("No training data available; models will be untrained")

    if not models:
        logger.error("No models available for evaluation.")
        sys.exit(1)

    logger.info("Trained %d models successfully", len(models))

    # Load SciEntsBank test records for perturbation
    seb_test_records: list[UnifiedRecord] = []
    for split_name in ["test_ua", "test_uq", "test_ud"]:
        try:
            recs = data_loader.get_split("scientsbank", split_name)
            seb_test_records.extend(recs)
        except ValueError:
            pass

    # Fallback: use any scientsbank test-like records
    if not seb_test_records:
        seb_test_records = [
            r for r in all_records
            if r.source_dataset == "scientsbank"
            and r.split.startswith("test")
        ]

    logger.info("SciEntsBank test records for perturbation: %d",
                len(seb_test_records))

    # Evaluate on clean data first (needed for Δ_F1 and ΔECE)
    logger.info("Evaluating models on clean test data...")
    clean_eval = evaluate_clean_baseline(
        models, seb_test_records, label_field,
    )
    clean_f1 = clean_eval["clean_f1"]
    clean_predictions = clean_eval["clean_predictions"]
    clean_metrics = clean_eval["clean_metrics"]

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all results
    all_experiment_results: dict[str, Any] = {
        "clean_metrics": clean_metrics,
    }

    # --- Programmatic perturbations (31.2) ---
    perturbed_by_type: dict[str, list[UnifiedRecord]] = {}
    if "perturbations" in args.experiments:
        perturbed_by_type = run_programmatic_perturbations(
            seb_test_records, cfg,
        )
        perturbation_summary = {
            pt: len(recs) for pt, recs in perturbed_by_type.items()
        }
        all_experiment_results["perturbation_counts"] = perturbation_summary
        save_results(
            {"perturbation_counts": perturbation_summary},
            "perturbation_counts",
            results_dir=str(RESULTS_DIR),
        )
        logger.info("Saved perturbation counts")

    # --- Adversarial evaluation (31.3) ---
    if "adversarial" in args.experiments:
        adversarial_results = run_adversarial_evaluation(
            models, data_loader, perturbed_by_type, clean_f1, cfg, label_field,
        )
        all_experiment_results["adversarial_evaluation"] = adversarial_results
        save_results(
            {"adversarial_evaluation": adversarial_results},
            "adversarial_evaluation_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info("Saved adversarial evaluation results")

    # --- Calibration analysis (31.4) ---
    if "calibration" in args.experiments:
        calibration_results = run_calibration_analysis(
            models, clean_predictions, perturbed_by_type, cfg, label_field,
        )
        all_experiment_results["calibration_analysis"] = calibration_results
        save_results(
            {"calibration_analysis": calibration_results},
            "calibration_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info("Saved calibration results")

    # --- Abstention threshold analysis (31.5) ---
    if "abstention" in args.experiments:
        abstention_results = run_abstention_analysis(
            clean_predictions, models, perturbed_by_type, cfg, label_field,
        )
        all_experiment_results["abstention_analysis"] = abstention_results
        save_results(
            {"abstention_analysis": abstention_results},
            "abstention_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info("Saved abstention results")

    # --- Build summary and save combined results (31.6) ---
    summary = build_summary(all_experiment_results)
    all_experiment_results["summary"] = summary

    save_results(
        all_experiment_results,
        "phase5_all_results",
        results_dir=str(RESULTS_DIR),
    )

    logger.info("=" * 70)
    logger.info("  Phase 5 robustness evaluation experiments complete.")
    logger.info("  Results saved to: %s", RESULTS_DIR)
    if "vulnerability_summary" in summary:
        logger.info("  Vulnerability summary:")
        for model_name, vs in summary["vulnerability_summary"].items():
            logger.info(
                "    %s: mean_Δ_F1=%.4f, most_vulnerable=%s",
                model_name,
                vs.get("mean_delta_f1", 0),
                vs.get("most_vulnerable_perturbation", "?"),
            )
    if "calibration_summary" in summary:
        logger.info("  Mean ΔECE: %.4f",
                     summary["calibration_summary"].get("mean_delta_ece", 0))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
