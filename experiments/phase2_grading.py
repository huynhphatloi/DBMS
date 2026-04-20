"""Phase 2: Grading Experiments

Runs all grading baselines + Reference-Answer-Aware DeBERTa model on
configured train/test splits. Supports:
  - In-domain experiments: SciEntsBank train → UA/UQ/UD (2-way, 3-way, 5-way)
  - MohlerASAG regression experiments (Pearson r, RMSE, QWK)
  - Cross-domain transfer experiments (Requirement 12)
  - Synthetic data augmentation ablation (Requirement 13)

All results are saved to results/phase2/ as JSON.

Usage:
    python experiments/phase2_grading.py
    python experiments/phase2_grading.py --config configs/grading.yaml
    python experiments/phase2_grading.py --skip-llm  # skip LLM zero-shot (requires API key)

Config is read from configs/grading.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2_grading")

CONFIG_PATH = PROJECT_ROOT / "configs" / "grading.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"


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
    import dataclasses

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
                # Filter to only fields that UnifiedRecord accepts
                valid_fields = {fld.name for fld in dataclasses.fields(UnifiedRecord)}
                filtered = {k: v for k, v in data.items() if k in valid_fields}
                try:
                    rec = UnifiedRecord(**filtered)
                    all_records.append(rec)
                except (TypeError, ValueError) as e:
                    logger.warning("Skipping record: %s", e)
    return all_records


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------

def create_lexical_models(cfg: dict) -> list[tuple[str, Any]]:
    """Create lexical baseline models from config."""
    from src.grading.baselines.lexical import (
        LexicalLogisticRegression,
        LexicalThresholdClassifier,
    )

    lexical_cfg = cfg.get("lexical", {})
    metric = lexical_cfg.get("metric", "rouge_l")
    threshold = lexical_cfg.get("threshold", 0.5)

    models = [
        (
            "lexical_threshold",
            LexicalThresholdClassifier(metric=metric, threshold=threshold),
        ),
        ("lexical_logreg", LexicalLogisticRegression()),
    ]
    return models


def create_tfidf_models(cfg: dict) -> list[tuple[str, Any]]:
    """Create TF-IDF + ML baseline models from config."""
    from src.grading.baselines.tfidf_ml import TfidfMLClassifier

    tfidf_cfg = cfg.get("tfidf_ml", {})
    max_features = tfidf_cfg.get("max_features", 5000)
    classifiers = tfidf_cfg.get(
        "classifiers",
        ["logistic_regression", "svm_linear", "random_forest", "gradient_boosting"],
    )

    models = []
    for clf_name in classifiers:
        models.append(
            (
                f"tfidf_{clf_name}",
                TfidfMLClassifier(classifier=clf_name, max_features=max_features),
            )
        )
    return models


def create_tfidf_regressors(cfg: dict) -> list[tuple[str, Any]]:
    """Create TF-IDF regressors for regression tasks."""
    from src.grading.baselines.tfidf_ml import TfidfMLRegressor  # noqa: PLC0415

    tfidf_cfg = cfg.get("tfidf_ml", {})
    max_features = tfidf_cfg.get("max_features", 5000)

    return [
        ("tfidf_ridge_reg", TfidfMLRegressor(
            regressor="ridge", max_features=max_features)),
        ("tfidf_svr_reg", TfidfMLRegressor(
            regressor="svr", max_features=max_features)),
    ]


def create_sbert_models(cfg: dict) -> list[tuple[str, Any]]:
    """Create SBERT similarity baseline models from config."""
    from src.grading.baselines.sbert_sim import (
        SBERTLogisticRegression,
        SBERTThresholdClassifier,
    )

    sbert_cfg = cfg.get("sbert", {})
    model_name = sbert_cfg.get("model_name", "all-MiniLM-L6-v2")
    threshold = sbert_cfg.get("threshold", 0.5)

    return [
        (
            "sbert_threshold",
            SBERTThresholdClassifier(model_name=model_name, threshold=threshold),
        ),
        ("sbert_logreg", SBERTLogisticRegression(model_name=model_name)),
    ]


def create_cross_encoder_classifier(cfg: dict, num_labels: int) -> tuple[str, Any]:
    """Create a cross-encoder classifier from config."""
    from src.grading.baselines.cross_encoder import CrossEncoderClassifier

    ce_cfg = cfg.get("cross_encoder", {})
    return (
        f"cross_encoder_{num_labels}way",
        CrossEncoderClassifier(
            model_name=ce_cfg.get("model_name", "cross-encoder/stsb-roberta-base"),
            num_labels=num_labels,
            num_epochs=ce_cfg.get("epochs", 5),
            batch_size=ce_cfg.get("batch_size", 16),
            learning_rate=ce_cfg.get("learning_rate", 2e-5),
        ),
    )


def create_cross_encoder_regressor(cfg: dict) -> tuple[str, Any]:
    """Create a cross-encoder regressor from config."""
    from src.grading.baselines.cross_encoder import CrossEncoderRegressor

    ce_cfg = cfg.get("cross_encoder", {})
    return (
        "cross_encoder_regression",
        CrossEncoderRegressor(
            model_name=ce_cfg.get("model_name", "cross-encoder/stsb-roberta-base"),
            num_epochs=ce_cfg.get("epochs", 5),
            batch_size=ce_cfg.get("batch_size", 16),
            learning_rate=ce_cfg.get("learning_rate", 2e-5),
        ),
    )


def create_ref_aware_classifier(cfg: dict, num_labels: int) -> tuple[str, Any]:
    """Create a reference-answer-aware DeBERTa classifier from config."""
    from src.grading.models.ref_aware import RefAwareClassifier

    ra_cfg = cfg.get("ref_aware", {})
    return (
        f"ref_aware_{num_labels}way",
        RefAwareClassifier(
            model_name=ra_cfg.get("encoder", "microsoft/deberta-v3-base"),
            num_labels=num_labels,
            num_epochs=ra_cfg.get("epochs", 10),
            batch_size=ra_cfg.get("batch_size", 16),
            learning_rate=ra_cfg.get("learning_rate", 2e-5),
            max_length=ra_cfg.get("max_length", 512),
        ),
    )


def create_ref_aware_multitask(cfg: dict, num_labels: int) -> tuple[str, Any]:
    """Create a reference-answer-aware DeBERTa multi-task model from config."""
    from src.grading.models.ref_aware import RefAwareMultiTask

    ra_cfg = cfg.get("ref_aware", {})
    mt_cfg = ra_cfg.get("multi_task", {})
    alpha = mt_cfg.get("alpha", 0.7)
    return (
        f"ref_aware_multitask_{num_labels}way",
        RefAwareMultiTask(
            model_name=ra_cfg.get("encoder", "microsoft/deberta-v3-base"),
            num_labels=num_labels,
            alpha=alpha,
            num_epochs=ra_cfg.get("epochs", 10),
            batch_size=ra_cfg.get("batch_size", 16),
            learning_rate=ra_cfg.get("learning_rate", 2e-5),
            max_length=ra_cfg.get("max_length", 512),
            task="classification",
        ),
    )


def create_llm_zeroshot(cfg: dict, task: str) -> tuple[str, Any] | None:
    """Create an LLM zero-shot grader from config. Returns None if no API key."""
    import os

    from src.grading.baselines.llm_zeroshot import LLMZeroShotGrader

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; skipping LLM zero-shot baseline")
        return None

    return (
        f"llm_zeroshot_{task}",
        LLMZeroShotGrader(api_key=api_key, task=task),
    )


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def get_label_field_for_task(task: str) -> str:
    """Map task formulation to the label field on UnifiedRecord."""
    mapping = {
        "2way": "label_2way",
        "3way": "label_3way",
        "5way": "label_5way",
        "regression": "score_normalized",
    }
    if task not in mapping:
        raise ValueError(f"Unknown task: {task!r}. Must be one of {list(mapping)}")
    return mapping[task]


def get_num_labels(task: str) -> int:
    """Map task formulation to number of classification labels."""
    mapping = {"2way": 2, "3way": 3, "5way": 5}
    return mapping.get(task, 3)


def run_classification_experiment(
    model_name: str,
    model: Any,
    train_records: list[UnifiedRecord],
    test_splits: dict[str, list[UnifiedRecord]],
    label_field: str,
    bootstrap_n: int = 1000,
) -> dict[str, Any]:
    """Train a classification model and evaluate on all test splits.

    Returns a dict mapping test split names to their metric results.
    """
    harness = EvaluationHarness()
    results: dict[str, Any] = {"model": model_name, "task": "classification"}

    # Train
    logger.info("  Training %s on %d records (label_field=%s)...",
                model_name, len(train_records), label_field)
    model.fit(train_records, label_field)

    # Evaluate on each test split
    split_results = {}
    for split_name, test_records in test_splits.items():
        if not test_records:
            logger.warning("  Skipping empty test split: %s", split_name)
            continue

        logger.info("  Evaluating %s on %s (%d records)...",
                    model_name, split_name, len(test_records))
        y_true = [str(getattr(r, label_field)) for r in test_records]
        y_pred = model.predict(test_records)
        # Ensure predictions are strings for classification
        y_pred = [str(p) for p in y_pred]

        metrics = harness.classification_metrics(y_true, y_pred, bootstrap_n=bootstrap_n)
        split_results[split_name] = metrics

    results["splits"] = split_results
    return results


def run_regression_experiment(
    model_name: str,
    model: Any,
    train_records: list[UnifiedRecord],
    test_splits: dict[str, list[UnifiedRecord]],
    label_field: str,
    bootstrap_n: int = 1000,
) -> dict[str, Any]:
    """Train a regression model and evaluate on all test splits.

    Returns a dict mapping test split names to their metric results.
    """
    harness = EvaluationHarness()
    results: dict[str, Any] = {"model": model_name, "task": "regression"}

    # Train
    logger.info("  Training %s on %d records (label_field=%s)...",
                model_name, len(train_records), label_field)
    model.fit(train_records, label_field)

    # Evaluate on each test split
    split_results = {}
    for split_name, test_records in test_splits.items():
        if not test_records:
            logger.warning("  Skipping empty test split: %s", split_name)
            continue

        logger.info("  Evaluating %s on %s (%d records)...",
                    model_name, split_name, len(test_records))
        y_true = [float(getattr(r, label_field)) for r in test_records]
        y_pred = [float(p) for p in model.predict(test_records)]

        metrics = harness.regression_metrics(y_true, y_pred, bootstrap_n=bootstrap_n)
        split_results[split_name] = metrics

    results["splits"] = split_results
    return results


# ---------------------------------------------------------------------------
# In-domain experiments (16.2)
# ---------------------------------------------------------------------------

def run_in_domain_experiments(
    data_loader: DataLoader,
    cfg: dict,
    bootstrap_n: int,
    skip_llm: bool = False,
) -> list[dict[str, Any]]:
    """Run in-domain experiments: SciEntsBank train → UA/UQ/UD.

    Tests 2-way, 3-way, and 5-way classification tasks.
    """
    logger.info("=" * 70)
    logger.info("  IN-DOMAIN EXPERIMENTS: SciEntsBank train → UA/UQ/UD")
    logger.info("=" * 70)

    all_results: list[dict[str, Any]] = []

    # Load SciEntsBank train data
    try:
        seb_train = data_loader.get_split("scientsbank", "train")
    except ValueError:
        logger.warning("SciEntsBank train split not available; skipping in-domain experiments")
        return all_results

    # Load test splits
    test_split_names = ["test_ua", "test_uq", "test_ud"]
    test_splits: dict[str, list[UnifiedRecord]] = {}
    for split_name in test_split_names:
        try:
            test_splits[split_name] = data_loader.get_split("scientsbank", split_name)
        except ValueError:
            logger.warning("SciEntsBank %s not available; skipping", split_name)

    if not test_splits:
        logger.warning("No SciEntsBank test splits available; skipping in-domain experiments")
        return all_results

    # Run for each task formulation
    for task in ["2way", "3way", "5way"]:
        logger.info("-" * 50)
        logger.info("  Task: %s", task)
        logger.info("-" * 50)

        label_field = get_label_field_for_task(task)
        num_labels = get_num_labels(task)

        # Filter records that have the required label
        train_filtered = [r for r in seb_train if getattr(r, label_field) is not None]
        test_filtered = {
            name: [r for r in recs if getattr(r, label_field) is not None]
            for name, recs in test_splits.items()
        }

        if not train_filtered:
            logger.warning("No training records with %s; skipping task %s", label_field, task)
            continue

        # Create models for this task
        models: list[tuple[str, Any]] = []
        models.extend(create_lexical_models(cfg))
        models.extend(create_tfidf_models(cfg))
        models.extend(create_sbert_models(cfg))
        models.append(create_cross_encoder_classifier(cfg, num_labels))
        models.append(create_ref_aware_classifier(cfg, num_labels))

        # Multi-task model if enabled
        mt_cfg = cfg.get("ref_aware", {}).get("multi_task", {})
        if mt_cfg.get("enabled", False):
            models.append(create_ref_aware_multitask(cfg, num_labels))

        # LLM zero-shot (no training needed)
        if not skip_llm:
            llm_model = create_llm_zeroshot(cfg, task)
            if llm_model is not None:
                models.append(llm_model)

        # Run each model
        for model_name, model in models:
            try:
                result = run_classification_experiment(
                    model_name=f"{model_name}_{task}",
                    model=model,
                    train_records=train_filtered,
                    test_splits=test_filtered,
                    label_field=label_field,
                    bootstrap_n=bootstrap_n,
                )
                result["experiment"] = "in_domain"
                result["task_formulation"] = task
                result["train_source"] = "scientsbank_train"
                all_results.append(result)
            except Exception as e:
                logger.error("Error running %s (%s): %s", model_name, task, e)

    return all_results


# ---------------------------------------------------------------------------
# MohlerASAG regression experiments (16.3)
# ---------------------------------------------------------------------------

def run_mohler_regression_experiments(
    data_loader: DataLoader,
    cfg: dict,
    bootstrap_n: int,
) -> list[dict[str, Any]]:
    """Run MohlerASAG regression experiments (Pearson r, RMSE, QWK)."""
    logger.info("=" * 70)
    logger.info("  MOHLER REGRESSION EXPERIMENTS")
    logger.info("=" * 70)

    all_results: list[dict[str, Any]] = []

    # Load MohlerASAG train/test
    try:
        mohler_train = data_loader.get_split("mohler", "train")
    except ValueError:
        logger.warning("MohlerASAG train split not available; skipping regression experiments")
        return all_results

    test_splits: dict[str, list[UnifiedRecord]] = {}
    for split_name in ["test", "validation"]:
        try:
            test_splits[split_name] = data_loader.get_split("mohler", split_name)
        except ValueError:
            pass

    if not test_splits:
        logger.warning("No MohlerASAG test splits available; skipping regression experiments")
        return all_results

    label_field = "score_normalized"

    # Filter records with valid scores
    train_filtered = [r for r in mohler_train if r.score_normalized is not None]
    test_filtered = {
        name: [r for r in recs if r.score_normalized is not None]
        for name, recs in test_splits.items()
    }

    if not train_filtered:
        logger.warning("No MohlerASAG training records with score_normalized; skipping")
        return all_results

    # Regression models
    models: list[tuple[str, Any]] = []
    models.extend(create_tfidf_regressors(cfg))
    models.append(create_cross_encoder_regressor(cfg))

    # RefAware multi-task in regression mode
    from src.grading.models.ref_aware import RefAwareMultiTask

    ra_cfg = cfg.get("ref_aware", {})
    mt_cfg = ra_cfg.get("multi_task", {})
    models.append((
        "ref_aware_multitask_regression",
        RefAwareMultiTask(
            model_name=ra_cfg.get("encoder", "microsoft/deberta-v3-base"),
            num_labels=3,
            alpha=mt_cfg.get("alpha", 0.7),
            num_epochs=ra_cfg.get("epochs", 10),
            batch_size=ra_cfg.get("batch_size", 16),
            learning_rate=ra_cfg.get("learning_rate", 2e-5),
            max_length=ra_cfg.get("max_length", 512),
            task="regression",
        ),
    ))

    for model_name, model in models:
        try:
            result = run_regression_experiment(
                model_name=model_name,
                model=model,
                train_records=train_filtered,
                test_splits=test_filtered,
                label_field=label_field,
                bootstrap_n=bootstrap_n,
            )
            result["experiment"] = "mohler_regression"
            result["train_source"] = "mohler_train"
            all_results.append(result)
        except Exception as e:
            logger.error("Error running %s (regression): %s", model_name, e)

    return all_results


# ---------------------------------------------------------------------------
# Cross-domain transfer experiments (16.4 / Requirement 12)
# ---------------------------------------------------------------------------

def run_cross_domain_experiments(
    data_loader: DataLoader,
    cfg: dict,
    bootstrap_n: int,
    skip_llm: bool = False,
) -> list[dict[str, Any]]:
    """Run cross-domain transfer experiments.

    Transfer configurations:
      - SciEntsBank train → MohlerASAG test
      - MohlerASAG train → SciEntsBank UQ
      - Data_Generate train → SciEntsBank UA/UQ/UD
    """
    logger.info("=" * 70)
    logger.info("  CROSS-DOMAIN TRANSFER EXPERIMENTS (Requirement 12)")
    logger.info("=" * 70)

    all_results: list[dict[str, Any]] = []
    label_field = "label_3way"  # Use 3-way for cross-domain comparability

    # Define transfer configurations
    transfer_configs = [
        {
            "name": "scientsbank_to_mohler",
            "train": [("scientsbank", "train")],
            "test": [("mohler", "test")],
        },
        {
            "name": "mohler_to_scientsbank_uq",
            "train": [("mohler", "train")],
            "test": [("scientsbank", "test_uq")],
        },
        {
            "name": "data_generate_to_scientsbank",
            "train": [("data_generate", "train")],
            "test": [
                ("scientsbank", "test_ua"),
                ("scientsbank", "test_uq"),
                ("scientsbank", "test_ud"),
            ],
        },
    ]

    for transfer_cfg in transfer_configs:
        logger.info("-" * 50)
        logger.info("  Transfer: %s", transfer_cfg["name"])
        logger.info("-" * 50)

        # Load training data
        train_records: list[UnifiedRecord] = []
        for source, split in transfer_cfg["train"]:
            try:
                recs = data_loader.get_split(source, split)
                train_records.extend(recs)
            except ValueError:
                logger.warning("  %s/%s not available; skipping", source, split)

        # Load test data
        test_splits: dict[str, list[UnifiedRecord]] = {}
        for source, split in transfer_cfg["test"]:
            try:
                recs = data_loader.get_split(source, split)
                test_splits[f"{source}_{split}"] = recs
            except ValueError:
                logger.warning("  %s/%s not available; skipping", source, split)

        if not train_records or not test_splits:
            logger.warning("  Insufficient data for transfer %s; skipping",
                           transfer_cfg["name"])
            continue

        # Filter for valid labels
        train_filtered = [r for r in train_records if getattr(r, label_field) is not None]
        test_filtered = {
            name: [r for r in recs if getattr(r, label_field) is not None]
            for name, recs in test_splits.items()
        }

        if not train_filtered:
            logger.warning("  No training records with %s; skipping", label_field)
            continue

        # Models for cross-domain (use 3-way)
        num_labels = 3
        models: list[tuple[str, Any]] = []
        models.extend(create_lexical_models(cfg))
        models.extend(create_tfidf_models(cfg))
        models.extend(create_sbert_models(cfg))
        models.append(create_cross_encoder_classifier(cfg, num_labels))
        models.append(create_ref_aware_classifier(cfg, num_labels))

        for model_name, model in models:
            try:
                result = run_classification_experiment(
                    model_name=model_name,
                    model=model,
                    train_records=train_filtered,
                    test_splits=test_filtered,
                    label_field=label_field,
                    bootstrap_n=bootstrap_n,
                )
                result["experiment"] = "cross_domain"
                result["transfer"] = transfer_cfg["name"]
                result["task_formulation"] = "3way"
                result["train_source"] = str(transfer_cfg["train"])
                all_results.append(result)
            except Exception as e:
                logger.error("Error running %s (cross-domain %s): %s",
                             model_name, transfer_cfg["name"], e)

    return all_results


# ---------------------------------------------------------------------------
# Synthetic data augmentation ablation (16.5 / Requirement 13)
# ---------------------------------------------------------------------------

def run_augmentation_ablation(
    data_loader: DataLoader,
    cfg: dict,
    bootstrap_n: int,
) -> list[dict[str, Any]]:
    """Run synthetic data augmentation ablation.

    Compares:
      - SciEntsBank train only → SciEntsBank UA/UQ/UD
      - SciEntsBank train + Data_Generate train → SciEntsBank UA/UQ/UD

    Also supports confidence-based filtering of Data_Generate records.
    """
    logger.info("=" * 70)
    logger.info("  SYNTHETIC DATA AUGMENTATION ABLATION (Requirement 13)")
    logger.info("=" * 70)

    all_results: list[dict[str, Any]] = []
    label_field = "label_3way"
    num_labels = 3

    # Load SciEntsBank train
    try:
        seb_train = data_loader.get_split("scientsbank", "train")
    except ValueError:
        logger.warning("SciEntsBank train not available; skipping augmentation ablation")
        return all_results

    # Load Data_Generate train
    try:
        gen_train = data_loader.get_split("data_generate", "train")
    except ValueError:
        logger.warning("Data_Generate train not available; skipping augmentation ablation")
        gen_train = []

    # Load SciEntsBank test splits
    test_splits: dict[str, list[UnifiedRecord]] = {}
    for split_name in ["test_ua", "test_uq", "test_ud"]:
        try:
            test_splits[split_name] = data_loader.get_split("scientsbank", split_name)
        except ValueError:
            pass

    if not test_splits:
        logger.warning("No SciEntsBank test splits available; skipping augmentation ablation")
        return all_results

    # Filter for valid labels
    seb_train_filtered = [r for r in seb_train if getattr(r, label_field) is not None]
    gen_train_filtered = [r for r in gen_train if getattr(r, label_field) is not None]
    test_filtered = {
        name: [r for r in recs if getattr(r, label_field) is not None]
        for name, recs in test_splits.items()
    }

    if not seb_train_filtered:
        logger.warning("No SciEntsBank training records with %s; skipping", label_field)
        return all_results

    # Confidence thresholds for ablation
    confidence_thresholds = [None, 0.85, 0.90, 0.95]

    # Training configurations
    train_configs = [
        {
            "name": "scientsbank_only",
            "records": seb_train_filtered,
            "description": "SciEntsBank train only",
        },
    ]

    # Add augmented configs with different confidence thresholds
    for threshold in confidence_thresholds:
        if threshold is None:
            aug_records = gen_train_filtered
            config_name = "scientsbank_plus_data_generate"
            desc = "SciEntsBank + Data_Generate (all)"
        else:
            aug_records = [
                r for r in gen_train_filtered
                if r.annotation_confidence is not None
                and r.annotation_confidence >= threshold
            ]
            config_name = f"scientsbank_plus_data_generate_conf{threshold}"
            desc = f"SciEntsBank + Data_Generate (confidence >= {threshold})"

        if aug_records:
            train_configs.append({
                "name": config_name,
                "records": seb_train_filtered + aug_records,
                "description": desc,
            })

    # Models to test
    def _create_models():
        models = []
        models.extend(create_lexical_models(cfg))
        models.extend(create_tfidf_models(cfg))
        models.extend(create_sbert_models(cfg))
        models.append(create_cross_encoder_classifier(cfg, num_labels))
        models.append(create_ref_aware_classifier(cfg, num_labels))
        return models

    # Run experiments for each training configuration
    for train_cfg in train_configs:
        logger.info("-" * 50)
        logger.info("  Training config: %s (%d records)",
                    train_cfg["description"], len(train_cfg["records"]))
        logger.info("-" * 50)

        models = _create_models()

        for model_name, model in models:
            try:
                result = run_classification_experiment(
                    model_name=model_name,
                    model=model,
                    train_records=train_cfg["records"],
                    test_splits=test_filtered,
                    label_field=label_field,
                    bootstrap_n=bootstrap_n,
                )
                result["experiment"] = "augmentation_ablation"
                result["train_config"] = train_cfg["name"]
                result["train_description"] = train_cfg["description"]
                result["train_size"] = len(train_cfg["records"])
                result["task_formulation"] = "3way"
                all_results.append(result)
            except Exception as e:
                logger.error("Error running %s (augmentation %s): %s",
                             model_name, train_cfg["name"], e)

    # Compute augmentation deltas
    augmentation_deltas = compute_augmentation_deltas(all_results)
    if augmentation_deltas:
        all_results.append({
            "experiment": "augmentation_ablation_deltas",
            "deltas": augmentation_deltas,
        })

    return all_results


def compute_augmentation_deltas(results: list[dict]) -> dict[str, Any]:
    """Compute Macro_F1 differences between augmented and non-augmented models."""
    deltas: dict[str, Any] = {}

    # Group results by model name
    baseline_results: dict[str, dict] = {}  # model → split → metrics
    augmented_results: dict[str, dict[str, dict]] = {}  # config → model → split → metrics

    for result in results:
        if result.get("experiment") != "augmentation_ablation":
            continue
        model_name = result.get("model", "")
        train_config = result.get("train_config", "")
        splits = result.get("splits", {})

        if train_config == "scientsbank_only":
            baseline_results[model_name] = splits
        else:
            if train_config not in augmented_results:
                augmented_results[train_config] = {}
            augmented_results[train_config][model_name] = splits

    # Compute deltas
    for config_name, config_results in augmented_results.items():
        config_deltas: dict[str, dict] = {}
        for model_name, aug_splits in config_results.items():
            if model_name not in baseline_results:
                continue
            base_splits = baseline_results[model_name]
            model_deltas: dict[str, float] = {}
            for split_name in aug_splits:
                if split_name not in base_splits:
                    continue
                aug_f1 = aug_splits[split_name].get("macro_f1", {}).get("value", 0)
                base_f1 = base_splits[split_name].get("macro_f1", {}).get("value", 0)
                model_deltas[split_name] = aug_f1 - base_f1
            if model_deltas:
                config_deltas[model_name] = model_deltas
        if config_deltas:
            deltas[config_name] = config_deltas

    return deltas


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all Phase 2 grading experiments."""
    parser = argparse.ArgumentParser(description="Phase 2: Grading Experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to grading config YAML (default: configs/grading.yaml)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM zero-shot baseline (requires API key)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["in_domain", "regression", "cross_domain", "augmentation"],
        default=["in_domain", "regression", "cross_domain", "augmentation"],
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
    bootstrap_n = cfg.get("bootstrap_iterations", 1000)
    logger.info("Loaded config from %s (seed=%d, bootstrap_n=%d)",
                config_path, seed, bootstrap_n)

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

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all results
    all_experiment_results: dict[str, list[dict]] = {}

    # --- In-domain experiments (16.2) ---
    if "in_domain" in args.experiments:
        in_domain_results = run_in_domain_experiments(
            data_loader, cfg, bootstrap_n, skip_llm=args.skip_llm
        )
        all_experiment_results["in_domain"] = in_domain_results
        save_results(
            {"experiments": in_domain_results},
            "in_domain_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info("Saved in-domain results (%d experiments)", len(in_domain_results))

    # --- MohlerASAG regression experiments (16.3) ---
    if "regression" in args.experiments:
        regression_results = run_mohler_regression_experiments(
            data_loader, cfg, bootstrap_n
        )
        all_experiment_results["regression"] = regression_results
        save_results(
            {"experiments": regression_results},
            "mohler_regression_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info("Saved regression results (%d experiments)", len(regression_results))

    # --- Cross-domain transfer experiments (16.4) ---
    if "cross_domain" in args.experiments:
        cross_domain_results = run_cross_domain_experiments(
            data_loader, cfg, bootstrap_n, skip_llm=args.skip_llm
        )
        all_experiment_results["cross_domain"] = cross_domain_results
        save_results(
            {"experiments": cross_domain_results},
            "cross_domain_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info("Saved cross-domain results (%d experiments)",
                    len(cross_domain_results))

    # --- Synthetic data augmentation ablation (16.5) ---
    if "augmentation" in args.experiments:
        augmentation_results = run_augmentation_ablation(
            data_loader, cfg, bootstrap_n
        )
        all_experiment_results["augmentation"] = augmentation_results
        save_results(
            {"experiments": augmentation_results},
            "augmentation_ablation_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info("Saved augmentation ablation results (%d experiments)",
                    len(augmentation_results))

    # --- Save combined results (16.6) ---
    save_results(
        all_experiment_results,
        "phase2_all_results",
        results_dir=str(RESULTS_DIR),
    )
    logger.info("=" * 70)
    logger.info("  Phase 2 grading experiments complete.")
    logger.info("  Results saved to: %s", RESULTS_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
