"""Phase 4: Feedback Generation Experiments

Runs all feedback strategies on Data_Generate test split and computes
automatic evaluation metrics. Supports:
  - All feedback strategies: template, retrieval, T5 generative (grounded + ungrounded), hybrid
  - Grounded vs. ungrounded ablation (T5 with and without missing_concepts)
  - Gold-label vs. predicted-label ablation
  - Human evaluation template generation (100 stratified samples)

All results are saved to results/phase4/ as JSON.

Usage:
    python experiments/phase4_feedback.py
    python experiments/phase4_feedback.py --config configs/feedback.yaml
    python experiments/phase4_feedback.py --experiments strategies ablation_grounding ablation_labels human_eval

Config is read from configs/feedback.yaml.
"""

from __future__ import annotations

import argparse
import dataclasses
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
from src.evaluation.metrics import (
    EvaluationHarness,
    generate_human_eval_template,
    export_human_eval_template_json,
)
from src.evaluation.reporting import save_results
from src.feedback.concept_gap import ConceptGapDetector, ConceptGapResult
from src.feedback.generative import (
    FactualConsistencyChecker,
    T5GenerativeFeedbackGenerator,
)
from src.feedback.hybrid import HybridFeedbackPipeline
from src.feedback.retrieval import RetrievalFeedbackGenerator
from src.feedback.template import TemplateFeedbackGenerator
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase4_feedback")

CONFIG_PATH = PROJECT_ROOT / "configs" / "feedback.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase4"


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
# Component creation helpers
# ---------------------------------------------------------------------------

def create_concept_gap_detector(cfg: dict) -> ConceptGapDetector:
    """Create a ConceptGapDetector from config."""
    gap_cfg = cfg.get("concept_gap", {})
    return ConceptGapDetector(
        model_name=gap_cfg.get("nli_model", "cross-encoder/nli-deberta-v3-base"),
    )


def create_template_generator(cfg: dict) -> TemplateFeedbackGenerator:
    """Create a TemplateFeedbackGenerator."""
    return TemplateFeedbackGenerator()


def create_retrieval_generator(
    cfg: dict,
    training_records: list[UnifiedRecord],
) -> RetrievalFeedbackGenerator:
    """Create a RetrievalFeedbackGenerator and index training records."""
    ret_cfg = cfg.get("retrieval", {})
    gen = RetrievalFeedbackGenerator(
        model_name=ret_cfg.get("sbert_model", "all-MiniLM-L6-v2"),
        similarity_threshold=ret_cfg.get("similarity_threshold", 0.5),
    )
    gen.index_training_records(training_records)
    return gen


def create_generative_generator(
    cfg: dict,
    grounded: bool = True,
    consistency_checker: FactualConsistencyChecker | None = None,
) -> T5GenerativeFeedbackGenerator:
    """Create a T5GenerativeFeedbackGenerator from config."""
    gen_cfg = cfg.get("generative", {})
    return T5GenerativeFeedbackGenerator(
        model_name=gen_cfg.get("model_name", "t5-base"),
        grounded=grounded,
        max_input_length=gen_cfg.get("max_input_length", 512),
        max_output_length=gen_cfg.get("max_output_length", 256),
        learning_rate=gen_cfg.get("learning_rate", 3e-4),
        epochs=gen_cfg.get("epochs", 5),
        batch_size=gen_cfg.get("batch_size", 8),
        consistency_checker=consistency_checker,
    )


def create_consistency_checker(cfg: dict) -> FactualConsistencyChecker:
    """Create a FactualConsistencyChecker from config."""
    gap_cfg = cfg.get("concept_gap", {})
    return FactualConsistencyChecker(
        model_name=gap_cfg.get("nli_model", "cross-encoder/nli-deberta-v3-base"),
    )


def create_hybrid_pipeline(
    cfg: dict,
    grading_model: object,
    concept_gap_detector: ConceptGapDetector,
    generative_generator: T5GenerativeFeedbackGenerator,
    consistency_checker: FactualConsistencyChecker,
) -> HybridFeedbackPipeline:
    """Create a HybridFeedbackPipeline from config."""
    hybrid_cfg = cfg.get("hybrid", {})
    return HybridFeedbackPipeline(
        grading_model=grading_model,
        concept_gap_detector=concept_gap_detector,
        generative_generator=generative_generator,
        consistency_checker=consistency_checker,
        consistency_threshold=hybrid_cfg.get("consistency_threshold", 0.7),
    )


# ---------------------------------------------------------------------------
# Feedback generation + evaluation helpers
# ---------------------------------------------------------------------------

def detect_concept_gaps(
    records: list[UnifiedRecord],
    detector: ConceptGapDetector,
) -> list[ConceptGapResult]:
    """Run concept gap detection on all records."""
    results: list[ConceptGapResult] = []
    for rec in records:
        key_concepts = list(rec.key_concepts) if rec.key_concepts else None
        gap = detector.detect(
            question=rec.question,
            reference_answer=rec.reference_answer,
            student_answer=rec.student_answer,
            key_concepts=key_concepts,
        )
        results.append(gap)
    return results


def generate_feedback_batch(
    generator: object,
    records: list[UnifiedRecord],
    gap_results: list[ConceptGapResult],
    labels: list[str],
) -> list[tuple[str, str]]:
    """Generate feedback for a batch of records.

    Returns list of (feedback_short, feedback_detailed) tuples.
    """
    outputs: list[tuple[str, str]] = []
    for rec, gap, label in zip(records, gap_results, labels):
        try:
            short, detailed = generator.generate(rec, gap, label)
            outputs.append((short, detailed))
        except Exception as e:
            logger.warning(
                "Feedback generation failed for %s: %s", rec.sample_id, e,
            )
            outputs.append(("", ""))
    return outputs


def evaluate_feedback(
    harness: EvaluationHarness,
    generated_feedback: list[str],
    gold_feedback: list[str],
    gold_missing_concepts: list[list[str]],
    reference_answers: list[str],
    nli_pipeline: object | None = None,
) -> dict[str, Any]:
    """Compute all feedback evaluation metrics."""
    return harness.feedback_metrics(
        generated_feedback=generated_feedback,
        gold_feedback=gold_feedback,
        gold_missing_concepts=gold_missing_concepts,
        reference_answers=reference_answers,
        nli_pipeline=nli_pipeline,
    )


def get_gold_labels(
    records: list[UnifiedRecord],
    label_field: str = "label_3way",
) -> list[str]:
    """Extract gold labels from records."""
    return [str(getattr(rec, label_field) or "unknown") for rec in records]


# ---------------------------------------------------------------------------
# Stub grading model for gold-label mode
# ---------------------------------------------------------------------------

class GoldLabelGradingModel:
    """A stub grading model that returns gold labels from records.

    Used in the hybrid pipeline when evaluating with gold labels
    instead of model-predicted labels.
    """

    def __init__(self, label_field: str = "label_3way") -> None:
        self._label_field = label_field

    def predict(self, records: list[UnifiedRecord]) -> list[str]:
        """Return gold labels for the given records."""
        return [
            str(getattr(r, self._label_field) or "unknown")
            for r in records
        ]

    def predict_proba(self, records: list[UnifiedRecord]) -> list[list[float]]:
        """Return dummy probabilities (not used in feedback pipeline)."""
        return [[1.0] for _ in records]

    def fit(self, records: list[UnifiedRecord], label_field: str) -> None:
        """No-op: gold labels don't require training."""
        pass


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_all_strategies(
    test_records: list[UnifiedRecord],
    training_records: list[UnifiedRecord],
    cfg: dict,
    concept_gap_detector: ConceptGapDetector,
    consistency_checker: FactualConsistencyChecker,
    nli_pipeline: object | None = None,
) -> list[dict[str, Any]]:
    """Run all feedback strategies on the test split (sub-task 27.2).

    Strategies: template, retrieval, T5 generative (grounded), hybrid.
    For each strategy, compute: ROUGE-L, BERTScore, concept coverage,
    factual consistency, hallucination rate.
    """
    logger.info("=" * 70)
    logger.info("  ALL FEEDBACK STRATEGIES ON DATA_GENERATE TEST SPLIT")
    logger.info("=" * 70)

    harness = EvaluationHarness()
    all_results: list[dict[str, Any]] = []

    # Prepare gold data
    gold_feedback = [rec.feedback_detailed or "" for rec in test_records]
    gold_missing = [list(rec.missing_concepts) for rec in test_records]
    reference_answers = [rec.reference_answer for rec in test_records]
    gold_labels = get_gold_labels(test_records)

    # Detect concept gaps for all test records
    logger.info("Detecting concept gaps for %d test records...", len(test_records))
    gap_results = detect_concept_gaps(test_records, concept_gap_detector)

    # --- Strategy 1: Template ---
    logger.info("-" * 50)
    logger.info("  Strategy: Template")
    logger.info("-" * 50)
    template_gen = create_template_generator(cfg)
    template_outputs = generate_feedback_batch(
        template_gen, test_records, gap_results, gold_labels,
    )
    template_detailed = [d for _, d in template_outputs]
    template_metrics = evaluate_feedback(
        harness, template_detailed, gold_feedback,
        gold_missing, reference_answers, nli_pipeline,
    )
    all_results.append({
        "strategy": "template",
        "experiment": "all_strategies",
        "n_records": len(test_records),
        "metrics": template_metrics,
    })
    logger.info("  Template ROUGE-L F1: %.4f", template_metrics.get("rouge_l", {}).get("f1", 0))

    # --- Strategy 2: Retrieval ---
    logger.info("-" * 50)
    logger.info("  Strategy: Retrieval")
    logger.info("-" * 50)
    retrieval_gen = create_retrieval_generator(cfg, training_records)
    retrieval_outputs = generate_feedback_batch(
        retrieval_gen, test_records, gap_results, gold_labels,
    )
    retrieval_detailed = [d for _, d in retrieval_outputs]
    retrieval_metrics = evaluate_feedback(
        harness, retrieval_detailed, gold_feedback,
        gold_missing, reference_answers, nli_pipeline,
    )
    all_results.append({
        "strategy": "retrieval",
        "experiment": "all_strategies",
        "n_records": len(test_records),
        "metrics": retrieval_metrics,
    })
    logger.info("  Retrieval ROUGE-L F1: %.4f", retrieval_metrics.get("rouge_l", {}).get("f1", 0))

    # --- Strategy 3: T5 Generative (grounded) ---
    logger.info("-" * 50)
    logger.info("  Strategy: T5 Generative (grounded)")
    logger.info("-" * 50)
    t5_grounded = create_generative_generator(
        cfg, grounded=True, consistency_checker=consistency_checker,
    )
    # Fine-tune on training records
    logger.info("  Fine-tuning T5 (grounded) on %d training records...", len(training_records))
    ft_summary = t5_grounded.fine_tune(training_records)
    logger.info("  Fine-tuning complete: %d records, %d epochs",
                ft_summary["num_records"], len(ft_summary["epoch_losses"]))

    t5_grounded_outputs = generate_feedback_batch(
        t5_grounded, test_records, gap_results, gold_labels,
    )
    t5_grounded_detailed = [d for _, d in t5_grounded_outputs]
    t5_grounded_metrics = evaluate_feedback(
        harness, t5_grounded_detailed, gold_feedback,
        gold_missing, reference_answers, nli_pipeline,
    )
    all_results.append({
        "strategy": "t5_generative_grounded",
        "experiment": "all_strategies",
        "n_records": len(test_records),
        "fine_tune_summary": ft_summary,
        "metrics": t5_grounded_metrics,
    })
    logger.info("  T5 Grounded ROUGE-L F1: %.4f",
                t5_grounded_metrics.get("rouge_l", {}).get("f1", 0))

    # --- Strategy 4: Hybrid ---
    logger.info("-" * 50)
    logger.info("  Strategy: Hybrid")
    logger.info("-" * 50)
    gold_grading_model = GoldLabelGradingModel()
    hybrid_pipeline = create_hybrid_pipeline(
        cfg, gold_grading_model, concept_gap_detector,
        t5_grounded, consistency_checker,
    )
    hybrid_outputs: list[tuple[str, str]] = []
    hybrid_metadata: list[dict[str, Any]] = []
    for rec in test_records:
        try:
            result = hybrid_pipeline.run(rec)
            hybrid_outputs.append((result.feedback_short, result.feedback_detailed))
            hybrid_metadata.append({
                "consistency_score": result.consistency_score,
                "used_fallback": result.used_fallback,
                "contradicting_claims": result.contradicting_claims,
            })
        except Exception as e:
            logger.warning("Hybrid pipeline failed for %s: %s", rec.sample_id, e)
            hybrid_outputs.append(("", ""))
            hybrid_metadata.append({
                "consistency_score": 0.0,
                "used_fallback": True,
                "contradicting_claims": [],
            })

    hybrid_detailed = [d for _, d in hybrid_outputs]
    hybrid_metrics = evaluate_feedback(
        harness, hybrid_detailed, gold_feedback,
        gold_missing, reference_answers, nli_pipeline,
    )
    fallback_rate = sum(1 for m in hybrid_metadata if m["used_fallback"]) / max(len(hybrid_metadata), 1)
    all_results.append({
        "strategy": "hybrid",
        "experiment": "all_strategies",
        "n_records": len(test_records),
        "metrics": hybrid_metrics,
        "fallback_rate": fallback_rate,
    })
    logger.info("  Hybrid ROUGE-L F1: %.4f (fallback rate: %.2f%%)",
                hybrid_metrics.get("rouge_l", {}).get("f1", 0), fallback_rate * 100)

    return all_results


def run_grounded_vs_ungrounded_ablation(
    test_records: list[UnifiedRecord],
    training_records: list[UnifiedRecord],
    cfg: dict,
    concept_gap_detector: ConceptGapDetector,
    consistency_checker: FactualConsistencyChecker,
    nli_pipeline: object | None = None,
) -> list[dict[str, Any]]:
    """Run grounded vs. ungrounded ablation (sub-task 27.3).

    Compares T5 generative feedback with and without missing_concepts
    in the input prompt.
    """
    logger.info("=" * 70)
    logger.info("  GROUNDED VS. UNGROUNDED ABLATION")
    logger.info("=" * 70)

    harness = EvaluationHarness()
    all_results: list[dict[str, Any]] = []

    # Prepare gold data
    gold_feedback = [rec.feedback_detailed or "" for rec in test_records]
    gold_missing = [list(rec.missing_concepts) for rec in test_records]
    reference_answers = [rec.reference_answer for rec in test_records]
    gold_labels = get_gold_labels(test_records)

    # Detect concept gaps
    logger.info("Detecting concept gaps for %d test records...", len(test_records))
    gap_results = detect_concept_gaps(test_records, concept_gap_detector)

    for grounded in [True, False]:
        mode_name = "grounded" if grounded else "ungrounded"
        logger.info("-" * 50)
        logger.info("  Mode: %s", mode_name)
        logger.info("-" * 50)

        t5_gen = create_generative_generator(
            cfg, grounded=grounded, consistency_checker=consistency_checker,
        )

        # Fine-tune
        logger.info("  Fine-tuning T5 (%s) on %d training records...",
                    mode_name, len(training_records))
        ft_summary = t5_gen.fine_tune(training_records)
        logger.info("  Fine-tuning complete: %d records", ft_summary["num_records"])

        # Generate feedback
        outputs = generate_feedback_batch(
            t5_gen, test_records, gap_results, gold_labels,
        )
        detailed = [d for _, d in outputs]

        # Evaluate
        metrics = evaluate_feedback(
            harness, detailed, gold_feedback,
            gold_missing, reference_answers, nli_pipeline,
        )

        all_results.append({
            "mode": mode_name,
            "grounded": grounded,
            "experiment": "grounded_vs_ungrounded",
            "n_records": len(test_records),
            "fine_tune_summary": ft_summary,
            "metrics": metrics,
        })
        logger.info("  %s ROUGE-L F1: %.4f", mode_name,
                    metrics.get("rouge_l", {}).get("f1", 0))

    # Compute deltas between grounded and ungrounded
    if len(all_results) == 2:
        grounded_metrics = all_results[0]["metrics"]
        ungrounded_metrics = all_results[1]["metrics"]
        deltas = _compute_metric_deltas(grounded_metrics, ungrounded_metrics)
        all_results.append({
            "experiment": "grounded_vs_ungrounded_deltas",
            "description": "grounded - ungrounded",
            "deltas": deltas,
        })

    return all_results


def run_gold_vs_predicted_label_ablation(
    test_records: list[UnifiedRecord],
    training_records: list[UnifiedRecord],
    cfg: dict,
    concept_gap_detector: ConceptGapDetector,
    consistency_checker: FactualConsistencyChecker,
    nli_pipeline: object | None = None,
) -> list[dict[str, Any]]:
    """Run gold-label vs. predicted-label ablation (sub-task 27.4).

    Compares feedback quality when using gold labels from the dataset
    versus labels predicted by a grading model.
    """
    logger.info("=" * 70)
    logger.info("  GOLD-LABEL VS. PREDICTED-LABEL ABLATION")
    logger.info("=" * 70)

    harness = EvaluationHarness()
    all_results: list[dict[str, Any]] = []

    # Prepare gold data
    gold_feedback = [rec.feedback_detailed or "" for rec in test_records]
    gold_missing = [list(rec.missing_concepts) for rec in test_records]
    reference_answers = [rec.reference_answer for rec in test_records]

    # Detect concept gaps
    logger.info("Detecting concept gaps for %d test records...", len(test_records))
    gap_results = detect_concept_gaps(test_records, concept_gap_detector)

    # Fine-tune a single T5 model (grounded) for both conditions
    t5_gen = create_generative_generator(
        cfg, grounded=True, consistency_checker=consistency_checker,
    )
    logger.info("Fine-tuning T5 (grounded) on %d training records...", len(training_records))
    ft_summary = t5_gen.fine_tune(training_records)

    # --- Condition 1: Gold labels ---
    logger.info("-" * 50)
    logger.info("  Condition: Gold labels")
    logger.info("-" * 50)
    gold_labels = get_gold_labels(test_records)
    gold_outputs = generate_feedback_batch(
        t5_gen, test_records, gap_results, gold_labels,
    )
    gold_detailed = [d for _, d in gold_outputs]
    gold_metrics = evaluate_feedback(
        harness, gold_detailed, gold_feedback,
        gold_missing, reference_answers, nli_pipeline,
    )
    all_results.append({
        "condition": "gold_labels",
        "experiment": "gold_vs_predicted_labels",
        "n_records": len(test_records),
        "metrics": gold_metrics,
    })
    logger.info("  Gold-label ROUGE-L F1: %.4f",
                gold_metrics.get("rouge_l", {}).get("f1", 0))

    # --- Condition 2: Predicted labels ---
    # Train a simple grading model on training data for label prediction
    logger.info("-" * 50)
    logger.info("  Condition: Predicted labels")
    logger.info("-" * 50)
    predicted_labels = _get_predicted_labels(
        test_records, training_records, cfg,
    )
    pred_outputs = generate_feedback_batch(
        t5_gen, test_records, gap_results, predicted_labels,
    )
    pred_detailed = [d for _, d in pred_outputs]
    pred_metrics = evaluate_feedback(
        harness, pred_detailed, gold_feedback,
        gold_missing, reference_answers, nli_pipeline,
    )
    all_results.append({
        "condition": "predicted_labels",
        "experiment": "gold_vs_predicted_labels",
        "n_records": len(test_records),
        "metrics": pred_metrics,
    })
    logger.info("  Predicted-label ROUGE-L F1: %.4f",
                pred_metrics.get("rouge_l", {}).get("f1", 0))

    # Compute deltas
    if len(all_results) == 2:
        deltas = _compute_metric_deltas(
            all_results[0]["metrics"], all_results[1]["metrics"],
        )
        all_results.append({
            "experiment": "gold_vs_predicted_labels_deltas",
            "description": "gold - predicted",
            "deltas": deltas,
        })

    return all_results


def _get_predicted_labels(
    test_records: list[UnifiedRecord],
    training_records: list[UnifiedRecord],
    cfg: dict,
) -> list[str]:
    """Get predicted labels for test records using a TF-IDF classifier.

    Falls back to gold labels if the classifier cannot be trained.
    """
    try:
        from src.grading.baselines.tfidf_ml import TfidfMLClassifier

        clf = TfidfMLClassifier(classifier="logistic_regression")
        label_field = "label_3way"

        # Filter training records with valid labels
        train_filtered = [
            r for r in training_records
            if getattr(r, label_field) is not None
        ]
        if not train_filtered:
            logger.warning("No training records with %s; using gold labels", label_field)
            return get_gold_labels(test_records)

        clf.fit(train_filtered, label_field)
        predictions = clf.predict(test_records)
        return [str(p) for p in predictions]

    except Exception as e:
        logger.warning(
            "Could not train grading model for predicted labels: %s. "
            "Falling back to gold labels.", e,
        )
        return get_gold_labels(test_records)


def run_human_eval_template(
    test_records: list[UnifiedRecord],
    cfg: dict,
    concept_gap_detector: ConceptGapDetector,
    all_strategy_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate human evaluation template for 100 stratified samples (sub-task 27.5).

    Selects 100 records stratified by predicted_label and generates
    a template with the 5-point rubric for human evaluation.
    """
    logger.info("=" * 70)
    logger.info("  HUMAN EVALUATION TEMPLATE GENERATION")
    logger.info("=" * 70)

    eval_cfg = cfg.get("evaluation", {})
    n_samples = eval_cfg.get("human_eval_samples", 100)
    seed = cfg.get("seed", 42)

    # Prepare records for template generation
    # Use template-based feedback as the default generated feedback
    template_gen = create_template_generator(cfg)
    gap_results = detect_concept_gaps(test_records, concept_gap_detector)
    gold_labels = get_gold_labels(test_records)

    template_records: list[dict[str, Any]] = []
    for rec, gap, label in zip(test_records, gap_results, gold_labels):
        try:
            short, detailed = template_gen.generate(rec, gap, label)
        except Exception:
            short, detailed = "", ""

        template_records.append({
            "sample_id": rec.sample_id,
            "question": rec.question,
            "reference_answer": rec.reference_answer,
            "student_answer": rec.student_answer,
            "generated_feedback": detailed,
            "predicted_label": label,
            "gold_feedback": rec.feedback_detailed or "",
            "missing_concepts": list(rec.missing_concepts),
        })

    # Generate the human evaluation template
    template = generate_human_eval_template(
        records=template_records,
        n_samples=min(n_samples, len(template_records)),
        stratify_by="predicted_label",
        seed=seed,
    )

    result = {
        "experiment": "human_eval_template",
        "n_samples": len(template),
        "n_total_records": len(test_records),
        "stratify_by": "predicted_label",
        "template": template,
    }

    logger.info("  Generated human evaluation template with %d samples", len(template))
    return result


# ---------------------------------------------------------------------------
# Metric delta computation
# ---------------------------------------------------------------------------

def _compute_metric_deltas(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> dict[str, Any]:
    """Compute deltas between two metric dicts (a - b).

    Handles nested metric structures (e.g., rouge_l.f1, bertscore.f1).
    """
    deltas: dict[str, Any] = {}

    for key in metrics_a:
        if key not in metrics_b:
            continue
        val_a = metrics_a[key]
        val_b = metrics_b[key]

        if isinstance(val_a, dict) and isinstance(val_b, dict):
            # Nested metrics (e.g., rouge_l: {precision, recall, f1})
            sub_deltas = {}
            for sub_key in val_a:
                if sub_key in val_b:
                    a_val = val_a[sub_key]
                    b_val = val_b[sub_key]
                    if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                        sub_deltas[sub_key] = a_val - b_val
            if sub_deltas:
                deltas[key] = sub_deltas
        elif isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            deltas[key] = val_a - val_b

    return deltas


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(all_results: dict[str, Any]) -> dict[str, Any]:
    """Build a summary of all Phase 4 experiment results."""
    summary: dict[str, Any] = {}

    # Strategy comparison
    strategies = all_results.get("all_strategies", [])
    if strategies:
        strategy_comparison: list[dict[str, Any]] = []
        for result in strategies:
            if "metrics" not in result:
                continue
            metrics = result["metrics"]
            entry: dict[str, Any] = {
                "strategy": result.get("strategy", "unknown"),
            }
            if "rouge_l" in metrics:
                entry["rouge_l_f1"] = metrics["rouge_l"].get("f1", 0)
            if "bertscore" in metrics:
                entry["bertscore_f1"] = metrics["bertscore"].get("f1", 0)
            if "concept_coverage" in metrics:
                entry["concept_coverage"] = metrics["concept_coverage"].get("mean", 0)
            if "factual_consistency" in metrics:
                entry["factual_consistency"] = metrics["factual_consistency"].get("mean", 0)
            if "hallucination_rate" in metrics:
                entry["hallucination_rate"] = metrics["hallucination_rate"].get("rate", 0)
            strategy_comparison.append(entry)
        summary["strategy_comparison"] = strategy_comparison

    # Grounding ablation
    grounding = all_results.get("grounded_vs_ungrounded", [])
    if grounding:
        grounding_summary = {}
        for result in grounding:
            if "mode" in result and "metrics" in result:
                mode = result["mode"]
                metrics = result["metrics"]
                grounding_summary[mode] = {
                    "rouge_l_f1": metrics.get("rouge_l", {}).get("f1", 0),
                    "concept_coverage": metrics.get("concept_coverage", {}).get("mean", 0),
                }
            elif "deltas" in result:
                grounding_summary["deltas"] = result["deltas"]
        summary["grounding_ablation"] = grounding_summary

    # Label ablation
    labels = all_results.get("gold_vs_predicted_labels", [])
    if labels:
        label_summary = {}
        for result in labels:
            if "condition" in result and "metrics" in result:
                condition = result["condition"]
                metrics = result["metrics"]
                label_summary[condition] = {
                    "rouge_l_f1": metrics.get("rouge_l", {}).get("f1", 0),
                    "concept_coverage": metrics.get("concept_coverage", {}).get("mean", 0),
                }
            elif "deltas" in result:
                label_summary["deltas"] = result["deltas"]
        summary["label_ablation"] = label_summary

    # Human eval
    human_eval = all_results.get("human_eval_template", {})
    if human_eval:
        summary["human_eval_samples"] = human_eval.get("n_samples", 0)

    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all Phase 4 feedback generation experiments."""
    parser = argparse.ArgumentParser(
        description="Phase 4: Feedback Generation Experiments",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to feedback config YAML (default: configs/feedback.yaml)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["strategies", "ablation_grounding", "ablation_labels", "human_eval"],
        default=["strategies", "ablation_grounding", "ablation_labels", "human_eval"],
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

    # Load Data_Generate training and test records
    try:
        training_records = data_loader.get_split("data_generate", "train")
    except ValueError:
        logger.warning("Data_Generate train split not available; using all usable records")
        training_records = [
            r for r in all_records
            if r.usable_for_feedback and r.feedback_detailed
        ]

    # Try multiple test split names
    test_records: list[UnifiedRecord] = []
    test_split_names = ["test_seen", "test_unseen_questions", "test_unseen_answers"]
    for split_name in test_split_names:
        try:
            recs = data_loader.get_split("data_generate", split_name)
            test_records.extend(recs)
        except ValueError:
            pass

    if not test_records:
        # Fallback: use any data_generate records not in training
        train_ids = {r.sample_id for r in training_records}
        test_records = [
            r for r in all_records
            if r.source_dataset == "data_generate"
            and r.sample_id not in train_ids
            and r.usable_for_feedback
        ]

    if not test_records:
        logger.error("No test records available for feedback experiments.")
        sys.exit(1)

    # Filter to records usable for feedback
    test_records = [r for r in test_records if r.usable_for_feedback]
    training_records = [r for r in training_records if r.usable_for_feedback]

    logger.info("Training records: %d, Test records: %d",
                len(training_records), len(test_records))

    # Create shared components
    logger.info("Creating shared components...")
    concept_gap_detector = create_concept_gap_detector(cfg)
    consistency_checker = create_consistency_checker(cfg)

    # NLI pipeline for factual consistency / hallucination metrics
    # Reuse the consistency checker's pipeline if available
    nli_pipeline = None
    try:
        consistency_checker._ensure_pipeline()
        nli_pipeline = consistency_checker._pipeline
    except Exception:
        logger.warning("Could not load NLI pipeline for metric computation")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all results
    all_experiment_results: dict[str, Any] = {}

    # --- Run all strategies (27.2) ---
    if "strategies" in args.experiments:
        strategy_results = run_all_strategies(
            test_records, training_records, cfg,
            concept_gap_detector, consistency_checker, nli_pipeline,
        )
        all_experiment_results["all_strategies"] = strategy_results
        save_results(
            {"experiments": strategy_results},
            "all_strategies_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info("Saved all-strategies results (%d experiments)", len(strategy_results))

    # --- Grounded vs. ungrounded ablation (27.3) ---
    if "ablation_grounding" in args.experiments:
        grounding_results = run_grounded_vs_ungrounded_ablation(
            test_records, training_records, cfg,
            concept_gap_detector, consistency_checker, nli_pipeline,
        )
        all_experiment_results["grounded_vs_ungrounded"] = grounding_results
        save_results(
            {"experiments": grounding_results},
            "grounded_vs_ungrounded_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info("Saved grounding ablation results (%d experiments)",
                    len(grounding_results))

    # --- Gold-label vs. predicted-label ablation (27.4) ---
    if "ablation_labels" in args.experiments:
        label_results = run_gold_vs_predicted_label_ablation(
            test_records, training_records, cfg,
            concept_gap_detector, consistency_checker, nli_pipeline,
        )
        all_experiment_results["gold_vs_predicted_labels"] = label_results
        save_results(
            {"experiments": label_results},
            "gold_vs_predicted_labels_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info("Saved label ablation results (%d experiments)",
                    len(label_results))

    # --- Human evaluation template (27.5) ---
    if "human_eval" in args.experiments:
        human_eval_result = run_human_eval_template(
            test_records, cfg, concept_gap_detector,
        )
        all_experiment_results["human_eval_template"] = human_eval_result

        # Save template as JSON
        save_results(
            human_eval_result,
            "human_eval_template",
            results_dir=str(RESULTS_DIR),
        )

        # Also save as standalone JSON for easy distribution
        template_json = export_human_eval_template_json(
            human_eval_result.get("template", []),
        )
        template_path = RESULTS_DIR / "human_eval_template_standalone.json"
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(template_json)
        logger.info("Saved human evaluation template to %s", template_path)

    # --- Build summary and save combined results (27.6) ---
    summary = build_summary(all_experiment_results)
    all_experiment_results["summary"] = summary

    save_results(
        all_experiment_results,
        "phase4_all_results",
        results_dir=str(RESULTS_DIR),
    )

    logger.info("=" * 70)
    logger.info("  Phase 4 feedback generation experiments complete.")
    logger.info("  Results saved to: %s", RESULTS_DIR)
    if "strategy_comparison" in summary:
        logger.info("  Strategy comparison:")
        for entry in summary["strategy_comparison"]:
            logger.info(
                "    %s: ROUGE-L=%.4f, BERTScore=%.4f",
                entry.get("strategy", "?"),
                entry.get("rouge_l_f1", 0),
                entry.get("bertscore_f1", 0),
            )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
