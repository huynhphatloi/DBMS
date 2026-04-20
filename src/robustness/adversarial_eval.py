"""Adversarial Evaluator for robustness analysis of grading models.

Evaluates trained grading models on adversarial test sets and produces
a vulnerability matrix showing performance degradation per model per
perturbation type.

Supports:
- Data_Generate ``test_adversarial`` split (12 perturbation types)
- Programmatically perturbed SciEntsBank test answers (5 perturbation types)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol, runtime_checkable, Iterable

from sklearn.metrics import f1_score

from src.data.schema import UnifiedRecord


# ── Perturbation taxonomy ─────────────────────────────────────────────

PERTURBATION_TAXONOMY: dict[str, list[str]] = {
    "surface_level": [
        "synonym_swap",
        "paraphrase_low_overlap",
        "word_order_change",
        "grammar_noise",
    ],
    "semantic": [
        "near-contradiction",
        "one_correct_plus_fatal_error",
        "concept-jumble",
        "vague_but_plausible",
    ],
    "gaming_deception": [
        "high_overlap_wrong_meaning",
        "misleading_fluent_explanation",
        "hedge_language",
        "distractor_sentence_added",
    ],
}

# Reverse lookup: perturbation_type → category
_PERTURBATION_TO_CATEGORY: dict[str, str] = {}
for _cat, _types in PERTURBATION_TAXONOMY.items():
    for _pt in _types:
        _PERTURBATION_TO_CATEGORY[_pt] = _cat


def categorize_perturbation(perturbation_type: str) -> str:
    """Return the category for a perturbation type.

    Returns ``"unknown"`` if the perturbation type is not in the taxonomy.
    """
    return _PERTURBATION_TO_CATEGORY.get(perturbation_type, "unknown")


# ── Grading model protocol ────────────────────────────────────────────

@runtime_checkable
class GradingModelProtocol(Protocol):
    """Minimal interface expected from grading models."""

    def predict(self, records: Iterable[UnifiedRecord]) -> list[str | float]: ...


# ── Helper: compute macro-F1 ─────────────────────────────────────────

def _compute_macro_f1(
    y_true: list[str],
    y_pred: list[str],
) -> float:
    """Compute macro-averaged F1 score.

    Returns 0.0 when inputs are empty.
    """
    if not y_true or not y_pred:
        return 0.0
    return float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )


# ── AdversarialEvaluator ─────────────────────────────────────────────

class AdversarialEvaluator:
    """Evaluate grading models on adversarial test data.

    Parameters
    ----------
    models : dict[str, GradingModelProtocol]
        Mapping of model name → trained grading model instance.
    label_field : str
        The ``UnifiedRecord`` attribute used as the ground-truth label
        (e.g. ``"label_3way"``).  Defaults to ``"label_3way"``.
    """

    def __init__(
        self,
        models: dict[str, Any],
        label_field: str = "label_3way",
    ) -> None:
        self.models = models
        self.label_field = label_field

    # ── public API ────────────────────────────────────────────────

    def evaluate(
        self,
        adversarial_records: list[UnifiedRecord],
        clean_f1: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run adversarial evaluation and produce the vulnerability matrix.

        Parameters
        ----------
        adversarial_records : list[UnifiedRecord]
            Records from the adversarial split.  Each record must have
            a non-null ``perturbation_type``.
        clean_f1 : dict[str, float] | None
            Mapping of ``model_name → Macro-F1 on clean test set``.
            When provided, ``Δ_F1 = F1_clean - F1_adversarial`` is
            computed per model per perturbation type.

        Returns
        -------
        dict
            Keys:
            - ``f1_per_model_perturbation``: nested dict
              ``{model_name: {perturbation_type: macro_f1}}``
            - ``delta_f1``: nested dict (present only when *clean_f1*
              is provided) ``{model_name: {perturbation_type: Δ_F1}}``
            - ``vulnerability_matrix``: same structure as ``delta_f1``
              (alias for downstream consumers)
            - ``perturbation_categories``: dict
              ``{perturbation_type: category}``
            - ``category_summary``: nested dict
              ``{model_name: {category: mean_Δ_F1}}``
        """
        # Group records by perturbation_type
        grouped = self._group_by_perturbation(adversarial_records)

        # Compute F1 per model × perturbation_type
        f1_results: dict[str, dict[str, float]] = {}
        for model_name, model in self.models.items():
            f1_results[model_name] = {}
            for pt, records in grouped.items():
                y_true = [
                    getattr(rec, self.label_field) for rec in records
                ]
                y_pred = model.predict(records)
                # Ensure predictions are strings for classification F1
                y_pred_str = [str(p) for p in y_pred]
                y_true_str = [str(t) for t in y_true]
                f1_results[model_name][pt] = _compute_macro_f1(
                    y_true_str, y_pred_str,
                )

        # Build result dict
        result: dict[str, Any] = {
            "f1_per_model_perturbation": f1_results,
        }

        # Compute Δ_F1 if clean results are provided
        if clean_f1 is not None:
            delta_f1 = self._compute_delta_f1(f1_results, clean_f1)
            result["delta_f1"] = delta_f1
            result["vulnerability_matrix"] = delta_f1

            # Category-level summary (mean Δ_F1 per category per model)
            result["category_summary"] = self._compute_category_summary(
                delta_f1,
            )

        # Perturbation categories for all observed types
        all_perturbation_types = set()
        for pt_dict in f1_results.values():
            all_perturbation_types.update(pt_dict.keys())

        result["perturbation_categories"] = {
            pt: categorize_perturbation(pt)
            for pt in sorted(all_perturbation_types)
        }

        return result

    # ── internal helpers ──────────────────────────────────────────

    @staticmethod
    def _group_by_perturbation(
        records: list[UnifiedRecord],
    ) -> dict[str, list[UnifiedRecord]]:
        """Group records by their ``perturbation_type``."""
        grouped: dict[str, list[UnifiedRecord]] = defaultdict(list)
        for rec in records:
            pt = rec.perturbation_type
            if pt is not None:
                grouped[pt].append(rec)
        return dict(grouped)

    @staticmethod
    def _compute_delta_f1(
        f1_results: dict[str, dict[str, float]],
        clean_f1: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        """Compute Δ_F1 = F1_clean - F1_adversarial per model per perturbation."""
        delta: dict[str, dict[str, float]] = {}
        for model_name, pt_f1 in f1_results.items():
            clean = clean_f1.get(model_name, 0.0)
            delta[model_name] = {
                pt: clean - adv_f1
                for pt, adv_f1 in pt_f1.items()
            }
        return delta

    @staticmethod
    def _compute_category_summary(
        delta_f1: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Compute mean Δ_F1 per perturbation category per model."""
        summary: dict[str, dict[str, float]] = {}
        for model_name, pt_deltas in delta_f1.items():
            cat_values: dict[str, list[float]] = defaultdict(list)
            for pt, d in pt_deltas.items():
                cat = categorize_perturbation(pt)
                cat_values[cat].append(d)
            summary[model_name] = {
                cat: sum(vals) / len(vals) if vals else 0.0
                for cat, vals in cat_values.items()
            }
        return summary
