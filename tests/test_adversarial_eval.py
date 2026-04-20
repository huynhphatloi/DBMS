"""Integration tests for the AdversarialEvaluator.

Verifies:
- Evaluation on adversarial split produces F1 per model × perturbation_type
- Δ_F1 computation is correct
- Vulnerability matrix is produced with expected structure
- Perturbation type categorization works
"""

from __future__ import annotations

import pytest

from src.data.schema import UnifiedRecord
from src.robustness.adversarial_eval import (
    AdversarialEvaluator,
    PERTURBATION_TAXONOMY,
    categorize_perturbation,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_record(
    sample_id: str,
    label_3way: str,
    perturbation_type: str | None = None,
    is_adversarial: bool = False,
) -> UnifiedRecord:
    """Create a minimal UnifiedRecord for testing."""
    return UnifiedRecord(
        sample_id=sample_id,
        source_dataset="data_generate",
        original_id=sample_id,
        question_id="q1",
        domain="science",
        subdomain="physics",
        difficulty="medium",
        question="What is gravity?",
        reference_answer="Gravity is a force.",
        student_answer="Gravity pulls things down.",
        label_3way=label_3way,
        perturbation_type=perturbation_type,
        is_adversarial=is_adversarial,
        split="test_adversarial",
    )


class FakeGradingModel:
    """A fake grading model that returns pre-configured predictions."""

    def __init__(self, predictions: list[str]) -> None:
        self._predictions = predictions
        self._call_idx = 0

    def predict(self, records) -> list[str]:
        records_list = list(records)
        n = len(records_list)
        result = self._predictions[self._call_idx : self._call_idx + n]
        self._call_idx += n
        return result

    def predict_proba(self, records) -> list[list[float]]:
        return [[1.0, 0.0, 0.0]] * len(list(records))


class PerfectModel:
    """A model that always predicts the ground-truth label."""

    def predict(self, records) -> list[str]:
        return [rec.label_3way for rec in records]

    def predict_proba(self, records) -> list[list[float]]:
        return [[1.0, 0.0, 0.0]] * len(list(records))


class ConstantModel:
    """A model that always predicts a fixed label."""

    def __init__(self, label: str = "incorrect") -> None:
        self._label = label

    def predict(self, records) -> list[str]:
        return [self._label] * len(list(records))

    def predict_proba(self, records) -> list[list[float]]:
        return [[1.0, 0.0, 0.0]] * len(list(records))


# ── Tests: Perturbation categorization ────────────────────────────────

class TestPerturbationCategorization:
    def test_surface_level_types(self):
        for pt in PERTURBATION_TAXONOMY["surface_level"]:
            assert categorize_perturbation(pt) == "surface_level"

    def test_semantic_types(self):
        for pt in PERTURBATION_TAXONOMY["semantic"]:
            assert categorize_perturbation(pt) == "semantic"

    def test_gaming_deception_types(self):
        for pt in PERTURBATION_TAXONOMY["gaming_deception"]:
            assert categorize_perturbation(pt) == "gaming_deception"

    def test_unknown_type(self):
        assert categorize_perturbation("nonexistent_type") == "unknown"

    def test_taxonomy_covers_12_types(self):
        all_types = []
        for types in PERTURBATION_TAXONOMY.values():
            all_types.extend(types)
        assert len(all_types) == 12


# ── Tests: AdversarialEvaluator basic evaluation ─────────────────────

class TestAdversarialEvaluatorBasic:
    def test_evaluate_produces_f1_per_model_perturbation(self):
        """Evaluate on adversarial split produces F1 per model × perturbation."""
        records = [
            _make_record("r1", "correct", "synonym_swap", True),
            _make_record("r2", "incorrect", "synonym_swap", True),
            _make_record("r3", "correct", "near-contradiction", True),
            _make_record("r4", "incorrect", "near-contradiction", True),
        ]
        model = PerfectModel()
        evaluator = AdversarialEvaluator(
            models={"perfect": model},
            label_field="label_3way",
        )
        result = evaluator.evaluate(records)

        assert "f1_per_model_perturbation" in result
        f1s = result["f1_per_model_perturbation"]
        assert "perfect" in f1s
        assert "synonym_swap" in f1s["perfect"]
        assert "near-contradiction" in f1s["perfect"]
        # Perfect model should get F1 = 1.0
        assert f1s["perfect"]["synonym_swap"] == 1.0
        assert f1s["perfect"]["near-contradiction"] == 1.0

    def test_evaluate_with_multiple_models(self):
        """Evaluate with multiple models produces results for each."""
        records = [
            _make_record("r1", "correct", "synonym_swap", True),
            _make_record("r2", "incorrect", "synonym_swap", True),
        ]
        models = {
            "perfect": PerfectModel(),
            "constant": ConstantModel("incorrect"),
        }
        evaluator = AdversarialEvaluator(models=models)
        result = evaluator.evaluate(records)

        f1s = result["f1_per_model_perturbation"]
        assert "perfect" in f1s
        assert "constant" in f1s
        assert f1s["perfect"]["synonym_swap"] == 1.0
        # Constant model predicts all "incorrect" → misses "correct"
        assert f1s["constant"]["synonym_swap"] < 1.0

    def test_records_without_perturbation_type_are_skipped(self):
        """Records with perturbation_type=None are excluded."""
        records = [
            _make_record("r1", "correct", None, False),
            _make_record("r2", "correct", "synonym_swap", True),
            _make_record("r3", "incorrect", "synonym_swap", True),
        ]
        evaluator = AdversarialEvaluator(
            models={"perfect": PerfectModel()},
        )
        result = evaluator.evaluate(records)
        f1s = result["f1_per_model_perturbation"]
        # Only synonym_swap should appear
        assert list(f1s["perfect"].keys()) == ["synonym_swap"]


# ── Tests: Δ_F1 and vulnerability matrix ─────────────────────────────

class TestDeltaF1:
    def test_delta_f1_computation(self):
        """Δ_F1 = F1_clean - F1_adversarial is computed correctly."""
        records = [
            _make_record("r1", "correct", "synonym_swap", True),
            _make_record("r2", "incorrect", "synonym_swap", True),
        ]
        # Constant model always predicts "incorrect"
        evaluator = AdversarialEvaluator(
            models={"constant": ConstantModel("incorrect")},
        )
        clean_f1 = {"constant": 0.8}
        result = evaluator.evaluate(records, clean_f1=clean_f1)

        assert "delta_f1" in result
        assert "vulnerability_matrix" in result
        # delta_f1 and vulnerability_matrix should be the same
        assert result["delta_f1"] == result["vulnerability_matrix"]

        delta = result["delta_f1"]["constant"]["synonym_swap"]
        adv_f1 = result["f1_per_model_perturbation"]["constant"]["synonym_swap"]
        assert abs(delta - (0.8 - adv_f1)) < 1e-9

    def test_delta_f1_zero_for_perfect_model(self):
        """Perfect model on adversarial data with clean_f1=1.0 → Δ_F1=0."""
        records = [
            _make_record("r1", "correct", "synonym_swap", True),
            _make_record("r2", "incorrect", "synonym_swap", True),
        ]
        evaluator = AdversarialEvaluator(
            models={"perfect": PerfectModel()},
        )
        result = evaluator.evaluate(records, clean_f1={"perfect": 1.0})
        assert result["delta_f1"]["perfect"]["synonym_swap"] == pytest.approx(0.0)

    def test_vulnerability_matrix_structure(self):
        """Vulnerability matrix has model × perturbation_type → Δ_F1."""
        records = [
            _make_record("r1", "correct", "synonym_swap", True),
            _make_record("r2", "incorrect", "near-contradiction", True),
        ]
        evaluator = AdversarialEvaluator(
            models={"m1": PerfectModel(), "m2": ConstantModel()},
        )
        result = evaluator.evaluate(
            records, clean_f1={"m1": 1.0, "m2": 0.5},
        )
        vm = result["vulnerability_matrix"]
        assert set(vm.keys()) == {"m1", "m2"}
        for model_name in vm:
            assert "synonym_swap" in vm[model_name]
            assert "near-contradiction" in vm[model_name]


# ── Tests: Category summary ──────────────────────────────────────────

class TestCategorySummary:
    def test_category_summary_produced(self):
        """Category summary groups Δ_F1 by perturbation category."""
        records = [
            _make_record("r1", "correct", "synonym_swap", True),
            _make_record("r2", "incorrect", "synonym_swap", True),
            _make_record("r3", "correct", "near-contradiction", True),
            _make_record("r4", "incorrect", "near-contradiction", True),
        ]
        evaluator = AdversarialEvaluator(
            models={"m": PerfectModel()},
        )
        result = evaluator.evaluate(
            records, clean_f1={"m": 1.0},
        )
        assert "category_summary" in result
        summary = result["category_summary"]["m"]
        assert "surface_level" in summary
        assert "semantic" in summary

    def test_perturbation_categories_in_result(self):
        """Result includes perturbation_categories mapping."""
        records = [
            _make_record("r1", "correct", "synonym_swap", True),
            _make_record("r2", "correct", "hedge_language", True),
        ]
        evaluator = AdversarialEvaluator(
            models={"m": PerfectModel()},
        )
        result = evaluator.evaluate(records)
        cats = result["perturbation_categories"]
        assert cats["synonym_swap"] == "surface_level"
        assert cats["hedge_language"] == "gaming_deception"


# ── Tests: Integration — full adversarial evaluation ─────────────────

class TestIntegrationAdversarialEval:
    def test_full_evaluation_with_multiple_perturbation_types(self):
        """Run evaluation on adversarial split with multiple perturbation
        types and verify the vulnerability matrix is produced."""
        # Create records spanning multiple perturbation types
        perturbation_types = [
            "synonym_swap",
            "paraphrase_low_overlap",
            "near-contradiction",
            "high_overlap_wrong_meaning",
        ]
        records = []
        for i, pt in enumerate(perturbation_types):
            records.append(
                _make_record(f"r{i*2}", "correct", pt, True),
            )
            records.append(
                _make_record(f"r{i*2+1}", "incorrect", pt, True),
            )

        models = {
            "perfect": PerfectModel(),
            "constant_incorrect": ConstantModel("incorrect"),
        }
        evaluator = AdversarialEvaluator(models=models)
        clean_f1 = {"perfect": 1.0, "constant_incorrect": 0.4}

        result = evaluator.evaluate(records, clean_f1=clean_f1)

        # Verify structure
        assert "f1_per_model_perturbation" in result
        assert "delta_f1" in result
        assert "vulnerability_matrix" in result
        assert "perturbation_categories" in result
        assert "category_summary" in result

        # Verify all perturbation types present
        for model_name in models:
            for pt in perturbation_types:
                assert pt in result["f1_per_model_perturbation"][model_name]
                assert pt in result["vulnerability_matrix"][model_name]

        # Verify categories
        cats = result["perturbation_categories"]
        assert cats["synonym_swap"] == "surface_level"
        assert cats["paraphrase_low_overlap"] == "surface_level"
        assert cats["near-contradiction"] == "semantic"
        assert cats["high_overlap_wrong_meaning"] == "gaming_deception"

        # Verify vulnerability matrix values are numeric
        for model_name, pt_deltas in result["vulnerability_matrix"].items():
            for pt, delta in pt_deltas.items():
                assert isinstance(delta, float)

    def test_evaluation_with_all_12_perturbation_types(self):
        """Verify evaluation handles all 12 Data_Generate perturbation types."""
        all_types = []
        for types in PERTURBATION_TAXONOMY.values():
            all_types.extend(types)

        records = []
        for i, pt in enumerate(all_types):
            records.append(
                _make_record(f"r{i}", "correct", pt, True),
            )

        evaluator = AdversarialEvaluator(
            models={"m": PerfectModel()},
        )
        result = evaluator.evaluate(records, clean_f1={"m": 1.0})

        # All 12 types should appear
        assert len(result["f1_per_model_perturbation"]["m"]) == 12
        assert len(result["vulnerability_matrix"]["m"]) == 12

        # All 3 categories should appear in summary
        summary = result["category_summary"]["m"]
        assert set(summary.keys()) == {
            "surface_level",
            "semantic",
            "gaming_deception",
        }

    def test_empty_adversarial_records(self):
        """Evaluation with no adversarial records produces empty results."""
        evaluator = AdversarialEvaluator(
            models={"m": PerfectModel()},
        )
        result = evaluator.evaluate([])
        assert result["f1_per_model_perturbation"]["m"] == {}
