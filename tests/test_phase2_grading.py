"""Tests for experiments/phase2_grading.py — Phase 2 grading experiments.

Validates config loading, data loading helpers, model creation factories,
experiment runner functions, augmentation delta computation, and result
serialization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DataLoader
from src.data.schema import UnifiedRecord


# ---------------------------------------------------------------------------
# Helpers — create minimal UnifiedRecord instances
# ---------------------------------------------------------------------------

def _make_record(
    sample_id: str = "SEB_0001",
    source_dataset: str = "scientsbank",
    split: str = "train",
    question: str = "What is photosynthesis?",
    reference_answer: str = "The process by which plants convert sunlight.",
    student_answer: str = "Plants use sunlight to make food.",
    label_2way: str | None = "correct",
    label_3way: str | None = "correct",
    label_5way: str | None = "correct",
    score_raw: float | None = None,
    score_normalized: float | None = None,
    annotation_confidence: float | None = None,
) -> UnifiedRecord:
    return UnifiedRecord(
        sample_id=sample_id,
        source_dataset=source_dataset,
        original_id=sample_id,
        question_id="Q1",
        domain="science",
        subdomain="biology",
        difficulty="medium",
        question=question,
        reference_answer=reference_answer,
        student_answer=student_answer,
        label_2way=label_2way,
        label_3way=label_3way,
        label_5way=label_5way,
        score_raw=score_raw,
        score_normalized=score_normalized,
        split=split,
        annotation_confidence=annotation_confidence,
    )


def _make_seb_dataset() -> list[UnifiedRecord]:
    """Create a small SciEntsBank-like dataset with train + test splits."""
    records = []
    labels_3way = ["correct", "partially_correct", "incorrect"]
    labels_5way = [
        "correct",
        "partially_correct_incomplete",
        "contradictory",
        "irrelevant",
        "non_domain",
    ]

    # Train records
    for i in range(15):
        lbl_3 = labels_3way[i % 3]
        lbl_5 = labels_5way[i % 5]
        lbl_2 = "correct" if lbl_3 == "correct" else "incorrect"
        records.append(
            _make_record(
                sample_id=f"SEB_TRAIN_{i:04d}",
                split="train",
                student_answer=f"Student answer {i} for training",
                label_2way=lbl_2,
                label_3way=lbl_3,
                label_5way=lbl_5,
            )
        )

    # Test UA records
    for i in range(6):
        lbl_3 = labels_3way[i % 3]
        lbl_2 = "correct" if lbl_3 == "correct" else "incorrect"
        records.append(
            _make_record(
                sample_id=f"SEB_UA_{i:04d}",
                split="test_ua",
                student_answer=f"Student answer {i} for test UA",
                label_2way=lbl_2,
                label_3way=lbl_3,
                label_5way=labels_5way[i % 5],
            )
        )

    # Test UQ records
    for i in range(6):
        lbl_3 = labels_3way[i % 3]
        lbl_2 = "correct" if lbl_3 == "correct" else "incorrect"
        records.append(
            _make_record(
                sample_id=f"SEB_UQ_{i:04d}",
                split="test_uq",
                student_answer=f"Student answer {i} for test UQ",
                label_2way=lbl_2,
                label_3way=lbl_3,
                label_5way=labels_5way[i % 5],
            )
        )

    # Test UD records
    for i in range(6):
        lbl_3 = labels_3way[i % 3]
        lbl_2 = "correct" if lbl_3 == "correct" else "incorrect"
        records.append(
            _make_record(
                sample_id=f"SEB_UD_{i:04d}",
                split="test_ud",
                student_answer=f"Student answer {i} for test UD",
                label_2way=lbl_2,
                label_3way=lbl_3,
                label_5way=labels_5way[i % 5],
            )
        )

    return records


def _make_mohler_dataset() -> list[UnifiedRecord]:
    """Create a small MohlerASAG-like dataset with train/test splits."""
    records = []
    for i in range(10):
        score = (i / 9) * 5.0
        norm = score / 5.0
        lbl_2 = "correct" if score >= 2.5 else "incorrect"
        if score < 1.0:
            lbl_3 = "incorrect"
        elif score < 4.0:
            lbl_3 = "partially_correct"
        else:
            lbl_3 = "correct"
        records.append(
            _make_record(
                sample_id=f"MOH_{i:04d}",
                source_dataset="mohler",
                split="train",
                student_answer=f"Mohler student answer {i}",
                label_2way=lbl_2,
                label_3way=lbl_3,
                label_5way=None,
                score_raw=score,
                score_normalized=norm,
            )
        )
    for i in range(5):
        score = (i / 4) * 5.0
        norm = score / 5.0
        lbl_2 = "correct" if score >= 2.5 else "incorrect"
        records.append(
            _make_record(
                sample_id=f"MOH_TEST_{i:04d}",
                source_dataset="mohler",
                split="test",
                student_answer=f"Mohler test answer {i}",
                label_2way=lbl_2,
                label_3way="correct" if score >= 4 else "incorrect",
                label_5way=None,
                score_raw=score,
                score_normalized=norm,
            )
        )
    return records


def _make_data_generate_dataset() -> list[UnifiedRecord]:
    """Create a small Data_Generate-like dataset."""
    records = []
    labels_3way = ["correct", "partially_correct", "incorrect"]
    for i in range(10):
        lbl_3 = labels_3way[i % 3]
        lbl_2 = "correct" if lbl_3 == "correct" else "incorrect"
        records.append(
            _make_record(
                sample_id=f"GEN_{i:04d}",
                source_dataset="data_generate",
                split="train",
                student_answer=f"Generated student answer {i}",
                label_2way=lbl_2,
                label_3way=lbl_3,
                label_5way=None,
                annotation_confidence=0.80 + (i * 0.02),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grading_config() -> dict:
    """Minimal grading config for testing."""
    return {
        "seed": 42,
        "task": "3way",
        "bootstrap_iterations": 10,
        "lexical": {"metric": "rouge_l", "threshold": 0.5},
        "tfidf_ml": {
            "max_features": 100,
            "classifiers": ["logistic_regression"],
        },
        "sbert": {"model_name": "all-MiniLM-L6-v2", "threshold": 0.5},
        "cross_encoder": {
            "model_name": "cross-encoder/stsb-roberta-base",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 2e-5,
        },
        "ref_aware": {
            "encoder": "microsoft/deberta-v3-base",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "max_length": 128,
            "multi_task": {"enabled": True, "alpha": 0.7},
        },
    }


@pytest.fixture
def seb_data_loader() -> DataLoader:
    """DataLoader with SciEntsBank records."""
    return DataLoader(_make_seb_dataset())


@pytest.fixture
def full_data_loader() -> DataLoader:
    """DataLoader with SciEntsBank + Mohler + Data_Generate records."""
    records = _make_seb_dataset() + _make_mohler_dataset() + _make_data_generate_dataset()
    return DataLoader(records)


# ---------------------------------------------------------------------------
# Tests: Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_config_from_yaml(self, tmp_path):
        """load_config reads a YAML file and returns a dict."""
        from experiments.phase2_grading import load_config

        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text("seed: 99\ntask: 2way\n")
        result = load_config(cfg_file)
        assert result["seed"] == 99
        assert result["task"] == "2way"

    def test_load_config_missing_file(self, tmp_path):
        """load_config raises FileNotFoundError for missing file."""
        from experiments.phase2_grading import load_config

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_get_label_field_for_task_2way(self):
        from experiments.phase2_grading import get_label_field_for_task

        assert get_label_field_for_task("2way") == "label_2way"

    def test_get_label_field_for_task_3way(self):
        from experiments.phase2_grading import get_label_field_for_task

        assert get_label_field_for_task("3way") == "label_3way"

    def test_get_label_field_for_task_5way(self):
        from experiments.phase2_grading import get_label_field_for_task

        assert get_label_field_for_task("5way") == "label_5way"

    def test_get_label_field_for_task_regression(self):
        from experiments.phase2_grading import get_label_field_for_task

        assert get_label_field_for_task("regression") == "score_normalized"

    def test_get_label_field_for_task_invalid(self):
        from experiments.phase2_grading import get_label_field_for_task

        with pytest.raises(ValueError, match="Unknown task"):
            get_label_field_for_task("invalid")

    def test_get_num_labels(self):
        from experiments.phase2_grading import get_num_labels

        assert get_num_labels("2way") == 2
        assert get_num_labels("3way") == 3
        assert get_num_labels("5way") == 5
        # Unknown defaults to 3
        assert get_num_labels("regression") == 3


# ---------------------------------------------------------------------------
# Tests: Model creation factories
# ---------------------------------------------------------------------------

class TestModelCreation:
    def test_create_lexical_models(self, grading_config):
        from experiments.phase2_grading import create_lexical_models

        models = create_lexical_models(grading_config)
        assert len(models) == 2
        names = [name for name, _ in models]
        assert "lexical_threshold" in names
        assert "lexical_logreg" in names

    def test_create_tfidf_models(self, grading_config):
        from experiments.phase2_grading import create_tfidf_models

        models = create_tfidf_models(grading_config)
        assert len(models) == 1  # Only logistic_regression in test config
        assert models[0][0] == "tfidf_logistic_regression"

    def test_create_tfidf_regressors(self, grading_config):
        from experiments.phase2_grading import create_tfidf_regressors

        models = create_tfidf_regressors(grading_config)
        assert len(models) == 2
        names = [name for name, _ in models]
        assert "tfidf_ridge_reg" in names
        assert "tfidf_svr_reg" in names

    def test_create_sbert_models(self, grading_config):
        from experiments.phase2_grading import create_sbert_models

        models = create_sbert_models(grading_config)
        assert len(models) == 2
        names = [name for name, _ in models]
        assert "sbert_threshold" in names
        assert "sbert_logreg" in names

    def test_create_llm_zeroshot_no_api_key(self, grading_config):
        """LLM zero-shot returns None when no API key is set."""
        from experiments.phase2_grading import create_llm_zeroshot

        with patch.dict("os.environ", {}, clear=True):
            result = create_llm_zeroshot(grading_config, "3way")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Classification experiment runner
# ---------------------------------------------------------------------------

class TestRunClassificationExperiment:
    def test_classification_experiment_with_lexical(self, seb_data_loader):
        """run_classification_experiment trains and evaluates a model."""
        from experiments.phase2_grading import (
            create_lexical_models,
            run_classification_experiment,
        )

        train = seb_data_loader.get_split("scientsbank", "train")
        test_splits = {
            "test_ua": seb_data_loader.get_split("scientsbank", "test_ua"),
        }

        models = create_lexical_models({"lexical": {"metric": "rouge_l", "threshold": 0.5}})
        model_name, model = models[0]

        result = run_classification_experiment(
            model_name=model_name,
            model=model,
            train_records=train,
            test_splits=test_splits,
            label_field="label_3way",
            bootstrap_n=10,
        )

        assert result["model"] == model_name
        assert result["task"] == "classification"
        assert "splits" in result
        assert "test_ua" in result["splits"]
        metrics = result["splits"]["test_ua"]
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "weighted_f1" in metrics
        assert "confusion_matrix" in metrics

    def test_classification_experiment_empty_test_split(self, seb_data_loader):
        """Empty test splits are skipped gracefully."""
        from experiments.phase2_grading import (
            create_lexical_models,
            run_classification_experiment,
        )

        train = seb_data_loader.get_split("scientsbank", "train")
        test_splits = {"empty_split": []}

        models = create_lexical_models({"lexical": {"metric": "rouge_l", "threshold": 0.5}})
        model_name, model = models[0]

        result = run_classification_experiment(
            model_name=model_name,
            model=model,
            train_records=train,
            test_splits=test_splits,
            label_field="label_3way",
            bootstrap_n=10,
        )

        assert result["splits"] == {}


# ---------------------------------------------------------------------------
# Tests: Regression experiment runner
# ---------------------------------------------------------------------------

class TestRunRegressionExperiment:
    def test_regression_experiment_with_tfidf(self):
        """run_regression_experiment trains and evaluates a regressor."""
        from experiments.phase2_grading import (
            create_tfidf_regressors,
            run_regression_experiment,
        )

        mohler = _make_mohler_dataset()
        train = [r for r in mohler if r.split == "train"]
        test = [r for r in mohler if r.split == "test"]

        models = create_tfidf_regressors({"tfidf_ml": {"max_features": 100}})
        model_name, model = models[0]

        result = run_regression_experiment(
            model_name=model_name,
            model=model,
            train_records=train,
            test_splits={"test": test},
            label_field="score_normalized",
            bootstrap_n=10,
        )

        assert result["model"] == model_name
        assert result["task"] == "regression"
        assert "splits" in result
        assert "test" in result["splits"]
        metrics = result["splits"]["test"]
        assert "pearson_r" in metrics
        assert "rmse" in metrics
        assert "qwk" in metrics


# ---------------------------------------------------------------------------
# Tests: In-domain experiments (16.2)
# ---------------------------------------------------------------------------

class TestInDomainExperiments:
    def test_in_domain_returns_results(self, seb_data_loader, grading_config):
        """run_in_domain_experiments returns results for each task."""
        from experiments.phase2_grading import run_in_domain_experiments

        # Use only lexical models for speed
        cfg = dict(grading_config)
        results = run_in_domain_experiments(
            data_loader=seb_data_loader,
            cfg=cfg,
            bootstrap_n=10,
            skip_llm=True,
        )

        assert isinstance(results, list)
        # Should have results for 2way, 3way, 5way tasks
        tasks_found = {r["task_formulation"] for r in results}
        assert "2way" in tasks_found
        assert "3way" in tasks_found

        # Each result should have experiment type
        for r in results:
            assert r["experiment"] == "in_domain"
            assert "splits" in r

    def test_in_domain_no_data_returns_empty(self, grading_config):
        """Returns empty list when no SciEntsBank data is available."""
        from experiments.phase2_grading import run_in_domain_experiments

        empty_loader = DataLoader([])
        results = run_in_domain_experiments(
            data_loader=empty_loader,
            cfg=grading_config,
            bootstrap_n=10,
        )
        assert results == []


# ---------------------------------------------------------------------------
# Tests: MohlerASAG regression experiments (16.3)
# ---------------------------------------------------------------------------

class TestMohlerRegressionExperiments:
    def test_mohler_regression_returns_results(self, grading_config):
        """run_mohler_regression_experiments returns regression metrics."""
        from experiments.phase2_grading import run_mohler_regression_experiments

        mohler_records = _make_mohler_dataset()
        loader = DataLoader(mohler_records)

        results = run_mohler_regression_experiments(
            data_loader=loader,
            cfg=grading_config,
            bootstrap_n=10,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert r["experiment"] == "mohler_regression"
            assert "splits" in r

    def test_mohler_regression_no_data_returns_empty(self, grading_config):
        """Returns empty list when no Mohler data is available."""
        from experiments.phase2_grading import run_mohler_regression_experiments

        empty_loader = DataLoader([])
        results = run_mohler_regression_experiments(
            data_loader=empty_loader,
            cfg=grading_config,
            bootstrap_n=10,
        )
        assert results == []


# ---------------------------------------------------------------------------
# Tests: Cross-domain experiments (16.4)
# ---------------------------------------------------------------------------

class TestCrossDomainExperiments:
    def test_cross_domain_returns_results(self, full_data_loader, grading_config):
        """run_cross_domain_experiments returns results for transfers."""
        from experiments.phase2_grading import run_cross_domain_experiments

        results = run_cross_domain_experiments(
            data_loader=full_data_loader,
            cfg=grading_config,
            bootstrap_n=10,
            skip_llm=True,
        )

        assert isinstance(results, list)
        # Should have at least some cross-domain results
        for r in results:
            assert r["experiment"] == "cross_domain"
            assert "transfer" in r

    def test_cross_domain_no_data_returns_empty(self, grading_config):
        """Returns empty list when no data is available."""
        from experiments.phase2_grading import run_cross_domain_experiments

        empty_loader = DataLoader([])
        results = run_cross_domain_experiments(
            data_loader=empty_loader,
            cfg=grading_config,
            bootstrap_n=10,
        )
        assert results == []


# ---------------------------------------------------------------------------
# Tests: Augmentation ablation (16.5)
# ---------------------------------------------------------------------------

class TestAugmentationAblation:
    def test_augmentation_ablation_returns_results(
        self, full_data_loader, grading_config
    ):
        """run_augmentation_ablation returns results for different configs."""
        from experiments.phase2_grading import run_augmentation_ablation

        results = run_augmentation_ablation(
            data_loader=full_data_loader,
            cfg=grading_config,
            bootstrap_n=10,
        )

        assert isinstance(results, list)
        assert len(results) > 0

        # Should have scientsbank_only and augmented configs
        train_configs = {
            r.get("train_config")
            for r in results
            if r.get("experiment") == "augmentation_ablation"
        }
        assert "scientsbank_only" in train_configs

    def test_augmentation_ablation_no_seb_returns_empty(self, grading_config):
        """Returns empty list when no SciEntsBank data is available."""
        from experiments.phase2_grading import run_augmentation_ablation

        gen_only = DataLoader(_make_data_generate_dataset())
        results = run_augmentation_ablation(
            data_loader=gen_only,
            cfg=grading_config,
            bootstrap_n=10,
        )
        assert results == []


# ---------------------------------------------------------------------------
# Tests: Augmentation delta computation
# ---------------------------------------------------------------------------

class TestComputeAugmentationDeltas:
    def test_compute_deltas_with_matching_results(self):
        """compute_augmentation_deltas computes F1 differences."""
        from experiments.phase2_grading import compute_augmentation_deltas

        results = [
            {
                "experiment": "augmentation_ablation",
                "model": "lexical_threshold",
                "train_config": "scientsbank_only",
                "splits": {
                    "test_ua": {"macro_f1": {"value": 0.5}},
                },
            },
            {
                "experiment": "augmentation_ablation",
                "model": "lexical_threshold",
                "train_config": "scientsbank_plus_data_generate",
                "splits": {
                    "test_ua": {"macro_f1": {"value": 0.6}},
                },
            },
        ]

        deltas = compute_augmentation_deltas(results)
        assert "scientsbank_plus_data_generate" in deltas
        model_deltas = deltas["scientsbank_plus_data_generate"]
        assert "lexical_threshold" in model_deltas
        assert model_deltas["lexical_threshold"]["test_ua"] == pytest.approx(0.1)

    def test_compute_deltas_empty_results(self):
        """Returns empty dict for empty results."""
        from experiments.phase2_grading import compute_augmentation_deltas

        assert compute_augmentation_deltas([]) == {}


# ---------------------------------------------------------------------------
# Tests: Result saving (16.6)
# ---------------------------------------------------------------------------

class TestResultSaving:
    def test_save_results_creates_json(self, tmp_path):
        """save_results writes a valid JSON file."""
        from src.evaluation.reporting import save_results

        results = {"model": "test", "accuracy": 0.95}
        path = save_results(results, "test_results", results_dir=str(tmp_path))

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "timestamp" in data
        assert data["results"]["model"] == "test"
        assert data["results"]["accuracy"] == 0.95

    def test_results_dir_created_if_missing(self, tmp_path):
        """save_results creates the results directory if it doesn't exist."""
        from src.evaluation.reporting import save_results

        results_dir = tmp_path / "new_dir" / "phase2"
        save_results({"test": True}, "output", results_dir=str(results_dir))
        assert (results_dir / "output.json").exists()


# ---------------------------------------------------------------------------
# Tests: load_unified_records
# ---------------------------------------------------------------------------

class TestLoadUnifiedRecords:
    def test_load_from_jsonl(self, tmp_path):
        """load_unified_records reads JSONL files into UnifiedRecord."""
        from experiments.phase2_grading import load_unified_records

        rec = _make_record()
        import dataclasses

        rec_dict = dataclasses.asdict(rec)
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(json.dumps(rec_dict) + "\n")

        records = load_unified_records(tmp_path)
        assert len(records) == 1
        assert records[0].sample_id == rec.sample_id

    def test_load_skips_malformed_lines(self, tmp_path):
        """Malformed JSONL lines are skipped with a warning."""
        from experiments.phase2_grading import load_unified_records

        jsonl_file = tmp_path / "bad.jsonl"
        jsonl_file.write_text("not valid json\n")

        records = load_unified_records(tmp_path)
        assert len(records) == 0

    def test_load_empty_dir(self, tmp_path):
        """Empty directory returns empty list."""
        from experiments.phase2_grading import load_unified_records

        records = load_unified_records(tmp_path)
        assert records == []
