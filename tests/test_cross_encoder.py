"""Integration tests for the CrossEncoder baseline (task 13.5).

All HuggingFace model/tokenizer/Trainer calls are mocked via sys.modules so
no real weights are downloaded and no GPU is required.  Tests verify:
  - fit() runs without error on a small dataset
  - predict() returns valid labels (classification) or floats (regression)
  - predict_proba() returns probabilities that sum to ~1 (classification)
  - evaluate_cross_encoder() returns macro_f1 with CI fields (classification)
  - evaluate_cross_encoder() returns pearson_r with CI fields (regression)
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock

import torch

from src.data.schema import UnifiedRecord


# ---------------------------------------------------------------------------
# Build a fake `transformers` module before importing cross_encoder
# ---------------------------------------------------------------------------

def _build_fake_transformers():
    """Return a minimal fake transformers module."""
    fake = types.ModuleType("transformers")

    # AutoTokenizer
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            tok = MagicMock()

            def _call(text_a, text_b, padding=True, truncation=True,
                      max_length=256, return_tensors=None):
                n = len(text_a)
                seq_len = 8
                ids = torch.ones(n, seq_len, dtype=torch.long)
                mask = torch.ones(n, seq_len, dtype=torch.long)
                if return_tensors == "pt":
                    return {"input_ids": ids, "attention_mask": mask}
                return {
                    "input_ids": ids.tolist(),
                    "attention_mask": mask.tolist(),
                }

            tok.side_effect = _call
            return tok

    # AutoModelForSequenceClassification
    class FakeAutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(name, num_labels=2):
            model = MagicMock()
            model.eval = MagicMock(return_value=model)
            model.train = MagicMock(return_value=model)
            model.parameters = MagicMock(return_value=iter([]))
            model.config = MagicMock()
            model.config.problem_type = None

            def _forward(**kwargs):
                n = kwargs["input_ids"].shape[0]
                logits = torch.randn(n, num_labels)
                out = MagicMock()
                out.logits = logits
                out.loss = torch.tensor(0.5)
                return out

            model.side_effect = _forward
            model.__call__ = _forward
            return model

    # TrainingArguments — just a data holder
    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Trainer — calls train() on the dataset without doing anything real
    class FakeTrainer:
        def __init__(self, model=None, args=None, train_dataset=None, **kwargs):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset

        def train(self):
            pass  # no-op

    fake.AutoTokenizer = FakeAutoTokenizer
    fake.AutoModelForSequenceClassification = FakeAutoModelForSequenceClassification
    fake.TrainingArguments = FakeTrainingArguments
    fake.Trainer = FakeTrainer
    return fake


# Install the fake module before importing cross_encoder
_FAKE_TRANSFORMERS = _build_fake_transformers()
sys.modules.setdefault("transformers", _FAKE_TRANSFORMERS)

# Now safe to import
from src.grading.baselines.cross_encoder import (  # noqa: E402
    CrossEncoderClassifier,
    CrossEncoderRegressor,
    evaluate_cross_encoder,
)


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def _make_record(
    ref: str = "The sky is blue.",
    stu: str = "The sky appears blue.",
    label_3way: str = "correct",
    score_norm: float = 1.0,
    idx: int = 0,
) -> UnifiedRecord:
    return UnifiedRecord(
        sample_id=f"TST_{idx:04d}",
        source_dataset="scientsbank",
        original_id=str(idx),
        question_id=f"Q{idx}",
        domain="science",
        subdomain="physics",
        difficulty="easy",
        question="What colour is the sky?",
        reference_answer=ref,
        student_answer=stu,
        label_3way=label_3way,
        score_normalized=score_norm,
    )


def _make_records_cls(n: int = 6) -> list[UnifiedRecord]:
    labels = ["correct", "partially_correct", "incorrect"]
    return [
        _make_record(stu=f"Student answer {i}", label_3way=labels[i % 3], idx=i)
        for i in range(n)
    ]


def _make_records_reg(n: int = 6) -> list[UnifiedRecord]:
    return [
        _make_record(stu=f"Student answer {i}", score_norm=round(i / (n - 1), 2), idx=i)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestCrossEncoderClassifier(unittest.TestCase):

    def test_fit_runs_without_error_2way(self):
        records = _make_records_cls(4)
        for i, r in enumerate(records):
            r.label_3way = "correct" if i % 2 == 0 else "incorrect"
        clf = CrossEncoderClassifier(num_labels=2, num_epochs=1)
        clf.fit(records, "label_3way")  # should not raise

    def test_fit_runs_without_error_3way(self):
        records = _make_records_cls(6)
        clf = CrossEncoderClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")

    def test_fit_runs_without_error_5way(self):
        records = _make_records_cls(5)
        labels_5 = ["correct", "partially_correct_incomplete", "incorrect",
                    "contradictory", "irrelevant"]
        for i, r in enumerate(records):
            r.label_5way = labels_5[i % 5]
        clf = CrossEncoderClassifier(num_labels=5, num_epochs=1)
        clf.fit(records, "label_5way")

    def test_predict_returns_valid_labels(self):
        records = _make_records_cls(6)
        clf = CrossEncoderClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")
        preds = clf.predict(records)

        self.assertEqual(len(preds), len(records))
        valid_labels = {"correct", "partially_correct", "incorrect"}
        for p in preds:
            self.assertIn(p, valid_labels)

    def test_predict_proba_sums_to_one(self):
        records = _make_records_cls(6)
        clf = CrossEncoderClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")
        probas = clf.predict_proba(records)

        self.assertEqual(len(probas), len(records))
        for row in probas:
            self.assertEqual(len(row), 3)
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_invalid_num_labels_raises(self):
        with self.assertRaises(ValueError):
            CrossEncoderClassifier(num_labels=4)

    def test_predict_empty_records(self):
        clf = CrossEncoderClassifier(num_labels=3)
        self.assertEqual(clf.predict([]), [])

    def test_predict_proba_empty_records(self):
        clf = CrossEncoderClassifier(num_labels=3)
        self.assertEqual(clf.predict_proba([]), [])


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

class TestCrossEncoderRegressor(unittest.TestCase):

    def test_fit_runs_without_error(self):
        records = _make_records_reg(6)
        reg = CrossEncoderRegressor(num_epochs=1)
        reg.fit(records, "score_normalized")

    def test_predict_returns_floats(self):
        records = _make_records_reg(6)
        reg = CrossEncoderRegressor(num_epochs=1)
        reg.fit(records, "score_normalized")
        preds = reg.predict(records)

        self.assertEqual(len(preds), len(records))
        for p in preds:
            self.assertIsInstance(p, float)

    def test_predict_proba_returns_single_value_lists(self):
        records = _make_records_reg(6)
        reg = CrossEncoderRegressor(num_epochs=1)
        reg.fit(records, "score_normalized")
        probas = reg.predict_proba(records)

        self.assertEqual(len(probas), len(records))
        for row in probas:
            self.assertEqual(len(row), 1)
            self.assertIsInstance(row[0], float)

    def test_predict_empty_records(self):
        reg = CrossEncoderRegressor()
        self.assertEqual(reg.predict([]), [])


# ---------------------------------------------------------------------------
# evaluate_cross_encoder integration tests
# ---------------------------------------------------------------------------

class TestEvaluateCrossEncoder(unittest.TestCase):

    def test_evaluate_classification_returns_macro_f1_with_ci(self):
        records = _make_records_cls(6)
        clf = CrossEncoderClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")
        result = evaluate_cross_encoder(
            clf, records, "label_3way", task="classification", bootstrap_n=10
        )

        self.assertIn("macro_f1", result)
        macro_f1 = result["macro_f1"]
        self.assertIn("value", macro_f1)
        self.assertIn("ci_lower", macro_f1)
        self.assertIn("ci_upper", macro_f1)
        self.assertIsInstance(macro_f1["value"], float)

    def test_evaluate_classification_returns_accuracy_with_ci(self):
        records = _make_records_cls(6)
        clf = CrossEncoderClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")
        result = evaluate_cross_encoder(
            clf, records, "label_3way", task="classification", bootstrap_n=10
        )

        self.assertIn("accuracy", result)
        acc = result["accuracy"]
        self.assertIn("value", acc)
        self.assertIn("ci_lower", acc)
        self.assertIn("ci_upper", acc)

    def test_evaluate_regression_returns_pearson_r_with_ci(self):
        records = _make_records_reg(6)
        reg = CrossEncoderRegressor(num_epochs=1)
        reg.fit(records, "score_normalized")
        result = evaluate_cross_encoder(
            reg, records, "score_normalized", task="regression", bootstrap_n=10
        )

        self.assertIn("pearson_r", result)
        pearson = result["pearson_r"]
        self.assertIn("value", pearson)
        self.assertIn("ci_lower", pearson)
        self.assertIn("ci_upper", pearson)
        self.assertIsInstance(pearson["value"], float)

    def test_evaluate_regression_returns_spearman_rho(self):
        records = _make_records_reg(6)
        reg = CrossEncoderRegressor(num_epochs=1)
        reg.fit(records, "score_normalized")
        result = evaluate_cross_encoder(
            reg, records, "score_normalized", task="regression", bootstrap_n=10
        )
        self.assertIn("spearman_rho", result)

    def test_evaluate_invalid_task_raises(self):
        records = _make_records_cls(4)
        clf = CrossEncoderClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")
        with self.assertRaises(ValueError):
            evaluate_cross_encoder(clf, records, "label_3way", task="invalid")


if __name__ == "__main__":
    unittest.main()
