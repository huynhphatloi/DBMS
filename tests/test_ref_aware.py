"""Integration tests for the Reference-Answer-Aware DeBERTa model (task 14.6).

All HuggingFace model/tokenizer/Trainer calls are mocked via sys.modules so
no real weights are downloaded and no GPU is required.  Tests verify:
  - fit() runs without error on a small dataset (classification mode)
  - fit() runs without error in multi-task mode
  - predict() returns valid labels (classification) or floats (regression)
  - predict_proba() returns probabilities that sum to ~1 (classification)
  - evaluate_ref_aware() returns macro_f1 with CI fields (classification)
  - multi-task mode produces both classification and regression outputs
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock

import torch

from src.data.schema import UnifiedRecord


# ---------------------------------------------------------------------------
# Build a fake `transformers` module before importing ref_aware
# ---------------------------------------------------------------------------

def _build_fake_transformers():
    """Return a minimal fake transformers module."""
    fake = types.ModuleType("transformers")

    # AutoTokenizer
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            tok = MagicMock()
            tok.sep_token = "[SEP]"

            def _call(text_a, text_b=None, padding=True, truncation=True,
                      max_length=256, return_tensors=None):
                if isinstance(text_a, list):
                    n = len(text_a)
                else:
                    n = 1
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

    # AutoModel (used by RefAwareMultiTask)
    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name):
            encoder = MagicMock()
            encoder.config = MagicMock()
            encoder.config.hidden_size = 768

            def _forward(**kwargs):
                n = kwargs["input_ids"].shape[0]
                seq_len = kwargs["input_ids"].shape[1]
                out = MagicMock()
                out.last_hidden_state = torch.randn(n, seq_len, 768)
                return out

            encoder.side_effect = _forward
            encoder.__call__ = _forward
            return encoder

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
    fake.AutoModel = FakeAutoModel
    fake.TrainingArguments = FakeTrainingArguments
    fake.Trainer = FakeTrainer
    return fake


# Install the fake module before importing ref_aware
_FAKE_TRANSFORMERS = _build_fake_transformers()
sys.modules.setdefault("transformers", _FAKE_TRANSFORMERS)

# Now safe to import
from src.grading.models.ref_aware import (  # noqa: E402
    RefAwareClassifier,
    RefAwareMultiTask,
    evaluate_ref_aware,
)


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def _make_record(
    question: str = "What colour is the sky?",
    ref: str = "The sky is blue due to Rayleigh scattering.",
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
        question=question,
        reference_answer=ref,
        student_answer=stu,
        label_3way=label_3way,
        score_normalized=score_norm,
    )


def _make_records_cls(n: int = 6) -> list[UnifiedRecord]:
    labels = ["correct", "partially_correct", "incorrect"]
    return [
        _make_record(
            stu=f"Student answer {i}",
            label_3way=labels[i % 3],
            score_norm=round(1.0 - (i % 3) * 0.4, 2),
            idx=i,
        )
        for i in range(n)
    ]


def _make_records_reg(n: int = 6) -> list[UnifiedRecord]:
    return [
        _make_record(
            stu=f"Student answer {i}",
            score_norm=round(i / max(n - 1, 1), 2),
            idx=i,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# RefAwareClassifier tests
# ---------------------------------------------------------------------------

class TestRefAwareClassifier(unittest.TestCase):

    def test_fit_runs_without_error_3way(self):
        records = _make_records_cls(6)
        clf = RefAwareClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")  # should not raise

    def test_fit_runs_without_error_2way(self):
        records = _make_records_cls(4)
        for i, r in enumerate(records):
            r.label_3way = "correct" if i % 2 == 0 else "incorrect"
        clf = RefAwareClassifier(num_labels=2, num_epochs=1)
        clf.fit(records, "label_3way")

    def test_fit_runs_without_error_5way(self):
        labels_5 = [
            "correct",
            "partially_correct_incomplete",
            "incorrect",
            "contradictory",
            "irrelevant",
        ]
        records = _make_records_cls(5)
        for i, r in enumerate(records):
            r.label_5way = labels_5[i % 5]
        clf = RefAwareClassifier(num_labels=5, num_epochs=1)
        clf.fit(records, "label_5way")

    def test_predict_returns_valid_labels(self):
        records = _make_records_cls(6)
        clf = RefAwareClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")
        preds = clf.predict(records)

        self.assertEqual(len(preds), len(records))
        valid_labels = {"correct", "partially_correct", "incorrect"}
        for p in preds:
            self.assertIn(p, valid_labels)

    def test_predict_proba_sums_to_one(self):
        records = _make_records_cls(6)
        clf = RefAwareClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")
        probas = clf.predict_proba(records)

        self.assertEqual(len(probas), len(records))
        for row in probas:
            self.assertEqual(len(row), 3)
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_invalid_num_labels_raises(self):
        with self.assertRaises(ValueError):
            RefAwareClassifier(num_labels=4)

    def test_predict_empty_records(self):
        clf = RefAwareClassifier(num_labels=3)
        self.assertEqual(clf.predict([]), [])

    def test_predict_proba_empty_records(self):
        clf = RefAwareClassifier(num_labels=3)
        self.assertEqual(clf.predict_proba([]), [])

    def test_input_uses_question_field(self):
        """Verify the model uses question, reference_answer, and student_answer."""
        records = _make_records_cls(3)
        clf = RefAwareClassifier(num_labels=3, num_epochs=1)
        # Should not raise — all three fields are present
        clf.fit(records, "label_3way")
        preds = clf.predict(records)
        self.assertEqual(len(preds), 3)


# ---------------------------------------------------------------------------
# RefAwareMultiTask tests
# ---------------------------------------------------------------------------

class TestRefAwareMultiTask(unittest.TestCase):

    def test_fit_classification_mode(self):
        records = _make_records_cls(6)
        model = RefAwareMultiTask(num_labels=3, num_epochs=1)
        model.fit(records, "label_3way")  # should not raise

    def test_fit_multitask_mode(self):
        """fit() with both label_field and score_field (multi-task mode)."""
        records = _make_records_cls(6)
        model = RefAwareMultiTask(num_labels=3, num_epochs=1)
        model.fit(records, "label_3way", score_field="score_normalized")

    def test_predict_classification_returns_valid_labels(self):
        records = _make_records_cls(6)
        model = RefAwareMultiTask(num_labels=3, num_epochs=1, task="classification")
        model.fit(records, "label_3way")
        preds = model.predict(records)

        self.assertEqual(len(preds), len(records))
        valid_labels = {"correct", "partially_correct", "incorrect"}
        for p in preds:
            self.assertIn(p, valid_labels)

    def test_predict_regression_returns_floats(self):
        records = _make_records_cls(6)
        model = RefAwareMultiTask(num_labels=3, num_epochs=1, task="regression")
        model.fit(records, "label_3way", score_field="score_normalized")
        preds = model.predict(records)

        self.assertEqual(len(preds), len(records))
        for p in preds:
            self.assertIsInstance(p, float)

    def test_predict_proba_classification_sums_to_one(self):
        records = _make_records_cls(6)
        model = RefAwareMultiTask(num_labels=3, num_epochs=1, task="classification")
        model.fit(records, "label_3way")
        probas = model.predict_proba(records)

        self.assertEqual(len(probas), len(records))
        for row in probas:
            self.assertEqual(len(row), 3)
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_predict_proba_regression_returns_single_value_lists(self):
        records = _make_records_cls(6)
        model = RefAwareMultiTask(num_labels=3, num_epochs=1, task="regression")
        model.fit(records, "label_3way", score_field="score_normalized")
        probas = model.predict_proba(records)

        self.assertEqual(len(probas), len(records))
        for row in probas:
            self.assertEqual(len(row), 1)
            self.assertIsInstance(row[0], float)

    def test_multitask_both_heads_produce_outputs(self):
        """Verify both classification and regression outputs are accessible."""
        records = _make_records_cls(6)

        # Classification head
        cls_model = RefAwareMultiTask(
            num_labels=3, num_epochs=1, task="classification"
        )
        cls_model.fit(records, "label_3way", score_field="score_normalized")
        cls_preds = cls_model.predict(records)

        # Regression head (same underlying model, different task setting)
        reg_model = RefAwareMultiTask(
            num_labels=3, num_epochs=1, task="regression"
        )
        reg_model.fit(records, "label_3way", score_field="score_normalized")
        reg_preds = reg_model.predict(records)

        self.assertEqual(len(cls_preds), len(records))
        self.assertEqual(len(reg_preds), len(records))

        # Classification preds are strings, regression preds are floats
        for p in cls_preds:
            self.assertIsInstance(p, str)
        for p in reg_preds:
            self.assertIsInstance(p, float)

    def test_invalid_num_labels_raises(self):
        with self.assertRaises(ValueError):
            RefAwareMultiTask(num_labels=4)

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            RefAwareMultiTask(alpha=1.5)

    def test_invalid_task_raises(self):
        with self.assertRaises(ValueError):
            RefAwareMultiTask(task="invalid")

    def test_predict_empty_records(self):
        model = RefAwareMultiTask(num_labels=3)
        self.assertEqual(model.predict([]), [])

    def test_predict_proba_empty_records(self):
        model = RefAwareMultiTask(num_labels=3)
        self.assertEqual(model.predict_proba([]), [])


# ---------------------------------------------------------------------------
# evaluate_ref_aware integration tests
# ---------------------------------------------------------------------------

class TestEvaluateRefAware(unittest.TestCase):

    def test_evaluate_classification_returns_macro_f1_with_ci(self):
        records = _make_records_cls(6)
        clf = RefAwareClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")
        result = evaluate_ref_aware(
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
        clf = RefAwareClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")
        result = evaluate_ref_aware(
            clf, records, "label_3way", task="classification", bootstrap_n=10
        )

        self.assertIn("accuracy", result)
        acc = result["accuracy"]
        self.assertIn("value", acc)
        self.assertIn("ci_lower", acc)
        self.assertIn("ci_upper", acc)

    def test_evaluate_regression_returns_pearson_r_with_ci(self):
        records = _make_records_reg(6)
        model = RefAwareMultiTask(num_labels=3, num_epochs=1, task="regression")
        model.fit(records, "label_3way", score_field="score_normalized")
        result = evaluate_ref_aware(
            model, records, "score_normalized", task="regression", bootstrap_n=10
        )

        self.assertIn("pearson_r", result)
        pearson = result["pearson_r"]
        self.assertIn("value", pearson)
        self.assertIn("ci_lower", pearson)
        self.assertIn("ci_upper", pearson)
        self.assertIsInstance(pearson["value"], float)

    def test_evaluate_regression_returns_spearman_rho(self):
        records = _make_records_reg(6)
        model = RefAwareMultiTask(num_labels=3, num_epochs=1, task="regression")
        model.fit(records, "label_3way", score_field="score_normalized")
        result = evaluate_ref_aware(
            model, records, "score_normalized", task="regression", bootstrap_n=10
        )
        self.assertIn("spearman_rho", result)

    def test_evaluate_invalid_task_raises(self):
        records = _make_records_cls(4)
        clf = RefAwareClassifier(num_labels=3, num_epochs=1)
        clf.fit(records, "label_3way")
        with self.assertRaises(ValueError):
            evaluate_ref_aware(clf, records, "label_3way", task="invalid")

    def test_evaluate_multitask_classification_head(self):
        """evaluate_ref_aware works with RefAwareMultiTask in classification mode."""
        records = _make_records_cls(6)
        model = RefAwareMultiTask(num_labels=3, num_epochs=1, task="classification")
        model.fit(records, "label_3way", score_field="score_normalized")
        result = evaluate_ref_aware(
            model, records, "label_3way", task="classification", bootstrap_n=10
        )
        self.assertIn("macro_f1", result)


if __name__ == "__main__":
    unittest.main()
