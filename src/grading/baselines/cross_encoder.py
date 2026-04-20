"""Cross-Encoder fine-tuning baseline grader.

Input format: [CLS] reference_answer [SEP] student_answer [SEP]
Fine-tunes `cross-encoder/stsb-roberta-base` (default).

Supports:
  - 2-way, 3-way, 5-way classification heads
  - Regression head (num_labels=1)

Reports Macro_F1 with Bootstrap_CI for classification,
Pearson r with Bootstrap_CI for regression.

NOTE: transformers is loaded lazily inside methods to avoid import errors
when the library is not installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from src.data.schema import UnifiedRecord
from src.evaluation.metrics import EvaluationHarness


# ---------------------------------------------------------------------------
# GradingModel ABC (mirrors other baselines)
# ---------------------------------------------------------------------------

class GradingModel(ABC):
    @abstractmethod
    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None: ...

    @abstractmethod
    def predict(self, records: Iterable[UnifiedRecord]) -> list[str | float]: ...

    @abstractmethod
    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tokenizer(model_name: str):
    """Lazily import and return an AutoTokenizer instance."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "transformers is required for the CrossEncoder baseline. "
            "Install it with: pip install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(model_name)


def _load_model_for_classification(model_name: str, num_labels: int):
    """Lazily import and return AutoModelForSequenceClassification."""
    try:
        from transformers import AutoModelForSequenceClassification  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "transformers is required for the CrossEncoder baseline. "
            "Install it with: pip install transformers"
        ) from exc
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


def _build_input_pairs(records: list[UnifiedRecord]) -> tuple[list[str], list[str]]:
    """Return (text_a, text_b) lists for tokenizer pair encoding.

    The tokenizer will produce [CLS] text_a [SEP] text_b [SEP] automatically.
    """
    text_a = [r.reference_answer for r in records]
    text_b = [r.student_answer for r in records]
    return text_a, text_b


# ---------------------------------------------------------------------------
# CrossEncoderClassifier — 2-way, 3-way, or 5-way
# ---------------------------------------------------------------------------

class CrossEncoderClassifier(GradingModel):
    """Fine-tune a cross-encoder for classification.

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of output classes — 2, 3, or 5.
        max_length: Maximum tokenizer sequence length.
        batch_size: Training and inference batch size.
        num_epochs: Number of fine-tuning epochs.
        learning_rate: AdamW learning rate.
        seed: Random seed for reproducibility.
    """

    VALID_NUM_LABELS = frozenset({2, 3, 5})

    def __init__(
        self,
        model_name: str = "cross-encoder/stsb-roberta-base",
        num_labels: int = 3,
        max_length: int = 256,
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        seed: int = 42,
    ) -> None:
        if num_labels not in self.VALID_NUM_LABELS:
            raise ValueError(
                f"num_labels must be one of {sorted(self.VALID_NUM_LABELS)}, got {num_labels}"
            )
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.seed = seed

        self._tokenizer = None
        self._model = None
        self._label2id: dict[str, int] = {}
        self._id2label: dict[int, str] = {}

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.model_name)
        return self._tokenizer

    def _get_model(self):
        if self._model is None:
            self._model = _load_model_for_classification(self.model_name, self.num_labels)
        return self._model

    def _tokenize_batch(self, records: list[UnifiedRecord]):
        """Tokenise records and return a dict of tensors."""
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("torch is required for CrossEncoderClassifier") from exc

        tokenizer = self._get_tokenizer()
        text_a, text_b = _build_input_pairs(records)
        encoding = tokenizer(
            text_a,
            text_b,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoding

    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None:
        """Fine-tune the cross-encoder on the provided records."""
        try:
            import torch  # noqa: PLC0415
            from torch.utils.data import Dataset as TorchDataset  # noqa: PLC0415
            from transformers import TrainingArguments, Trainer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for CrossEncoderClassifier.fit()"
            ) from exc

        recs = list(records)
        if not recs:
            return

        # Build label mapping from string labels to integer indices
        unique_labels = sorted(set(str(getattr(r, label_field)) for r in recs))
        self._label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        self._id2label = {idx: lbl for lbl, idx in self._label2id.items()}

        tokenizer = self._get_tokenizer()
        text_a, text_b = _build_input_pairs(recs)
        encodings = tokenizer(
            text_a,
            text_b,
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        int_labels = [self._label2id[str(getattr(r, label_field))] for r in recs]

        class _Dataset(TorchDataset):
            def __init__(self, enc, lbls):
                self.enc = enc
                self.lbls = lbls

            def __len__(self):
                return len(self.lbls)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.lbls[idx])
                return item

        dataset = _Dataset(encodings, int_labels)
        model = self._get_model()

        training_args = TrainingArguments(
            output_dir="models/cross_encoder_cls",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            seed=self.seed,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            no_cuda=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        self._model = model

    def predict(self, records: Iterable[UnifiedRecord]) -> list[str]:
        """Return predicted string labels."""
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("torch is required for CrossEncoderClassifier.predict()") from exc

        recs = list(records)
        if not recs:
            return []

        model = self._get_model()
        model.eval()
        encoding = self._tokenize_batch(recs)

        with torch.no_grad():
            outputs = model(**encoding)

        pred_ids = outputs.logits.argmax(dim=-1).tolist()

        if self._id2label:
            return [self._id2label[i] for i in pred_ids]
        return [str(i) for i in pred_ids]

    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]:
        """Return softmax probabilities for each class."""
        try:
            import torch  # noqa: PLC0415
            import torch.nn.functional as F  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("torch is required for CrossEncoderClassifier.predict_proba()") from exc

        recs = list(records)
        if not recs:
            return []

        model = self._get_model()
        model.eval()
        encoding = self._tokenize_batch(recs)

        with torch.no_grad():
            outputs = model(**encoding)

        probs = F.softmax(outputs.logits, dim=-1)
        return probs.tolist()


# ---------------------------------------------------------------------------
# CrossEncoderRegressor — continuous score prediction (num_labels=1)
# ---------------------------------------------------------------------------

class CrossEncoderRegressor(GradingModel):
    """Fine-tune a cross-encoder for regression.

    Uses num_labels=1 which makes HuggingFace use MSE loss automatically.

    Args:
        model_name: HuggingFace model identifier.
        max_length: Maximum tokenizer sequence length.
        batch_size: Training and inference batch size.
        num_epochs: Number of fine-tuning epochs.
        learning_rate: AdamW learning rate.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/stsb-roberta-base",
        max_length: int = 256,
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.seed = seed

        self._tokenizer = None
        self._model = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.model_name)
        return self._tokenizer

    def _get_model(self):
        if self._model is None:
            # num_labels=1 → regression head with MSE loss
            self._model = _load_model_for_classification(self.model_name, num_labels=1)
        return self._model

    def _tokenize_batch(self, records: list[UnifiedRecord]):
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("torch is required for CrossEncoderRegressor") from exc

        tokenizer = self._get_tokenizer()
        text_a, text_b = _build_input_pairs(records)
        encoding = tokenizer(
            text_a,
            text_b,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoding

    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None:
        """Fine-tune the cross-encoder for regression on the provided records."""
        try:
            import torch  # noqa: PLC0415
            from torch.utils.data import Dataset as TorchDataset  # noqa: PLC0415
            from transformers import TrainingArguments, Trainer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for CrossEncoderRegressor.fit()"
            ) from exc

        recs = list(records)
        if not recs:
            return

        tokenizer = self._get_tokenizer()
        text_a, text_b = _build_input_pairs(recs)
        encodings = tokenizer(
            text_a,
            text_b,
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        float_labels = [float(getattr(recs[i], label_field)) for i in range(len(recs))]

        class _Dataset(TorchDataset):
            def __init__(self, enc, lbls):
                self.enc = enc
                self.lbls = lbls

            def __len__(self):
                return len(self.lbls)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
                # HuggingFace expects float labels for regression (num_labels=1)
                item["labels"] = torch.tensor(self.lbls[idx], dtype=torch.float)
                return item

        dataset = _Dataset(encodings, float_labels)
        model = self._get_model()

        training_args = TrainingArguments(
            output_dir="models/cross_encoder_reg",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            seed=self.seed,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            no_cuda=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        self._model = model

    def predict(self, records: Iterable[UnifiedRecord]) -> list[float]:
        """Return predicted continuous scores."""
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("torch is required for CrossEncoderRegressor.predict()") from exc

        recs = list(records)
        if not recs:
            return []

        model = self._get_model()
        model.eval()
        encoding = self._tokenize_batch(recs)

        with torch.no_grad():
            outputs = model(**encoding)

        # logits shape: (N, 1) → squeeze to (N,)
        scores = outputs.logits.squeeze(-1).tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores

    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]:
        """Return [[score]] for each record (single-value list for regression).

        For regression there is no probability distribution; we return the raw
        predicted score wrapped in a list for interface compatibility.
        """
        scores = self.predict(records)
        return [[s] for s in scores]


# ---------------------------------------------------------------------------
# Evaluation helpers — integrate with EvaluationHarness (task 13.4)
# ---------------------------------------------------------------------------

def evaluate_cross_encoder(
    model: GradingModel,
    records: Iterable[UnifiedRecord],
    label_field: str,
    task: str = "classification",
    bootstrap_n: int = 1000,
) -> dict:
    """Run the cross-encoder model on records and return EvaluationHarness metrics.

    Args:
        model: A fitted CrossEncoderClassifier or CrossEncoderRegressor.
        records: Iterable of UnifiedRecord instances.
        label_field: Name of the label attribute on UnifiedRecord.
        task: "classification" or "regression".
        bootstrap_n: Number of bootstrap iterations for CI computation.

    Returns:
        For classification: dict with macro_f1, accuracy, weighted_f1, per_class_f1,
            confusion_matrix — each metric (except confusion_matrix) has
            'value', 'ci_lower', 'ci_upper'.
        For regression: dict with pearson_r, spearman_rho, rmse, mae, qwk —
            each has 'value', 'ci_lower', 'ci_upper'.
    """
    recs = list(records)
    harness = EvaluationHarness()

    if task == "classification":
        y_true = [str(getattr(r, label_field)) for r in recs]
        y_pred = model.predict(recs)
        return harness.classification_metrics(y_true, y_pred, bootstrap_n=bootstrap_n)

    elif task == "regression":
        y_true = [float(getattr(r, label_field)) for r in recs]
        y_pred = [float(p) for p in model.predict(recs)]
        return harness.regression_metrics(y_true, y_pred, bootstrap_n=bootstrap_n)

    else:
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
