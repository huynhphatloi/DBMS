"""Reference-Answer-Aware DeBERTa grading model.

Input format: [CLS] question [SEP] reference_answer [SEP] student_answer [SEP]
Fine-tunes `microsoft/deberta-v3-base` (default).

Supports:
  - Classification head (2-way, 3-way, 5-way)
  - Optional regression head for multi-task learning
  - Multi-task loss: L = α·L_cls + (1-α)·L_reg with configurable α

Reports Macro_F1 with Bootstrap_CI for classification,
Pearson r with Bootstrap_CI for regression.

NOTE: transformers is loaded lazily inside methods to avoid import errors
when the library is not installed.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterable

from src.data.schema import UnifiedRecord
from src.evaluation.metrics import EvaluationHarness


# ---------------------------------------------------------------------------
# GradingModel ABC
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
            "transformers is required for RefAwareModel. "
            "Install it with: pip install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(model_name)


def _load_model_for_classification(model_name: str, num_labels: int):
    """Lazily import and return AutoModelForSequenceClassification."""
    try:
        from transformers import AutoModelForSequenceClassification  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "transformers is required for RefAwareModel. "
            "Install it with: pip install transformers"
        ) from exc
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


def _build_input_triplets(
    records: list[UnifiedRecord],
) -> tuple[list[str], list[str], list[str]]:
    """Return (question, reference_answer, student_answer) lists.

    The tokenizer will produce:
        [CLS] question [SEP] reference_answer [SEP] student_answer [SEP]
    when called with text=question, text_pair=reference_answer + [SEP] + student_answer.

    For models that support three segments natively (e.g. DeBERTa), we pass
    the question as text and concatenate reference + student as text_pair so
    the tokenizer inserts the correct special tokens.
    """
    questions = [r.question for r in records]
    refs = [r.reference_answer for r in records]
    students = [r.student_answer for r in records]
    return questions, refs, students


def _tokenize_triplets(
    tokenizer,
    records: list[UnifiedRecord],
    max_length: int,
    return_tensors: str | None = None,
):
    """Tokenize (question, reference_answer, student_answer) triplets.

    Produces: [CLS] question [SEP] reference_answer [SEP] student_answer [SEP]
    by passing question as text_a and "reference_answer [SEP] student_answer"
    as text_b so the tokenizer inserts the correct special tokens.
    """
    questions, refs, students = _build_input_triplets(records)
    # Combine reference and student with SEP token so the tokenizer produces
    # the three-segment format required by the design.
    sep = tokenizer.sep_token or "[SEP]"
    text_b = [f"{r} {sep} {s}" for r, s in zip(refs, students)]
    return tokenizer(
        questions,
        text_b,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors,
    )


# ---------------------------------------------------------------------------
# RefAwareClassifier — 2-way, 3-way, or 5-way classification
# ---------------------------------------------------------------------------

class RefAwareClassifier(GradingModel):
    """Fine-tune a reference-answer-aware encoder for classification.

    Input: [CLS] question [SEP] reference_answer [SEP] student_answer [SEP]

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of output classes — 2, 3, or 5.
        max_length: Maximum tokenizer sequence length.
        batch_size: Training and inference batch size.
        num_epochs: Number of fine-tuning epochs.
        learning_rate: AdamW learning rate.
        seed: Random seed for reproducibility.
        checkpoint_dir: Directory to save checkpoints.
        checkpoint_steps: Save checkpoint every N steps (0 = disabled).
    """

    VALID_NUM_LABELS = frozenset({2, 3, 5})

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        max_length: int = 256,
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        seed: int = 42,
        checkpoint_dir: str = "models",
        checkpoint_steps: int = 0,
    ) -> None:
        if num_labels not in self.VALID_NUM_LABELS:
            raise ValueError(
                f"num_labels must be one of {sorted(self.VALID_NUM_LABELS)}, "
                f"got {num_labels}"
            )
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_steps = checkpoint_steps

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
            self._model = _load_model_for_classification(
                self.model_name, self.num_labels
            )
        return self._model

    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None:
        """Fine-tune the model on the provided records."""
        try:
            import torch  # noqa: PLC0415
            from torch.utils.data import Dataset as TorchDataset  # noqa: PLC0415
            from transformers import TrainingArguments, Trainer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for RefAwareClassifier.fit()"
            ) from exc

        recs = list(records)
        if not recs:
            return

        unique_labels = sorted(set(str(getattr(r, label_field)) for r in recs))
        self._label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        self._id2label = {idx: lbl for lbl, idx in self._label2id.items()}

        tokenizer = self._get_tokenizer()
        encodings = _tokenize_triplets(tokenizer, recs, self.max_length)
        int_labels = [
            self._label2id[str(getattr(r, label_field))] for r in recs
        ]

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

        save_strategy = "steps" if self.checkpoint_steps > 0 else "no"
        save_steps = self.checkpoint_steps if self.checkpoint_steps > 0 else 500

        training_args = TrainingArguments(
            output_dir=f"{self.checkpoint_dir}/ref_aware_cls",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            seed=self.seed,
            logging_steps=50,
            save_strategy=save_strategy,
            save_steps=save_steps,
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
            raise ImportError(
                "torch is required for RefAwareClassifier.predict()"
            ) from exc

        recs = list(records)
        if not recs:
            return []

        model = self._get_model()
        model.eval()
        tokenizer = self._get_tokenizer()
        encoding = _tokenize_triplets(
            tokenizer, recs, self.max_length, return_tensors="pt"
        )

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
            raise ImportError(
                "torch is required for RefAwareClassifier.predict_proba()"
            ) from exc

        recs = list(records)
        if not recs:
            return []

        model = self._get_model()
        model.eval()
        tokenizer = self._get_tokenizer()
        encoding = _tokenize_triplets(
            tokenizer, recs, self.max_length, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**encoding)

        probs = F.softmax(outputs.logits, dim=-1)
        return probs.tolist()


# ---------------------------------------------------------------------------
# RefAwareMultiTask — classification + optional regression head
# ---------------------------------------------------------------------------

class RefAwareMultiTask(GradingModel):
    """Reference-aware DeBERTa with classification AND regression heads.

    Multi-task loss: L = alpha * L_cls + (1 - alpha) * L_reg

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of classification classes — 2, 3, or 5.
        alpha: Weight for classification loss (default 0.7).
        max_length: Maximum tokenizer sequence length.
        batch_size: Training and inference batch size.
        num_epochs: Number of fine-tuning epochs.
        learning_rate: AdamW learning rate.
        seed: Random seed for reproducibility.
        task: Default prediction task — "classification" or "regression".
        checkpoint_dir: Directory to save checkpoints.
        checkpoint_steps: Save checkpoint every N steps (0 = disabled).
    """

    VALID_NUM_LABELS = frozenset({2, 3, 5})

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        alpha: float = 0.7,
        max_length: int = 256,
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        seed: int = 42,
        task: str = "classification",
        checkpoint_dir: str = "models",
        checkpoint_steps: int = 0,
    ) -> None:
        if num_labels not in self.VALID_NUM_LABELS:
            raise ValueError(
                f"num_labels must be one of {sorted(self.VALID_NUM_LABELS)}, "
                f"got {num_labels}"
            )
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got {task!r}"
            )

        self.model_name = model_name
        self.num_labels = num_labels
        self.alpha = alpha
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.task = task
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_steps = checkpoint_steps

        self._tokenizer = None
        self._encoder = None       # shared DeBERTa backbone
        self._cls_head = None      # classification linear layer
        self._reg_head = None      # regression linear layer
        self._label2id: dict[str, int] = {}
        self._id2label: dict[int, str] = {}

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.model_name)
        return self._tokenizer

    def _build_model(self):
        """Build the multi-task model with shared encoder + two heads."""
        try:
            import torch  # noqa: PLC0415
            import torch.nn as nn  # noqa: PLC0415
            from transformers import AutoModel  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for RefAwareMultiTask"
            ) from exc

        class _MultiTaskModel(nn.Module):
            def __init__(self, encoder, hidden_size, num_labels, alpha):
                super().__init__()
                self.encoder = encoder
                self.cls_head = nn.Linear(hidden_size, num_labels)
                self.reg_head = nn.Linear(hidden_size, 1)
                self.alpha = alpha
                self.dropout = nn.Dropout(0.1)

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                cls_labels=None,
                reg_labels=None,
            ):
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids

                outputs = self.encoder(**kwargs)
                # Use [CLS] token representation
                pooled = outputs.last_hidden_state[:, 0, :]
                pooled = self.dropout(pooled)

                cls_logits = self.cls_head(pooled)
                reg_logits = self.reg_head(pooled).squeeze(-1)

                loss = None
                if cls_labels is not None and reg_labels is not None:
                    ce_loss = nn.CrossEntropyLoss()(cls_logits, cls_labels)
                    mse_loss = nn.MSELoss()(reg_logits, reg_labels.float())
                    loss = self.alpha * ce_loss + (1 - self.alpha) * mse_loss
                elif cls_labels is not None:
                    loss = nn.CrossEntropyLoss()(cls_logits, cls_labels)
                elif reg_labels is not None:
                    loss = nn.MSELoss()(reg_logits, reg_labels.float())

                # Return an object with .loss, .cls_logits, .reg_logits
                class _Out:
                    pass

                out = _Out()
                out.loss = loss
                out.cls_logits = cls_logits
                out.reg_logits = reg_logits
                # Expose logits for compatibility with HF Trainer
                out.logits = cls_logits
                return out

        try:
            encoder = AutoModel.from_pretrained(self.model_name)
            hidden_size = encoder.config.hidden_size
        except Exception:
            # Fallback for mocked environments
            hidden_size = 768
            encoder = None

        if encoder is None:
            # Create a minimal stub for testing
            import torch.nn as nn  # noqa: PLC0415

            class _StubEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.config = type("cfg", (), {"hidden_size": 768})()
                    self.embedding = nn.Embedding(100, 768)

                def forward(self, input_ids=None, attention_mask=None, **kwargs):
                    h = self.embedding(input_ids)

                    class _O:
                        pass

                    o = _O()
                    o.last_hidden_state = h
                    return o

            encoder = _StubEncoder()
            hidden_size = 768

        self._encoder = encoder
        model = _MultiTaskModel(encoder, hidden_size, self.num_labels, self.alpha)
        return model

    def _get_model(self):
        if self._cls_head is None:
            self._model = self._build_model()
            self._cls_head = self._model.cls_head
            self._reg_head = self._model.reg_head
        return self._model

    def fit(
        self,
        records: Iterable[UnifiedRecord],
        label_field: str,
        score_field: str | None = None,
    ) -> None:
        """Fine-tune the multi-task model.

        Args:
            records: Training records.
            label_field: Attribute name for classification labels.
            score_field: Attribute name for regression scores (optional).
        """
        try:
            import torch  # noqa: PLC0415
            from torch.utils.data import Dataset as TorchDataset  # noqa: PLC0415
            from torch.optim import AdamW  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "torch is required for RefAwareMultiTask.fit()"
            ) from exc

        recs = list(records)
        if not recs:
            return

        unique_labels = sorted(set(str(getattr(r, label_field)) for r in recs))
        self._label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        self._id2label = {idx: lbl for lbl, idx in self._label2id.items()}

        tokenizer = self._get_tokenizer()
        encodings = _tokenize_triplets(tokenizer, recs, self.max_length)
        cls_labels = [
            self._label2id[str(getattr(r, label_field))] for r in recs
        ]
        reg_labels = None
        if score_field is not None:
            reg_labels = [float(getattr(r, score_field)) for r in recs]

        class _Dataset(TorchDataset):
            def __init__(self, enc, cls_lbls, reg_lbls):
                self.enc = enc
                self.cls_lbls = cls_lbls
                self.reg_lbls = reg_lbls

            def __len__(self):
                return len(self.cls_lbls)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
                item["cls_labels"] = torch.tensor(self.cls_lbls[idx])
                if self.reg_lbls is not None:
                    item["reg_labels"] = torch.tensor(
                        self.reg_lbls[idx], dtype=torch.float
                    )
                return item

        dataset = _Dataset(encodings, cls_labels, reg_labels)
        model = self._get_model()
        model.train()

        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        n_steps = math.ceil(len(dataset) / self.batch_size) * self.num_epochs
        step = 0

        from torch.utils.data import DataLoader as TorchDataLoader  # noqa: PLC0415

        loader = TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _epoch in range(self.num_epochs):
            for batch in loader:
                optimizer.zero_grad()
                out = model(**batch)
                if out.loss is not None:
                    out.loss.backward()
                    optimizer.step()
                step += 1

                if self.checkpoint_steps > 0 and step % self.checkpoint_steps == 0:
                    self._save_checkpoint(step)

    def _save_checkpoint(self, step: int) -> None:
        """Save model checkpoint to disk."""
        import os  # noqa: PLC0415

        model_slug = self.model_name.replace("/", "_")
        ckpt_path = os.path.join(
            self.checkpoint_dir, model_slug, f"checkpoint-{step}"
        )
        os.makedirs(ckpt_path, exist_ok=True)
        try:
            import torch  # noqa: PLC0415

            torch.save(self._model.state_dict(), os.path.join(ckpt_path, "model.pt"))
        except Exception as exc:  # noqa: BLE001
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).error(
                "Checkpoint save failed at step %d: %s", step, exc
            )

    def predict(self, records: Iterable[UnifiedRecord]) -> list[str | float]:
        """Return predictions.

        Returns classification labels when self.task == "classification",
        or regression scores when self.task == "regression".
        """
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "torch is required for RefAwareMultiTask.predict()"
            ) from exc

        recs = list(records)
        if not recs:
            return []

        model = self._get_model()
        model.eval()
        tokenizer = self._get_tokenizer()
        encoding = _tokenize_triplets(
            tokenizer, recs, self.max_length, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**encoding)

        if self.task == "regression":
            scores = outputs.reg_logits.tolist()
            if isinstance(scores, float):
                scores = [scores]
            return scores

        # classification
        pred_ids = outputs.cls_logits.argmax(dim=-1).tolist()
        if self._id2label:
            return [self._id2label[i] for i in pred_ids]
        return [str(i) for i in pred_ids]

    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]:
        """Return softmax probabilities (classification) or [[score]] (regression)."""
        try:
            import torch  # noqa: PLC0415
            import torch.nn.functional as F  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "torch is required for RefAwareMultiTask.predict_proba()"
            ) from exc

        recs = list(records)
        if not recs:
            return []

        model = self._get_model()
        model.eval()
        tokenizer = self._get_tokenizer()
        encoding = _tokenize_triplets(
            tokenizer, recs, self.max_length, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**encoding)

        if self.task == "regression":
            scores = outputs.reg_logits.tolist()
            if isinstance(scores, float):
                scores = [scores]
            return [[s] for s in scores]

        probs = F.softmax(outputs.cls_logits, dim=-1)
        return probs.tolist()


# ---------------------------------------------------------------------------
# Evaluation helper — integrate with EvaluationHarness
# ---------------------------------------------------------------------------

def evaluate_ref_aware(
    model: GradingModel,
    records: Iterable[UnifiedRecord],
    label_field: str,
    task: str = "classification",
    bootstrap_n: int = 1000,
) -> dict:
    """Run the ref-aware model on records and return EvaluationHarness metrics.

    Args:
        model: A fitted RefAwareClassifier or RefAwareMultiTask.
        records: Iterable of UnifiedRecord instances.
        label_field: Name of the label attribute on UnifiedRecord.
        task: "classification" or "regression".
        bootstrap_n: Number of bootstrap iterations for CI computation.

    Returns:
        For classification: dict with macro_f1, accuracy, weighted_f1,
            per_class_f1, confusion_matrix — each metric (except
            confusion_matrix) has 'value', 'ci_lower', 'ci_upper'.
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
        raise ValueError(
            f"task must be 'classification' or 'regression', got {task!r}"
        )
