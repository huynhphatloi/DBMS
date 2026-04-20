"""SBERT Cosine Similarity baseline grader.

Encodes reference_answer and student_answer using a configurable SBERT model
(default: all-MiniLM-L6-v2) and computes cosine similarity.

Supports two classification modes:
  - threshold: classify by comparing cosine similarity against a threshold
  - logistic_regression: use cosine similarity as a single feature in sklearn LR

Integrates with EvaluationHarness for Macro_F1, Weighted F1, Accuracy with Bootstrap_CI.

NOTE: sentence_transformers is loaded lazily inside methods that need it so the
module can be imported in environments where the library is not installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.schema import UnifiedRecord
from src.evaluation.metrics import EvaluationHarness


# ---------------------------------------------------------------------------
# GradingModel ABC (mirrors lexical.py / tfidf_ml.py)
# ---------------------------------------------------------------------------

class GradingModel(ABC):
    @abstractmethod
    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None: ...

    @abstractmethod
    def predict(self, records: Iterable[UnifiedRecord]) -> list[str | float]: ...

    @abstractmethod
    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# SBERT encoder helper
# ---------------------------------------------------------------------------

def _load_sbert(model_name: str):
    """Lazily import and return a SentenceTransformer instance.

    Raises ImportError with a helpful message if sentence_transformers is not
    installed.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "sentence_transformers is required for the SBERT baseline. "
            "Install it with: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


def cosine_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy arrays.

    Returns a value in [-1, 1]; for normalised SBERT embeddings this is [0, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def encode_pairs(
    model,
    reference_texts: list[str],
    student_texts: list[str],
) -> np.ndarray:
    """Encode all texts and return cosine similarities as a 1-D array.

    Args:
        model: A SentenceTransformer instance (or any object with an .encode method).
        reference_texts: List of reference answer strings.
        student_texts: List of student answer strings.

    Returns:
        1-D numpy array of cosine similarities, one per pair.
    """
    all_texts = reference_texts + student_texts
    all_embeddings = model.encode(all_texts, convert_to_numpy=True)
    n = len(reference_texts)
    ref_embeddings = all_embeddings[:n]
    stu_embeddings = all_embeddings[n:]

    similarities = np.array(
        [
            cosine_similarity_vectors(ref_embeddings[i], stu_embeddings[i])
            for i in range(n)
        ]
    )
    return similarities


# ---------------------------------------------------------------------------
# Threshold-based classifier
# ---------------------------------------------------------------------------

class SBERTThresholdClassifier(GradingModel):
    """Classify by comparing SBERT cosine similarity against a threshold.

    Args:
        model_name: SBERT model name (default: all-MiniLM-L6-v2).
        threshold: Similarity >= threshold → positive_label, else negative_label.
        positive_label: Label assigned when similarity >= threshold.
        negative_label: Label assigned when similarity < threshold.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.5,
        positive_label: str = "correct",
        negative_label: str = "incorrect",
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.positive_label = positive_label
        self.negative_label = negative_label
        self._model = None
        self._classes: list[str] = [negative_label, positive_label]

    def _get_model(self):
        if self._model is None:
            self._model = _load_sbert(self.model_name)
        return self._model

    def _similarities(self, records: list[UnifiedRecord]) -> np.ndarray:
        model = self._get_model()
        refs = [r.reference_answer for r in records]
        stus = [r.student_answer for r in records]
        return encode_pairs(model, refs, stus)

    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None:
        """No training needed; collects class list from data."""
        labels = set()
        for r in records:
            lbl = getattr(r, label_field, None)
            if lbl is not None:
                labels.add(str(lbl))
        if labels:
            self._classes = sorted(labels)

    def predict(self, records: Iterable[UnifiedRecord]) -> list[str]:
        recs = list(records)
        sims = self._similarities(recs)
        return [
            self.positive_label if s >= self.threshold else self.negative_label
            for s in sims
        ]

    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]:
        """Return [P(negative), P(positive)] using similarity as P(positive).

        Cosine similarity is clamped to [0, 1] before use as a probability,
        since SBERT embeddings are normalised and typical sentence pairs yield
        non-negative similarities; clamping handles edge cases gracefully.
        """
        recs = list(records)
        sims = self._similarities(recs)
        result = []
        for s in sims:
            p = float(np.clip(s, 0.0, 1.0))
            result.append([1.0 - p, p])
        return result


# ---------------------------------------------------------------------------
# Logistic Regression classifier (single feature: cosine similarity)
# ---------------------------------------------------------------------------

class SBERTLogisticRegression(GradingModel):
    """Use SBERT cosine similarity as a single feature in Logistic Regression.

    Args:
        model_name: SBERT model name (default: all-MiniLM-L6-v2).
        lr_kwargs: Keyword arguments forwarded to sklearn LogisticRegression.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        **lr_kwargs,
    ) -> None:
        self.model_name = model_name
        defaults = {"max_iter": 1000, "random_state": 42}
        defaults.update(lr_kwargs)
        self._lr = LogisticRegression(**defaults)
        self._sbert_model = None
        self._classes: list[str] = []

    def _get_model(self):
        if self._sbert_model is None:
            self._sbert_model = _load_sbert(self.model_name)
        return self._sbert_model

    def _feature_matrix(self, records: list[UnifiedRecord]) -> np.ndarray:
        model = self._get_model()
        refs = [r.reference_answer for r in records]
        stus = [r.student_answer for r in records]
        sims = encode_pairs(model, refs, stus)
        return sims.reshape(-1, 1)

    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None:
        recs = list(records)
        X = self._feature_matrix(recs)
        y = [str(getattr(r, label_field)) for r in recs]
        self._lr.fit(X, y)
        self._classes = list(self._lr.classes_)

    def predict(self, records: Iterable[UnifiedRecord]) -> list[str]:
        recs = list(records)
        X = self._feature_matrix(recs)
        return list(self._lr.predict(X))

    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]:
        recs = list(records)
        X = self._feature_matrix(recs)
        return self._lr.predict_proba(X).tolist()


# ---------------------------------------------------------------------------
# Evaluation helper — integrates with EvaluationHarness
# ---------------------------------------------------------------------------

def evaluate_sbert_model(
    model: GradingModel,
    records: Iterable[UnifiedRecord],
    label_field: str,
    bootstrap_n: int = 1000,
) -> dict:
    """Run the SBERT model on records and return EvaluationHarness classification metrics.

    Returns a dict with keys: accuracy, macro_f1, weighted_f1, per_class_f1,
    confusion_matrix — each (except confusion_matrix) has 'value', 'ci_lower',
    'ci_upper'.

    Args:
        model: A fitted GradingModel (SBERTThresholdClassifier or SBERTLogisticRegression).
        records: Iterable of UnifiedRecord instances.
        label_field: Name of the label attribute on UnifiedRecord (e.g. "label_2way").
        bootstrap_n: Number of bootstrap iterations for CI computation.
    """
    recs = list(records)
    y_true = [str(getattr(r, label_field)) for r in recs]
    y_pred = model.predict(recs)
    harness = EvaluationHarness()
    return harness.classification_metrics(y_true, y_pred, bootstrap_n=bootstrap_n)
