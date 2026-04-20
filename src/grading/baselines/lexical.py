"""Lexical Overlap baseline grader.

Computes BLEU-1, BLEU-4, ROUGE-L, Jaccard similarity, and word overlap ratio
between reference_answer and student_answer.

Supports two classification modes:
  - threshold: classify by comparing a single metric against a threshold
  - logistic_regression: use all 5 metrics as features in sklearn LogisticRegression

Both BLEU and ROUGE-L are implemented without external NLP libraries so the
module works in environments where nltk / rouge_score are unavailable.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.schema import UnifiedRecord
from src.evaluation.metrics import EvaluationHarness


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Individual metric functions (pure, no side-effects)
# ---------------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_n(reference: str, hypothesis: str, n: int) -> float:
    """Compute BLEU-n (precision of n-grams with brevity penalty).

    Returns a value in [0.0, 1.0].
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = math.exp(min(0.0, 1.0 - len(ref_tokens) / len(hyp_tokens))) if ref_tokens else 0.0

    if len(hyp_tokens) < n:
        return 0.0

    ref_ngrams = _ngrams(ref_tokens, n)
    hyp_ngrams = _ngrams(hyp_tokens, n)

    if not hyp_ngrams:
        return 0.0

    clipped = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
    precision = clipped / sum(hyp_ngrams.values())

    return float(bp * precision)


def rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 using the Longest Common Subsequence.

    Returns a value in [0.0, 1.0].
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(ref_tokens), len(hyp_tokens)
    # Use 1-D DP to save memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    lcs_len = prev[n]
    precision = lcs_len / n
    recall = lcs_len / m
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def jaccard_similarity(reference: str, hypothesis: str) -> float:
    """Compute Jaccard similarity: |intersection| / |union| of word sets."""
    ref_set = set(_tokenize(reference))
    hyp_set = set(_tokenize(hypothesis))
    if not ref_set and not hyp_set:
        return 1.0
    union = ref_set | hyp_set
    if not union:
        return 0.0
    return float(len(ref_set & hyp_set) / len(union))


def word_overlap_ratio(reference: str, hypothesis: str) -> float:
    """Compute word overlap ratio: |intersection| / |reference_words|."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens:
        return 0.0
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    return float(len(ref_set & hyp_set) / len(ref_set))


def compute_lexical_features(reference: str, hypothesis: str) -> dict[str, float]:
    """Return all five lexical metrics as a dict."""
    return {
        "bleu_1": bleu_n(reference, hypothesis, 1),
        "bleu_4": bleu_n(reference, hypothesis, 4),
        "rouge_l": rouge_l(reference, hypothesis),
        "jaccard": jaccard_similarity(reference, hypothesis),
        "word_overlap": word_overlap_ratio(reference, hypothesis),
    }


# ---------------------------------------------------------------------------
# GradingModel ABC (mirrors the design doc interface)
# ---------------------------------------------------------------------------

class GradingModel(ABC):
    @abstractmethod
    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None: ...

    @abstractmethod
    def predict(self, records: Iterable[UnifiedRecord]) -> list[str | float]: ...

    @abstractmethod
    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Threshold-based classifier (task 10.2)
# ---------------------------------------------------------------------------

class LexicalThresholdClassifier(GradingModel):
    """Classify by comparing a single lexical metric against a threshold.

    Args:
        metric: One of "bleu_1", "bleu_4", "rouge_l", "jaccard", "word_overlap".
        threshold: Score >= threshold → positive_label, else negative_label.
        positive_label: Label assigned when metric >= threshold.
        negative_label: Label assigned when metric < threshold.
    """

    VALID_METRICS = frozenset({"bleu_1", "bleu_4", "rouge_l", "jaccard", "word_overlap"})

    def __init__(
        self,
        metric: str = "rouge_l",
        threshold: float = 0.5,
        positive_label: str = "correct",
        negative_label: str = "incorrect",
    ) -> None:
        if metric not in self.VALID_METRICS:
            raise ValueError(f"metric must be one of {sorted(self.VALID_METRICS)}, got {metric!r}")
        self.metric = metric
        self.threshold = threshold
        self.positive_label = positive_label
        self.negative_label = negative_label
        self._classes: list[str] = [negative_label, positive_label]

    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None:
        """No training needed for threshold classifier; collects class list."""
        labels = set()
        for r in records:
            lbl = getattr(r, label_field, None)
            if lbl is not None:
                labels.add(str(lbl))
        if labels:
            self._classes = sorted(labels)

    def _score(self, record: UnifiedRecord) -> float:
        feats = compute_lexical_features(record.reference_answer, record.student_answer)
        return feats[self.metric]

    def predict(self, records: Iterable[UnifiedRecord]) -> list[str]:
        return [
            self.positive_label if self._score(r) >= self.threshold else self.negative_label
            for r in records
        ]

    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]:
        """Return [P(negative), P(positive)] using the raw metric score as P(positive)."""
        result = []
        for r in records:
            p = self._score(r)
            result.append([1.0 - p, p])
        return result


# ---------------------------------------------------------------------------
# Logistic Regression classifier (task 10.3)
# ---------------------------------------------------------------------------

class LexicalLogisticRegression(GradingModel):
    """Use all 5 lexical metrics as features in a Logistic Regression classifier.

    Args:
        lr_kwargs: Keyword arguments forwarded to sklearn LogisticRegression.
    """

    FEATURE_ORDER = ["bleu_1", "bleu_4", "rouge_l", "jaccard", "word_overlap"]

    def __init__(self, **lr_kwargs) -> None:
        defaults = {"max_iter": 1000, "random_state": 42}
        defaults.update(lr_kwargs)
        self._lr = LogisticRegression(**defaults)
        self._classes: list[str] = []

    def _feature_matrix(self, records: list[UnifiedRecord]) -> np.ndarray:
        rows = []
        for r in records:
            feats = compute_lexical_features(r.reference_answer, r.student_answer)
            rows.append([feats[k] for k in self.FEATURE_ORDER])
        return np.array(rows, dtype=float)

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
# Evaluation helper (task 10.4)
# ---------------------------------------------------------------------------

def evaluate_lexical_model(
    model: GradingModel,
    records: Iterable[UnifiedRecord],
    label_field: str,
    bootstrap_n: int = 1000,
) -> dict:
    """Run the model on records and return EvaluationHarness classification metrics.

    Returns a dict with keys: accuracy, macro_f1, weighted_f1, per_class_f1,
    confusion_matrix — each (except confusion_matrix) has 'value', 'ci_lower',
    'ci_upper'.
    """
    recs = list(records)
    y_true = [str(getattr(r, label_field)) for r in recs]
    y_pred = model.predict(recs)
    harness = EvaluationHarness()
    return harness.classification_metrics(y_true, y_pred, bootstrap_n=bootstrap_n)
