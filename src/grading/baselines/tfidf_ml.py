"""TF-IDF + Traditional ML baseline grader.

Features:
  - TF-IDF of student_answer
  - TF-IDF of reference_answer
  - Cosine similarity between student and reference TF-IDF vectors (scalar)
  - Element-wise absolute difference between student and reference TF-IDF vectors

Classifiers: Logistic Regression, SVM (linear + RBF), Random Forest, Gradient Boosting
Regression: SVR, Ridge

Both modes integrate with EvaluationHarness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from scipy.sparse import hstack, issparse
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC, SVR

from src.data.schema import UnifiedRecord
from src.evaluation.metrics import EvaluationHarness


# ---------------------------------------------------------------------------
# GradingModel ABC (mirrors lexical.py)
# ---------------------------------------------------------------------------

class GradingModel(ABC):
    @abstractmethod
    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None: ...

    @abstractmethod
    def predict(self, records: Iterable[UnifiedRecord]) -> list[str | float]: ...

    @abstractmethod
    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# TF-IDF feature builder
# ---------------------------------------------------------------------------

class _TfidfFeatureBuilder:
    """Fits a shared TF-IDF vocabulary on all texts and builds feature matrices."""

    def __init__(self, max_features: int = 5000) -> None:
        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    def fit(self, student_texts: list[str], reference_texts: list[str]) -> None:
        all_texts = student_texts + reference_texts
        self._vectorizer.fit(all_texts)
        self._fitted = True

    def transform(
        self, student_texts: list[str], reference_texts: list[str]
    ) -> np.ndarray:
        """Return dense feature matrix: [tfidf_student | tfidf_ref | cos_sim | elem_diff]."""
        stu_mat = self._vectorizer.transform(student_texts).toarray()
        ref_mat = self._vectorizer.transform(reference_texts).toarray()

        # Cosine similarity: one scalar per sample
        cos_sims = np.array(
            [
                float(cosine_similarity(stu_mat[i : i + 1], ref_mat[i : i + 1])[0, 0])
                for i in range(len(student_texts))
            ]
        ).reshape(-1, 1)

        # Element-wise absolute difference
        elem_diff = np.abs(stu_mat - ref_mat)

        return np.hstack([stu_mat, ref_mat, cos_sims, elem_diff])


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

_CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "svm_linear": lambda: SVC(kernel="linear", probability=True, random_state=42),
    "svm_rbf": lambda: SVC(kernel="rbf", probability=True, random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
}

_REGRESSORS = {
    "svr": lambda: SVR(kernel="rbf"),
    "ridge": lambda: Ridge(alpha=1.0),
}


class TfidfMLClassifier(GradingModel):
    """TF-IDF + traditional ML classifier.

    Args:
        classifier: One of "logistic_regression", "svm_linear", "svm_rbf",
                    "random_forest", "gradient_boosting".
        max_features: Maximum number of TF-IDF features per text side.
    """

    VALID_CLASSIFIERS = frozenset(_CLASSIFIERS)

    def __init__(
        self,
        classifier: str = "logistic_regression",
        max_features: int = 5000,
    ) -> None:
        if classifier not in self.VALID_CLASSIFIERS:
            raise ValueError(
                f"classifier must be one of {sorted(self.VALID_CLASSIFIERS)}, got {classifier!r}"
            )
        self._clf_name = classifier
        self._clf = _CLASSIFIERS[classifier]()
        self._feature_builder = _TfidfFeatureBuilder(max_features=max_features)
        self._classes: list[str] = []

    def _build_features(self, records: list[UnifiedRecord]) -> np.ndarray:
        student_texts = [r.student_answer for r in records]
        reference_texts = [r.reference_answer for r in records]
        return self._feature_builder.transform(student_texts, reference_texts)

    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None:
        recs = list(records)
        student_texts = [r.student_answer for r in recs]
        reference_texts = [r.reference_answer for r in recs]
        self._feature_builder.fit(student_texts, reference_texts)
        X = self._build_features(recs)
        y = [str(getattr(r, label_field)) for r in recs]
        self._clf.fit(X, y)
        self._classes = list(self._clf.classes_)

    def predict(self, records: Iterable[UnifiedRecord]) -> list[str]:
        recs = list(records)
        X = self._build_features(recs)
        return list(self._clf.predict(X))

    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]:
        recs = list(records)
        X = self._build_features(recs)
        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(X).tolist()
        # Fallback: one-hot from hard predictions
        preds = list(self._clf.predict(X))
        result = []
        for p in preds:
            row = [1.0 if c == p else 0.0 for c in self._classes]
            result.append(row)
        return result


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------

class TfidfMLRegressor(GradingModel):
    """TF-IDF + traditional ML regressor.

    Args:
        regressor: One of "svr", "ridge".
        max_features: Maximum number of TF-IDF features per text side.
    """

    VALID_REGRESSORS = frozenset(_REGRESSORS)

    def __init__(
        self,
        regressor: str = "ridge",
        max_features: int = 5000,
    ) -> None:
        if regressor not in self.VALID_REGRESSORS:
            raise ValueError(
                f"regressor must be one of {sorted(self.VALID_REGRESSORS)}, got {regressor!r}"
            )
        self._reg_name = regressor
        self._reg = _REGRESSORS[regressor]()
        self._feature_builder = _TfidfFeatureBuilder(max_features=max_features)

    def _build_features(self, records: list[UnifiedRecord]) -> np.ndarray:
        student_texts = [r.student_answer for r in records]
        reference_texts = [r.reference_answer for r in records]
        return self._feature_builder.transform(student_texts, reference_texts)

    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None:
        recs = list(records)
        student_texts = [r.student_answer for r in recs]
        reference_texts = [r.reference_answer for r in recs]
        self._feature_builder.fit(student_texts, reference_texts)
        X = self._build_features(recs)
        y = [float(getattr(r, label_field)) for r in recs]
        self._reg.fit(X, y)

    def predict(self, records: Iterable[UnifiedRecord]) -> list[float]:
        recs = list(records)
        X = self._build_features(recs)
        return list(self._reg.predict(X))

    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]:
        """Not meaningful for regression; returns [[score]] per record."""
        preds = self.predict(records)
        return [[p] for p in preds]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_tfidf_model(
    model: TfidfMLClassifier,
    records: Iterable[UnifiedRecord],
    label_field: str,
    bootstrap_n: int = 1000,
) -> dict:
    """Run classifier on records and return EvaluationHarness classification metrics.

    Returns a dict with keys: accuracy, macro_f1, weighted_f1, per_class_f1,
    confusion_matrix — each (except confusion_matrix) has 'value', 'ci_lower', 'ci_upper'.
    """
    recs = list(records)
    y_true = [str(getattr(r, label_field)) for r in recs]
    y_pred = model.predict(recs)
    harness = EvaluationHarness()
    return harness.classification_metrics(y_true, y_pred, bootstrap_n=bootstrap_n)


def evaluate_tfidf_regressor(
    model: TfidfMLRegressor,
    records: Iterable[UnifiedRecord],
    label_field: str,
    bootstrap_n: int = 1000,
) -> dict:
    """Run regressor on records and return EvaluationHarness regression metrics.

    Returns a dict with keys: pearson_r, spearman_rho, rmse, mae, qwk —
    each has 'value', 'ci_lower', 'ci_upper'.
    """
    recs = list(records)
    y_true = [float(getattr(r, label_field)) for r in recs]
    y_pred = model.predict(recs)
    harness = EvaluationHarness()
    return harness.regression_metrics(y_true, y_pred, bootstrap_n=bootstrap_n)
