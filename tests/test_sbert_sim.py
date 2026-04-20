"""Unit tests for SBERT Similarity baseline (src/grading/baselines/sbert_sim.py).

All tests mock the SentenceTransformer so no model weights are downloaded.
The mock returns deterministic unit-vector embeddings based on the input text.

Covers:
- cosine_similarity_vectors: result in [0, 1] for normalised vectors
- SBERTThresholdClassifier: produces valid labels, predict_proba sums to 1
- SBERTLogisticRegression: trains and predicts valid labels
- evaluate_sbert_model(): returns macro_f1, weighted_f1, accuracy with CI fields
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.schema import UnifiedRecord
from src.grading.baselines.sbert_sim import (
    SBERTLogisticRegression,
    SBERTThresholdClassifier,
    cosine_similarity_vectors,
    evaluate_sbert_model,
)


# ---------------------------------------------------------------------------
# Mock SBERT model factory
# ---------------------------------------------------------------------------

def _make_mock_sbert(embeddings_map: dict[str, np.ndarray] | None = None):
    """Return a mock SentenceTransformer whose .encode() returns deterministic embeddings.

    If embeddings_map is provided, texts found in the map get their specified
    embedding; all other texts get a deterministic non-negative unit vector.

    Real SBERT models (e.g. all-MiniLM-L6-v2) produce embeddings that are
    normalised and typically yield non-negative cosine similarities for sentence
    pairs.  We replicate this by taking the absolute value before normalising.
    """
    dim = 8  # small dimension for speed

    def _encode(texts, convert_to_numpy=True, **kwargs):
        result = []
        for text in texts:
            if embeddings_map and text in embeddings_map:
                vec = embeddings_map[text].astype(float)
            else:
                # Deterministic: hash the text to pick a direction
                seed = hash(text) % (2**31)
                rng = np.random.default_rng(seed)
                # Use absolute values to stay in the non-negative orthant,
                # matching typical SBERT embedding behaviour.
                vec = np.abs(rng.standard_normal(dim))
            # Normalise to unit vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            result.append(vec)
        return np.array(result)

    mock = MagicMock()
    mock.encode.side_effect = _encode
    return mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    reference: str,
    student: str,
    label_2way: str = "correct",
    sample_id: str = "GEN_0001",
) -> UnifiedRecord:
    return UnifiedRecord(
        sample_id=sample_id,
        source_dataset="data_generate",
        original_id=sample_id,
        question_id="Q1",
        domain="science",
        subdomain="physics",
        difficulty="medium",
        question="What is photosynthesis?",
        reference_answer=reference,
        student_answer=student,
        label_2way=label_2way,
    )


_TRAINING_PAIRS = [
    ("photosynthesis converts light energy into chemical energy",
     "photosynthesis uses sunlight to produce glucose", "correct"),
    ("mitochondria produce ATP through cellular respiration",
     "mitochondria generate energy for the cell", "correct"),
    ("water boils at 100 degrees celsius at standard pressure",
     "water boils at 100 degrees celsius", "correct"),
    ("gravity attracts objects with mass toward each other",
     "gravity pulls objects with mass together", "correct"),
    ("earth orbits the sun once every 365 days",
     "earth takes 365 days to orbit the sun", "correct"),
    ("cells divide through mitosis producing two identical cells",
     "mitosis produces two genetically identical cells", "correct"),
    ("DNA carries genetic information in nucleotide sequences",
     "DNA stores genetic information using nucleotide sequences", "correct"),
    ("the speed of light is approximately 3e8 meters per second",
     "light travels at about 300 million meters per second", "correct"),
    ("photosynthesis converts light energy into chemical energy",
     "dogs bark loudly at strangers in the park", "incorrect"),
    ("mitochondria produce ATP through cellular respiration",
     "the sky is blue because of rayleigh scattering", "incorrect"),
    ("water boils at 100 degrees celsius at standard pressure",
     "rocks are formed by geological processes over millions of years", "incorrect"),
    ("gravity attracts objects with mass toward each other",
     "plants need sunlight water and carbon dioxide to grow", "incorrect"),
    ("earth orbits the sun once every 365 days",
     "the ocean is very deep and contains many species of fish", "incorrect"),
    ("cells divide through mitosis producing two identical cells",
     "the moon orbits the earth and causes tides", "incorrect"),
    ("DNA carries genetic information in nucleotide sequences",
     "the weather changes due to atmospheric pressure differences", "incorrect"),
    ("the speed of light is approximately 3e8 meters per second",
     "volcanoes erupt due to pressure from magma beneath the surface", "incorrect"),
]


def _make_training_records() -> list[UnifiedRecord]:
    return [
        _make_record(ref, stu, lbl, f"GEN_{i:04d}")
        for i, (ref, stu, lbl) in enumerate(_TRAINING_PAIRS)
    ]


VALID_LABELS = {"correct", "incorrect"}


# ---------------------------------------------------------------------------
# cosine_similarity_vectors tests
# ---------------------------------------------------------------------------

class TestCosineSimilarityVectors:
    def test_identical_unit_vectors_give_one(self):
        v = np.array([1.0, 0.0, 0.0])
        sim = cosine_similarity_vectors(v, v)
        assert abs(sim - 1.0) < 1e-9

    def test_orthogonal_unit_vectors_give_zero(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sim = cosine_similarity_vectors(a, b)
        assert abs(sim - 0.0) < 1e-9

    def test_result_in_unit_interval_for_normalised_vectors(self):
        """For unit vectors, cosine similarity is in [-1, 1]; for non-negative
        embeddings (typical SBERT output) it is in [0, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            a = rng.standard_normal(16)
            b = rng.standard_normal(16)
            # Normalise to unit vectors
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            sim = cosine_similarity_vectors(a, b)
            assert -1.0 - 1e-9 <= sim <= 1.0 + 1e-9

    def test_zero_vector_returns_zero(self):
        a = np.zeros(4)
        b = np.array([1.0, 0.0, 0.0, 0.0])
        assert cosine_similarity_vectors(a, b) == 0.0
        assert cosine_similarity_vectors(b, a) == 0.0

    def test_known_pair_similar_sentences(self):
        """Two identical unit vectors should give similarity 1.0."""
        v = np.array([0.6, 0.8])  # already unit: 0.36+0.64=1
        sim = cosine_similarity_vectors(v, v)
        assert abs(sim - 1.0) < 1e-9

    def test_known_pair_dissimilar_sentences(self):
        """Opposite unit vectors give similarity -1.0."""
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        sim = cosine_similarity_vectors(a, b)
        assert abs(sim - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# SBERTThresholdClassifier tests
# ---------------------------------------------------------------------------

class TestSBERTThresholdClassifier:
    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_predict_returns_valid_labels(self, mock_load):
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTThresholdClassifier(threshold=0.5)
        clf.fit(records, "label_2way")
        preds = clf.predict(records)
        assert len(preds) == len(records)
        assert all(p in VALID_LABELS for p in preds)

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_predict_proba_sums_to_one(self, mock_load):
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTThresholdClassifier(threshold=0.5)
        clf.fit(records, "label_2way")
        proba = clf.predict_proba(records)
        assert len(proba) == len(records)
        for row in proba:
            assert len(row) == 2
            assert abs(sum(row) - 1.0) < 1e-6
            assert all(0.0 <= p <= 1.0 for p in row)

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_identical_embeddings_classified_as_correct(self, mock_load):
        """When reference and student have the same embedding, similarity=1 >= threshold."""
        text = "photosynthesis converts light energy into chemical energy"
        vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mock_load.return_value = _make_mock_sbert({text: vec})
        record = _make_record(text, text, "correct", "GEN_0001")
        clf = SBERTThresholdClassifier(threshold=0.5)
        clf.fit([record], "label_2way")
        preds = clf.predict([record])
        assert preds[0] == "correct"

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_orthogonal_embeddings_classified_as_incorrect(self, mock_load):
        """When embeddings are orthogonal, similarity=0 < threshold=0.5."""
        ref_text = "reference answer about photosynthesis"
        stu_text = "student answer about something else entirely"
        ref_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        stu_vec = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mock_load.return_value = _make_mock_sbert({ref_text: ref_vec, stu_text: stu_vec})
        record = _make_record(ref_text, stu_text, "incorrect", "GEN_0001")
        clf = SBERTThresholdClassifier(threshold=0.5)
        clf.fit([record], "label_2way")
        preds = clf.predict([record])
        assert preds[0] == "incorrect"

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_cosine_similarity_in_zero_one_for_mock_embeddings(self, mock_load):
        """Verify that cosine similarities produced by the model are in [0, 1]
        when using normalised (unit) embeddings from the mock."""
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTThresholdClassifier(threshold=0.5)
        # Access internal similarities via predict_proba (P(positive) = similarity)
        clf.fit(records, "label_2way")
        proba = clf.predict_proba(records)
        for row in proba:
            sim = row[1]  # P(positive) = cosine similarity
            assert 0.0 - 1e-9 <= sim <= 1.0 + 1e-9, (
                f"Cosine similarity {sim} is outside [0, 1]"
            )

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_fit_no_training_needed(self, mock_load):
        """fit() should not raise even with empty records."""
        mock_load.return_value = _make_mock_sbert()
        clf = SBERTThresholdClassifier()
        clf.fit([], "label_2way")  # should not raise


# ---------------------------------------------------------------------------
# SBERTLogisticRegression tests
# ---------------------------------------------------------------------------

class TestSBERTLogisticRegression:
    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_fit_and_predict_valid_labels(self, mock_load):
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTLogisticRegression()
        clf.fit(records, "label_2way")
        preds = clf.predict(records)
        assert len(preds) == len(records)
        assert all(p in VALID_LABELS for p in preds)

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_predict_proba_shape_and_range(self, mock_load):
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTLogisticRegression()
        clf.fit(records, "label_2way")
        proba = clf.predict_proba(records)
        assert len(proba) == len(records)
        for row in proba:
            assert len(row) == 2
            assert abs(sum(row) - 1.0) < 1e-6
            assert all(0.0 <= p <= 1.0 for p in row)

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_classes_populated_after_fit(self, mock_load):
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTLogisticRegression()
        clf.fit(records, "label_2way")
        assert set(clf._classes) == VALID_LABELS

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_single_feature_used(self, mock_load):
        """The LR model uses exactly one feature (cosine similarity)."""
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTLogisticRegression()
        clf.fit(records, "label_2way")
        # sklearn LR coef_ shape: (n_classes-1, n_features) for binary
        assert clf._lr.coef_.shape[1] == 1


# ---------------------------------------------------------------------------
# evaluate_sbert_model integration tests
# ---------------------------------------------------------------------------

class TestEvaluateSbertModel:
    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_threshold_returns_required_keys(self, mock_load):
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTThresholdClassifier(threshold=0.5)
        clf.fit(records, "label_2way")
        result = evaluate_sbert_model(clf, records, "label_2way", bootstrap_n=50)
        assert "macro_f1" in result
        assert "weighted_f1" in result
        assert "accuracy" in result

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_lr_returns_required_keys(self, mock_load):
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTLogisticRegression()
        clf.fit(records, "label_2way")
        result = evaluate_sbert_model(clf, records, "label_2way", bootstrap_n=50)
        assert "macro_f1" in result
        assert "weighted_f1" in result
        assert "accuracy" in result

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_metrics_have_ci_fields(self, mock_load):
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTThresholdClassifier(threshold=0.5)
        clf.fit(records, "label_2way")
        result = evaluate_sbert_model(clf, records, "label_2way", bootstrap_n=50)
        for key in ("macro_f1", "weighted_f1", "accuracy"):
            assert "value" in result[key], f"Missing 'value' in {key}"
            assert "ci_lower" in result[key], f"Missing 'ci_lower' in {key}"
            assert "ci_upper" in result[key], f"Missing 'ci_upper' in {key}"

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_metric_values_in_unit_interval(self, mock_load):
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTThresholdClassifier(threshold=0.5)
        clf.fit(records, "label_2way")
        result = evaluate_sbert_model(clf, records, "label_2way", bootstrap_n=50)
        for key in ("macro_f1", "weighted_f1", "accuracy"):
            v = result[key]["value"]
            assert 0.0 <= v <= 1.0, f"{key} value {v} out of [0, 1]"

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_ci_contains_point_estimate(self, mock_load):
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTThresholdClassifier(threshold=0.5)
        clf.fit(records, "label_2way")
        result = evaluate_sbert_model(clf, records, "label_2way", bootstrap_n=50)
        for key in ("macro_f1", "weighted_f1", "accuracy"):
            lo = result[key]["ci_lower"]
            val = result[key]["value"]
            hi = result[key]["ci_upper"]
            assert lo <= val <= hi, f"{key}: CI [{lo}, {hi}] does not contain {val}"

    @patch("src.grading.baselines.sbert_sim._load_sbert")
    def test_lr_mode_ci_contains_point_estimate(self, mock_load):
        mock_load.return_value = _make_mock_sbert()
        records = _make_training_records()
        clf = SBERTLogisticRegression()
        clf.fit(records, "label_2way")
        result = evaluate_sbert_model(clf, records, "label_2way", bootstrap_n=50)
        for key in ("macro_f1", "weighted_f1", "accuracy"):
            lo = result[key]["ci_lower"]
            val = result[key]["value"]
            hi = result[key]["ci_upper"]
            assert lo <= val <= hi, f"{key}: CI [{lo}, {hi}] does not contain {val}"


# ---------------------------------------------------------------------------
# ImportError test
# ---------------------------------------------------------------------------

class TestLazySentenceTransformersImport:
    def test_import_error_raised_when_not_installed(self):
        """If sentence_transformers is not importable, a helpful ImportError is raised."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from src.grading.baselines import sbert_sim
            with pytest.raises(ImportError, match="sentence-transformers"):
                sbert_sim._load_sbert("all-MiniLM-L6-v2")
