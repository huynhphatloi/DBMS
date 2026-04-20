"""Tests for the Retrieval-Based Feedback Generator.

Covers:
- Unit tests with a known training set (high-similarity retrieval)
- Edge-case test for low-similarity fallback (similarity < 0.5)
- Verification that both feedback_short and feedback_detailed are produced
- Metadata: similarity_score, low_confidence_retrieval flag, retrieved_sample_id

Validates: Requirement 19
"""

from __future__ import annotations

import numpy as np

from src.data.schema import UnifiedRecord
from src.feedback.concept_gap import ConceptGapResult
from src.feedback.retrieval import (
    RetrievalFeedbackGenerator,
    _first_sentence,
)
from src.feedback.template import FeedbackGenerator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_record(**overrides) -> UnifiedRecord:
    """Create a minimal UnifiedRecord with sensible defaults."""
    defaults = dict(
        sample_id="GEN_0001",
        source_dataset="data_generate",
        original_id="orig_1",
        question_id="q_1",
        domain="biology",
        subdomain="cell_biology",
        difficulty="medium",
        question="What is photosynthesis?",
        reference_answer="Photosynthesis converts light energy into chemical energy.",
        student_answer="Plants make food from sunlight.",
        key_concepts=["photosynthesis", "light energy", "chemical energy"],
    )
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


def _deterministic_encode(texts: list[str]) -> np.ndarray:
    """A deterministic mock encoder that maps text to a fixed-dim vector.

    Uses a simple hash-based approach so that identical texts produce
    identical embeddings and similar texts produce somewhat similar
    embeddings (enough for testing retrieval logic).
    """
    dim = 8
    embeddings = []
    for text in texts:
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        vec = rng.randn(dim).astype(np.float32)
        embeddings.append(vec)
    return np.array(embeddings, dtype=np.float32)


def _constant_encode_factory(vector: np.ndarray):
    """Return an encode function that always returns *vector* for every text."""

    def _encode(texts: list[str]) -> np.ndarray:
        return np.tile(vector, (len(texts), 1)).astype(np.float32)

    return _encode


# ------------------------------------------------------------------
# Interface tests
# ------------------------------------------------------------------


class TestRetrievalFeedbackGeneratorInterface:
    """Verify the generator implements the abstract FeedbackGenerator."""

    def test_is_subclass_of_feedback_generator(self):
        assert issubclass(RetrievalFeedbackGenerator, FeedbackGenerator)

    def test_returns_tuple_of_two_strings(self):
        gen = RetrievalFeedbackGenerator(_encode_fn=_deterministic_encode)
        # Index one training record.
        train = _make_record(
            sample_id="TRAIN_01",
            student_answer="Plants use sunlight to make food.",
            feedback_detailed="Good attempt. Review chloroplast function.",
        )
        gen.index_training_records([train])

        record = _make_record()
        gap = ConceptGapResult(present_concepts=["photosynthesis"])
        result = gen.generate(record, gap, "correct")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)


# ------------------------------------------------------------------
# High-similarity retrieval tests
# ------------------------------------------------------------------


class TestHighSimilarityRetrieval:
    """When the nearest training record is above the threshold."""

    def _build_generator(self) -> RetrievalFeedbackGenerator:
        """Build a generator where query and training texts are identical."""
        # Use a constant encoder so every text maps to the same vector
        # → cosine similarity = 1.0 (always above threshold).
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        gen = RetrievalFeedbackGenerator(
            similarity_threshold=0.5,
            _encode_fn=_constant_encode_factory(vec),
        )
        return gen

    def test_returns_retrieved_feedback_detailed(self):
        gen = self._build_generator()
        train = _make_record(
            sample_id="TRAIN_01",
            student_answer="Plants use sunlight.",
            feedback_detailed="You should mention chloroplasts and ATP production.",
        )
        gen.index_training_records([train])

        record = _make_record(student_answer="Plants use sunlight.")
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "partially_correct")

        assert result.feedback_detailed == (
            "You should mention chloroplasts and ATP production."
        )
        assert result.low_confidence_retrieval is False
        assert result.similarity_score >= 0.5

    def test_feedback_short_is_first_sentence(self):
        gen = self._build_generator()
        train = _make_record(
            sample_id="TRAIN_01",
            student_answer="Plants use sunlight.",
            feedback_detailed=(
                "Good start. But you missed key details"
                " about energy conversion."
            ),
        )
        gen.index_training_records([train])

        record = _make_record(student_answer="Plants use sunlight.")
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "partially_correct")

        assert result.feedback_short == "Good start."

    def test_retrieved_sample_id_is_set(self):
        gen = self._build_generator()
        train = _make_record(
            sample_id="TRAIN_42",
            student_answer="Plants use sunlight.",
            feedback_detailed="Review the light reactions.",
        )
        gen.index_training_records([train])

        record = _make_record(student_answer="Plants use sunlight.")
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "correct")

        assert result.retrieved_sample_id == "TRAIN_42"

    def test_similarity_score_reported(self):
        gen = self._build_generator()
        train = _make_record(
            sample_id="TRAIN_01",
            student_answer="Plants use sunlight.",
            feedback_detailed="Review the light reactions.",
        )
        gen.index_training_records([train])

        record = _make_record(student_answer="Plants use sunlight.")
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "correct")

        assert isinstance(result.similarity_score, float)
        assert result.similarity_score > 0.0

    def test_generate_returns_same_as_metadata(self):
        """The simple generate() should return the same short/detailed."""
        gen = self._build_generator()
        train = _make_record(
            sample_id="TRAIN_01",
            student_answer="Plants use sunlight.",
            feedback_detailed="Review the light reactions.",
        )
        gen.index_training_records([train])

        record = _make_record(student_answer="Plants use sunlight.")
        gap = ConceptGapResult()
        short, detailed = gen.generate(record, gap, "correct")
        meta = gen.generate_with_metadata(record, gap, "correct")

        assert short == meta.feedback_short
        assert detailed == meta.feedback_detailed


# ------------------------------------------------------------------
# Low-similarity fallback tests
# ------------------------------------------------------------------


class TestLowSimilarityFallback:
    """When the best similarity is below the threshold → template fallback."""

    def _build_low_sim_generator(self) -> RetrievalFeedbackGenerator:
        """Build a generator that produces orthogonal embeddings.

        Training text maps to [1, 0, 0, 0] and query maps to [0, 1, 0, 0]
        → cosine similarity = 0.0, well below the 0.5 threshold.
        """
        call_count = {"n": 0}

        def _encode(texts: list[str]) -> np.ndarray:
            results = []
            for _ in texts:
                call_count["n"] += 1
                if call_count["n"] <= 1:
                    # First call: training record embedding.
                    results.append([1.0, 0.0, 0.0, 0.0])
                else:
                    # Subsequent calls: query embedding (orthogonal).
                    results.append([0.0, 1.0, 0.0, 0.0])
            return np.array(results, dtype=np.float32)

        gen = RetrievalFeedbackGenerator(
            similarity_threshold=0.5,
            _encode_fn=_encode,
        )
        return gen

    def test_falls_back_to_template(self):
        gen = self._build_low_sim_generator()
        train = _make_record(
            sample_id="TRAIN_01",
            student_answer="Completely unrelated answer.",
            feedback_detailed="This feedback should NOT be returned.",
        )
        gen.index_training_records([train])

        record = _make_record(student_answer="Something totally different.")
        gap = ConceptGapResult(missing_concepts=["photosynthesis"])
        result = gen.generate_with_metadata(record, gap, "incorrect")

        # Should NOT return the training record's feedback.
        assert result.feedback_detailed != "This feedback should NOT be returned."
        # Should be flagged as low confidence.
        assert result.low_confidence_retrieval is True
        # Template feedback for "incorrect" should mention "incorrect".
        assert "incorrect" in result.feedback_short.lower()

    def test_similarity_score_below_threshold(self):
        gen = self._build_low_sim_generator()
        train = _make_record(
            sample_id="TRAIN_01",
            student_answer="Unrelated.",
            feedback_detailed="Unused feedback.",
        )
        gen.index_training_records([train])

        record = _make_record(student_answer="Different.")
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "incorrect")

        assert result.similarity_score < 0.5

    def test_low_confidence_flag_is_true(self):
        gen = self._build_low_sim_generator()
        train = _make_record(
            sample_id="TRAIN_01",
            student_answer="Unrelated.",
            feedback_detailed="Unused feedback.",
        )
        gen.index_training_records([train])

        record = _make_record(student_answer="Different.")
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "incorrect")

        assert result.low_confidence_retrieval is True


# ------------------------------------------------------------------
# Empty index tests
# ------------------------------------------------------------------


class TestEmptyIndex:
    """When no training records are indexed."""

    def test_falls_back_to_template_with_empty_index(self):
        gen = RetrievalFeedbackGenerator(_encode_fn=_deterministic_encode)
        gen.index_training_records([])

        record = _make_record()
        gap = ConceptGapResult(missing_concepts=["photosynthesis"])
        result = gen.generate_with_metadata(record, gap, "incorrect")

        assert result.low_confidence_retrieval is True
        assert result.similarity_score == 0.0
        assert result.retrieved_sample_id is None

    def test_records_without_feedback_are_skipped(self):
        gen = RetrievalFeedbackGenerator(_encode_fn=_deterministic_encode)
        # Record with no feedback_detailed → should be skipped.
        train = _make_record(
            sample_id="TRAIN_01",
            student_answer="Some answer.",
            feedback_detailed=None,
        )
        gen.index_training_records([train])

        record = _make_record()
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "correct")

        assert result.low_confidence_retrieval is True


# ------------------------------------------------------------------
# Multiple training records
# ------------------------------------------------------------------


class TestMultipleTrainingRecords:
    """Verify the nearest record is selected from multiple candidates."""

    def test_selects_most_similar_record(self):
        """With controlled embeddings, the closest record should be picked."""
        # Embed training records as [1,0,0,0] and [0,0,1,0].
        # Query will be [0.9, 0.1, 0, 0] → closest to first record.
        call_count = {"n": 0}

        def _encode(texts: list[str]) -> np.ndarray:
            results = []
            for _ in texts:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    results.append([1.0, 0.0, 0.0, 0.0])
                elif call_count["n"] == 2:
                    results.append([0.0, 0.0, 1.0, 0.0])
                else:
                    # Query: close to first training record.
                    results.append([0.9, 0.1, 0.0, 0.0])
            return np.array(results, dtype=np.float32)

        gen = RetrievalFeedbackGenerator(
            similarity_threshold=0.3,
            _encode_fn=_encode,
        )

        train1 = _make_record(
            sample_id="TRAIN_01",
            student_answer="Answer A",
            feedback_detailed="Feedback for answer A.",
        )
        train2 = _make_record(
            sample_id="TRAIN_02",
            student_answer="Answer B",
            feedback_detailed="Feedback for answer B.",
        )
        gen.index_training_records([train1, train2])

        record = _make_record(student_answer="Query answer")
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "correct")

        assert result.retrieved_sample_id == "TRAIN_01"
        assert result.feedback_detailed == "Feedback for answer A."
        assert result.low_confidence_retrieval is False


# ------------------------------------------------------------------
# Utility function tests
# ------------------------------------------------------------------


class TestFirstSentence:
    """Tests for the _first_sentence helper."""

    def test_period_ending(self):
        assert _first_sentence("Hello world. More text.") == "Hello world."

    def test_exclamation_ending(self):
        assert _first_sentence("Great job! Keep going.") == "Great job!"

    def test_question_ending(self):
        assert _first_sentence("Did you know? It's true.") == "Did you know?"

    def test_no_punctuation(self):
        assert _first_sentence("No ending punctuation") == "No ending punctuation"

    def test_empty_string(self):
        assert _first_sentence("") == ""

    def test_whitespace_only(self):
        assert _first_sentence("   ") == ""


# ------------------------------------------------------------------
# Configurable threshold tests
# ------------------------------------------------------------------


class TestConfigurableThreshold:
    """Verify the similarity_threshold parameter works correctly."""

    def test_custom_threshold_high(self):
        """With threshold=0.9, even moderate similarity triggers fallback."""
        # Encode training as [1,0,0,0], query as [0.8, 0.6, 0, 0]
        # → cosine sim ≈ 0.8, below threshold of 0.9.
        call_count = {"n": 0}

        def _encode(texts: list[str]) -> np.ndarray:
            results = []
            for _ in texts:
                call_count["n"] += 1
                if call_count["n"] <= 1:
                    results.append([1.0, 0.0, 0.0, 0.0])
                else:
                    results.append([0.8, 0.6, 0.0, 0.0])
            return np.array(results, dtype=np.float32)

        gen = RetrievalFeedbackGenerator(
            similarity_threshold=0.9,
            _encode_fn=_encode,
        )
        train = _make_record(
            sample_id="TRAIN_01",
            student_answer="Answer.",
            feedback_detailed="Retrieved feedback.",
        )
        gen.index_training_records([train])

        record = _make_record(student_answer="Query.")
        gap = ConceptGapResult(missing_concepts=["photosynthesis"])
        result = gen.generate_with_metadata(record, gap, "incorrect")

        assert result.low_confidence_retrieval is True
        assert result.feedback_detailed != "Retrieved feedback."

    def test_custom_threshold_low(self):
        """With threshold=0.1, even low similarity uses retrieval."""
        call_count = {"n": 0}

        def _encode(texts: list[str]) -> np.ndarray:
            results = []
            for _ in texts:
                call_count["n"] += 1
                if call_count["n"] <= 1:
                    results.append([1.0, 0.0, 0.0, 0.0])
                else:
                    results.append([0.2, 0.98, 0.0, 0.0])
            return np.array(results, dtype=np.float32)

        gen = RetrievalFeedbackGenerator(
            similarity_threshold=0.1,
            _encode_fn=_encode,
        )
        train = _make_record(
            sample_id="TRAIN_01",
            student_answer="Answer.",
            feedback_detailed="Retrieved feedback.",
        )
        gen.index_training_records([train])

        record = _make_record(student_answer="Query.")
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "correct")

        assert result.low_confidence_retrieval is False
        assert result.feedback_detailed == "Retrieved feedback."
