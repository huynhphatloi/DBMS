"""Retrieval-based feedback generation — SBERT nearest-neighbour baseline.

Encodes the student answer with SBERT, retrieves the most similar
training record by cosine similarity, and returns its ``feedback_detailed``.
Falls back to template-based feedback when the best similarity is below
a configurable threshold (default 0.5), flagging the result as
``low_confidence_retrieval``.

Validates: Requirement 19
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.schema import UnifiedRecord
from src.feedback.concept_gap import ConceptGapResult
from src.feedback.template import FeedbackGenerator, TemplateFeedbackGenerator


@dataclass
class RetrievalResult:
    """Metadata returned alongside the generated feedback."""

    feedback_short: str = ""
    feedback_detailed: str = ""
    similarity_score: float = 0.0
    low_confidence_retrieval: bool = False
    retrieved_sample_id: str | None = None


class RetrievalFeedbackGenerator(FeedbackGenerator):
    """Retrieve feedback from the nearest training record by SBERT similarity.

    Parameters
    ----------
    model_name : str
        Sentence-transformers model identifier.
        Default: ``"all-MiniLM-L6-v2"``.
    similarity_threshold : float
        Minimum cosine similarity to trust the retrieved feedback.
        Below this value the generator falls back to template-based output
        and flags the record as ``low_confidence_retrieval``.
    _encode_fn : callable | None
        Optional injection point for the encoding function (for testing).
        Signature: ``(texts: list[str]) -> np.ndarray`` returning an array
        of shape ``(len(texts), embedding_dim)``.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        *,
        _encode_fn: object | None = None,
    ) -> None:
        self._model_name = model_name
        self._similarity_threshold = similarity_threshold
        self._encode_fn = _encode_fn
        self._model: object | None = None
        self._loaded = _encode_fn is not None

        # Training index: populated via ``index_training_records``.
        self._training_records: list[UnifiedRecord] = []
        self._training_embeddings: np.ndarray | None = None

        # Template fallback generator.
        self._template_gen = TemplateFeedbackGenerator()

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazily load the sentence-transformers model."""
        if self._loaded:
            return
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        self._model = SentenceTransformer(self._model_name)
        self._encode_fn = self._model.encode  # type: ignore[union-attr]
        self._loaded = True

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into normalised embeddings."""
        self._ensure_model()
        embeddings = self._encode_fn(texts)  # type: ignore[misc]
        embeddings = np.asarray(embeddings, dtype=np.float32)
        # L2-normalise so dot product == cosine similarity.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    # ------------------------------------------------------------------
    # Training index
    # ------------------------------------------------------------------

    def index_training_records(self, records: list[UnifiedRecord]) -> None:
        """Build the retrieval index from training records.

        Parameters
        ----------
        records : list[UnifiedRecord]
            Training records whose ``feedback_detailed`` will be used as
            retrieval candidates.  Records without ``feedback_detailed``
            are silently skipped.
        """
        valid = [r for r in records if r.feedback_detailed]
        if not valid:
            self._training_records = []
            self._training_embeddings = None
            return

        self._training_records = valid
        texts = [r.student_answer for r in valid]
        self._training_embeddings = self._encode(texts)

    # ------------------------------------------------------------------
    # Cosine similarity retrieval
    # ------------------------------------------------------------------

    def _retrieve_nearest(
        self, query_embedding: np.ndarray
    ) -> tuple[UnifiedRecord | None, float]:
        """Return the training record most similar to *query_embedding*.

        Returns ``(None, 0.0)`` when the index is empty.
        """
        if (
            self._training_embeddings is None
            or len(self._training_records) == 0
        ):
            return None, 0.0

        # query_embedding is already L2-normalised, so dot == cosine.
        similarities = self._training_embeddings @ query_embedding
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        return self._training_records[best_idx], best_sim

    # ------------------------------------------------------------------
    # FeedbackGenerator interface
    # ------------------------------------------------------------------

    def generate(
        self,
        record: UnifiedRecord,
        gap_result: ConceptGapResult,
        predicted_label: str,
    ) -> tuple[str, str]:
        """Generate feedback via retrieval (or template fallback).

        Returns
        -------
        tuple[str, str]
            ``(feedback_short, feedback_detailed)``
        """
        result = self.generate_with_metadata(record, gap_result, predicted_label)
        return result.feedback_short, result.feedback_detailed

    def generate_with_metadata(
        self,
        record: UnifiedRecord,
        gap_result: ConceptGapResult,
        predicted_label: str,
    ) -> RetrievalResult:
        """Generate feedback and return full retrieval metadata.

        This extended method exposes the similarity score and the
        ``low_confidence_retrieval`` flag for downstream analysis.
        """
        query_emb = self._encode([record.student_answer])[0]
        nearest, similarity = self._retrieve_nearest(query_emb)

        if nearest is None or similarity < self._similarity_threshold:
            # Fall back to template-based feedback.
            short, detailed = self._template_gen.generate(
                record, gap_result, predicted_label
            )
            return RetrievalResult(
                feedback_short=short,
                feedback_detailed=detailed,
                similarity_score=similarity,
                low_confidence_retrieval=True,
                retrieved_sample_id=nearest.sample_id if nearest else None,
            )

        # Use the retrieved record's feedback.
        feedback_detailed = nearest.feedback_detailed or ""
        # Construct a short summary from the first sentence.
        feedback_short = _first_sentence(feedback_detailed)

        return RetrievalResult(
            feedback_short=feedback_short,
            feedback_detailed=feedback_detailed,
            similarity_score=similarity,
            low_confidence_retrieval=False,
            retrieved_sample_id=nearest.sample_id,
        )


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------


def _first_sentence(text: str) -> str:
    """Extract the first sentence from *text* as a short summary."""
    # Find the earliest sentence-ending punctuation.
    earliest_idx = -1
    for sep in (".", "!", "?"):
        idx = text.find(sep)
        if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
            earliest_idx = idx
    if earliest_idx != -1:
        return text[: earliest_idx + 1].strip()
    # No sentence-ending punctuation found — return the whole text.
    return text.strip()
