"""Concept Gap Detector — NLI-based classification of key concepts.

Uses a Natural Language Inference model to classify each key concept
from the reference answer as present, missing, or contradicted in
the student answer. Falls back to noun-phrase extraction from the
reference answer when no key concepts are provided.

Validates: Requirement 17
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ConceptGapResult:
    """Structured result of concept gap detection."""

    present_concepts: list[str] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)
    contradicted_concepts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Noun-phrase extraction fallback (regex-based, no spacy dependency)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might",
    "can", "could", "must", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "and", "but",
    "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no",
    "only", "own", "same", "than", "too", "very",
    "it", "its", "they", "them", "their", "we", "us",
    "he", "she", "him", "her", "his", "my", "your",
    "our", "which", "who", "whom", "what", "where",
    "when", "how", "if", "then", "also", "about",
})


def extract_noun_phrases(text: str) -> list[str]:
    """Extract candidate noun phrases from *text* via chunking.

    Uses a lightweight approach: split on punctuation / stop-word
    boundaries, keep multi-word spans that look like noun phrases.
    No spacy dependency required.
    """
    if not text or not text.strip():
        return []

    # Split into word tokens
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)

    phrases: list[str] = []
    current_chunk: list[str] = []

    for word in words:
        lower = word.lower()
        if lower in _STOP_WORDS:
            # Flush current chunk
            if current_chunk:
                phrases.append(" ".join(current_chunk))
                current_chunk = []
        else:
            current_chunk.append(lower)

    # Flush remaining
    if current_chunk:
        phrases.append(" ".join(current_chunk))

    # Filter: keep phrases with at least one token of length >= 3
    filtered = [
        p for p in phrases
        if any(len(t) >= 3 for t in p.split())
    ]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in filtered:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# NLI label mapping
# ---------------------------------------------------------------------------

# Standard NLI label names → our concept-gap categories.
# Different models may use different label conventions; we normalise here.
_ENTAILMENT_LABELS = {"entailment", "ENTAILMENT", "LABEL_0"}
_CONTRADICTION_LABELS = {"contradiction", "CONTRADICTION", "LABEL_2"}
# Everything else (neutral, NEUTRAL, LABEL_1, …) → missing


def _map_nli_label(label: str) -> str:
    """Map an NLI model output label to present/missing/contradicted."""
    if label in _ENTAILMENT_LABELS:
        return "present"
    if label in _CONTRADICTION_LABELS:
        return "contradicted"
    return "missing"


# ---------------------------------------------------------------------------
# ConceptGapDetector
# ---------------------------------------------------------------------------


class ConceptGapDetector:
    """Detect which key concepts are present, missing, or contradicted.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the NLI classifier.
        Default: ``"cross-encoder/nli-deberta-v3-base"``.
    device : int
        Device ordinal for the transformers pipeline (``-1`` = CPU).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: int = -1,
        *,
        _pipeline: object | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        # Allow injection for testing (avoids downloading weights).
        self._pipeline = _pipeline
        self._loaded = _pipeline is not None

    # -- lazy loading -------------------------------------------------------

    def _ensure_pipeline(self) -> None:
        """Lazily import transformers and instantiate the NLI pipeline."""
        if self._loaded:
            return
        from transformers import pipeline as hf_pipeline  # noqa: WPS433

        self._pipeline = hf_pipeline(
            "text-classification",
            model=self._model_name,
            device=self._device,
        )
        self._loaded = True

    # -- public API ---------------------------------------------------------

    def detect(
        self,
        question: str,
        reference_answer: str,
        student_answer: str,
        key_concepts: list[str] | None = None,
    ) -> ConceptGapResult:
        """Classify each key concept as present / missing / contradicted.

        Parameters
        ----------
        question : str
            The question text (used as context for NLI premise).
        reference_answer : str
            The reference (model) answer.
        student_answer : str
            The student's answer to evaluate.
        key_concepts : list[str] | None
            Concepts to check.  If empty or ``None``, concepts are
            extracted from *reference_answer* via noun-phrase extraction.

        Returns
        -------
        ConceptGapResult
            Lists of present, missing, and contradicted concepts.
        """
        # Fallback: extract concepts from reference answer when none given.
        if not key_concepts:
            key_concepts = extract_noun_phrases(reference_answer)

        # Edge case: still no concepts after extraction.
        if not key_concepts:
            return ConceptGapResult()

        self._ensure_pipeline()

        present: list[str] = []
        missing: list[str] = []
        contradicted: list[str] = []

        for concept in key_concepts:
            # Premise: the student answer (what we know).
            # Hypothesis: the concept is addressed.
            premise = student_answer
            hypothesis = f"The answer discusses {concept}."

            result = self._pipeline(
                {"text": premise, "text_pair": hypothesis},
                top_k=1,
            )

            # result is a list of dicts: [{"label": "...", "score": ...}]
            if result and isinstance(result, list):
                top = result[0] if isinstance(result[0], dict) else result[0][0]
                label = _map_nli_label(top["label"])
            else:
                # Defensive: treat unexpected output as missing.
                label = "missing"

            if label == "present":
                present.append(concept)
            elif label == "contradicted":
                contradicted.append(concept)
            else:
                missing.append(concept)

        return ConceptGapResult(
            present_concepts=present,
            missing_concepts=missing,
            contradicted_concepts=contradicted,
        )
