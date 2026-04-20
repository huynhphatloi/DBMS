"""T5 Generative Feedback — fine-tuned T5-base for feedback generation.

Fine-tunes T5-base on Data_Generate training records.
Input format: ``"question: [Q] reference: [R] answer: [A] label: [L] missing: [M]"``
Target: ``feedback_detailed``.

Supports two modes:
- **Grounded**: includes ``missing_concepts`` in the input prompt.
- **Ungrounded**: omits ``missing_concepts`` for ablation comparison.

Includes an NLI-based factual consistency check that flags records
whose generated feedback contains claims contradicting the reference
answer as potential hallucinations.

Validates: Requirement 20
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.schema import UnifiedRecord
from src.feedback.concept_gap import ConceptGapResult
from src.feedback.template import FeedbackGenerator


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GenerativeFeedbackResult:
    """Metadata returned alongside the generated feedback."""

    feedback_short: str = ""
    feedback_detailed: str = ""
    grounded: bool = True
    is_potential_hallucination: bool = False
    consistency_score: float = 1.0
    contradicting_claims: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input formatting
# ---------------------------------------------------------------------------


def format_input(
    record: UnifiedRecord,
    predicted_label: str,
    gap_result: ConceptGapResult | None = None,
    *,
    grounded: bool = True,
) -> str:
    """Build the T5 input string from a record.

    Parameters
    ----------
    record : UnifiedRecord
        The student answer record.
    predicted_label : str
        The predicted grading label.
    gap_result : ConceptGapResult | None
        Concept gap analysis (used in grounded mode).
    grounded : bool
        If ``True``, include missing concepts in the prompt.

    Returns
    -------
    str
        Formatted input string for T5.
    """
    parts = [
        f"question: {record.question}",
        f"reference: {record.reference_answer}",
        f"answer: {record.student_answer}",
        f"label: {predicted_label}",
    ]

    if grounded:
        missing: list[str] = []
        if gap_result and gap_result.missing_concepts:
            missing = gap_result.missing_concepts
        elif record.missing_concepts:
            missing = record.missing_concepts
        parts.append(f"missing: {', '.join(missing) if missing else 'none'}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# NLI-based factual consistency check
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple heuristic)."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class FactualConsistencyChecker:
    """Check generated feedback for factual consistency via NLI.

    Flags sentences that contradict the reference answer as potential
    hallucinations.

    Parameters
    ----------
    model_name : str
        HuggingFace NLI model identifier.
    device : int
        Device ordinal (``-1`` = CPU).
    _pipeline : object | None
        Injected pipeline for testing.
    """

    # NLI label sets (same as concept_gap.py)
    _CONTRADICTION_LABELS = {"contradiction", "CONTRADICTION", "LABEL_2"}

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: int = -1,
        *,
        _pipeline: object | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._pipeline = _pipeline
        self._loaded = _pipeline is not None

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

    def check(
        self,
        generated_feedback: str,
        reference_answer: str,
    ) -> tuple[float, list[str]]:
        """Check factual consistency of generated feedback.

        Parameters
        ----------
        generated_feedback : str
            The generated feedback text.
        reference_answer : str
            The reference (model) answer to check against.

        Returns
        -------
        tuple[float, list[str]]
            ``(consistency_score, contradicting_claims)`` where
            ``consistency_score`` is the fraction of sentences that are
            NOT contradicting, and ``contradicting_claims`` lists the
            sentences flagged as contradictions.
        """
        sentences = _split_sentences(generated_feedback)
        if not sentences:
            return 1.0, []

        self._ensure_pipeline()

        contradicting: list[str] = []
        for sentence in sentences:
            premise = reference_answer
            hypothesis = sentence

            result = self._pipeline(
                {"text": premise, "text_pair": hypothesis},
                top_k=1,
            )

            if result and isinstance(result, list):
                top = result[0] if isinstance(result[0], dict) else result[0][0]
                label = top["label"]
            else:
                label = "neutral"

            if label in self._CONTRADICTION_LABELS:
                contradicting.append(sentence)

        n_consistent = len(sentences) - len(contradicting)
        consistency_score = n_consistent / len(sentences)
        return consistency_score, contradicting


# ---------------------------------------------------------------------------
# T5GenerativeFeedbackGenerator
# ---------------------------------------------------------------------------


class T5GenerativeFeedbackGenerator(FeedbackGenerator):
    """Fine-tuned T5-base feedback generator.

    Parameters
    ----------
    model_name : str
        HuggingFace T5 model identifier (default: ``"t5-base"``).
    grounded : bool
        Whether to include missing concepts in the input prompt.
    max_input_length : int
        Maximum input token length for T5.
    max_output_length : int
        Maximum output token length for generation.
    learning_rate : float
        Learning rate for fine-tuning.
    epochs : int
        Number of fine-tuning epochs.
    batch_size : int
        Training batch size.
    device : str
        Device string (``"cpu"`` or ``"cuda"``).
    consistency_checker : FactualConsistencyChecker | None
        Optional NLI consistency checker. If ``None``, no consistency
        check is performed.
    _model : object | None
        Injected T5 model for testing.
    _tokenizer : object | None
        Injected T5 tokenizer for testing.
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        grounded: bool = True,
        max_input_length: int = 512,
        max_output_length: int = 256,
        learning_rate: float = 3e-4,
        epochs: int = 5,
        batch_size: int = 8,
        device: str = "cpu",
        consistency_checker: FactualConsistencyChecker | None = None,
        *,
        _model: object | None = None,
        _tokenizer: object | None = None,
    ) -> None:
        self._model_name = model_name
        self._grounded = grounded
        self._max_input_length = max_input_length
        self._max_output_length = max_output_length
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._device = device
        self._consistency_checker = consistency_checker

        self._model = _model
        self._tokenizer = _tokenizer
        self._loaded = _model is not None and _tokenizer is not None
        self._fine_tuned = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazily import transformers and load T5 model + tokenizer."""
        if self._loaded:
            return
        from transformers import T5ForConditionalGeneration, T5Tokenizer  # noqa: WPS433

        self._tokenizer = T5Tokenizer.from_pretrained(self._model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._loaded = True

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        records: list[UnifiedRecord],
        label_field: str = "label_3way",
    ) -> dict:
        """Fine-tune T5 on training records.

        Parameters
        ----------
        records : list[UnifiedRecord]
            Training records with ``feedback_detailed`` as target.
        label_field : str
            Which label field to use for the input prompt.

        Returns
        -------
        dict
            Training summary with keys ``epoch_losses``, ``num_records``,
            ``grounded``.
        """
        import torch  # noqa: WPS433

        self._ensure_model()

        # Filter records that have feedback_detailed
        valid = [r for r in records if r.feedback_detailed]
        if not valid:
            return {"epoch_losses": [], "num_records": 0, "grounded": self._grounded}

        # Prepare input/target pairs
        inputs: list[str] = []
        targets: list[str] = []
        for r in valid:
            label = getattr(r, label_field, None) or "unknown"
            gap = ConceptGapResult(missing_concepts=list(r.missing_concepts))
            inp = format_input(r, str(label), gap, grounded=self._grounded)
            inputs.append(inp)
            targets.append(r.feedback_detailed)

        # Tokenize
        input_encodings = self._tokenizer(
            inputs,
            max_length=self._max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target_encodings = self._tokenizer(
            targets,
            max_length=self._max_output_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token id with -100 for loss computation
        labels = target_encodings.input_ids.clone()
        has_pad = (
            hasattr(self._tokenizer, "pad_token_id")
            and self._tokenizer.pad_token_id is not None
        )
        if has_pad:
            labels[labels == self._tokenizer.pad_token_id] = -100

        # Move to device
        device = self._device
        input_ids = input_encodings.input_ids.to(device)
        attention_mask = input_encodings.attention_mask.to(device)
        labels = labels.to(device)

        # Training loop
        self._model.train()
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._learning_rate
        )

        epoch_losses: list[float] = []
        n = len(inputs)

        for epoch in range(self._epochs):
            total_loss = 0.0
            steps = 0

            for start in range(0, n, self._batch_size):
                end = min(start + self._batch_size, n)
                batch_input_ids = input_ids[start:end]
                batch_attention = attention_mask[start:end]
                batch_labels = labels[start:end]

                outputs = self._model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention,
                    labels=batch_labels,
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

            avg_loss = total_loss / max(steps, 1)
            epoch_losses.append(avg_loss)

        self._model.eval()
        self._fine_tuned = True

        return {
            "epoch_losses": epoch_losses,
            "num_records": len(valid),
            "grounded": self._grounded,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_text(self, input_text: str) -> str:
        """Generate feedback text from a formatted input string."""
        import torch  # noqa: WPS433

        self._ensure_model()

        encoding = self._tokenizer(
            input_text,
            max_length=self._max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding.input_ids.to(self._device)
        attention_mask = encoding.attention_mask.to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self._max_output_length,
                num_beams=4,
                early_stopping=True,
            )

        decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return decoded.strip()

    def _extract_short_feedback(self, detailed: str) -> str:
        """Extract a short feedback summary from the detailed output."""
        if not detailed:
            return ""
        # Take the first sentence.
        for sep in (".", "!", "?"):
            idx = detailed.find(sep)
            if idx != -1:
                return detailed[: idx + 1].strip()
        # No sentence-ending punctuation — return the whole text.
        return detailed.strip()

    # ------------------------------------------------------------------
    # FeedbackGenerator interface
    # ------------------------------------------------------------------

    def generate(
        self,
        record: UnifiedRecord,
        gap_result: ConceptGapResult,
        predicted_label: str,
    ) -> tuple[str, str]:
        """Generate feedback for a student answer.

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
    ) -> GenerativeFeedbackResult:
        """Generate feedback with full metadata including consistency check.

        Returns
        -------
        GenerativeFeedbackResult
            Feedback text plus hallucination flags.
        """
        input_text = format_input(
            record, predicted_label, gap_result, grounded=self._grounded
        )
        detailed = self._generate_text(input_text)
        short = self._extract_short_feedback(detailed)

        # NLI factual consistency check
        is_hallucination = False
        consistency_score = 1.0
        contradicting: list[str] = []

        if self._consistency_checker is not None and detailed:
            consistency_score, contradicting = self._consistency_checker.check(
                generated_feedback=detailed,
                reference_answer=record.reference_answer,
            )
            is_hallucination = len(contradicting) > 0

        return GenerativeFeedbackResult(
            feedback_short=short,
            feedback_detailed=detailed,
            grounded=self._grounded,
            is_potential_hallucination=is_hallucination,
            consistency_score=consistency_score,
            contradicting_claims=contradicting,
        )

    @property
    def grounded(self) -> bool:
        """Whether the generator is in grounded mode."""
        return self._grounded

    @grounded.setter
    def grounded(self, value: bool) -> None:
        self._grounded = value

    @property
    def is_fine_tuned(self) -> bool:
        """Whether the model has been fine-tuned."""
        return self._fine_tuned
