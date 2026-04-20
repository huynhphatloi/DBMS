"""LLM Zero-Shot baseline grader.

Constructs a prompt with question, reference answer, student answer, and a
grading rubric, then sends it to a configurable OpenAI-compatible LLM API.

Supports:
  - 2-way classification: correct / incorrect
  - 3-way classification: correct / partially_correct / incorrect
  - 5-way classification: correct / partially_correct_incomplete /
                          contradictory / irrelevant / non_domain

Unparseable responses are logged and counted as prediction failures.
Retry logic: up to `max_retries` attempts with exponential backoff.

Reports Macro_F1 with Bootstrap_CI for classification tasks.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterable

from src.data.schema import UnifiedRecord
from src.evaluation.metrics import EvaluationHarness

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label sets per task
# ---------------------------------------------------------------------------

LABELS_2WAY = ["correct", "incorrect"]
LABELS_3WAY = ["correct", "partially_correct", "incorrect"]
LABELS_5WAY = [
    "correct",
    "partially_correct_incomplete",
    "contradictory",
    "irrelevant",
    "non_domain",
]

TASK_LABELS: dict[str, list[str]] = {
    "2way": LABELS_2WAY,
    "3way": LABELS_3WAY,
    "5way": LABELS_5WAY,
}

RUBRIC_DESCRIPTIONS: dict[str, str] = {
    "2way": (
        "- correct: The student answer is fully correct.\n"
        "- incorrect: The student answer is wrong or missing key information."
    ),
    "3way": (
        "- correct: The student answer is fully correct.\n"
        "- partially_correct: The student answer is partially correct but missing "
        "some key information or contains minor errors.\n"
        "- incorrect: The student answer is wrong or missing key information."
    ),
    "5way": (
        "- correct: The student answer is fully correct.\n"
        "- partially_correct_incomplete: The student answer is partially correct "
        "but incomplete.\n"
        "- contradictory: The student answer contradicts the reference answer.\n"
        "- irrelevant: The student answer is irrelevant to the question.\n"
        "- non_domain: The student answer is outside the domain of the question."
    ),
}


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
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    reference_answer: str,
    student_answer: str,
    task: str,
) -> str:
    """Construct the grading prompt for the LLM.

    Args:
        question: The question text.
        reference_answer: The reference (correct) answer.
        student_answer: The student's answer to grade.
        task: One of "2way", "3way", "5way".

    Returns:
        A formatted prompt string.
    """
    if task not in TASK_LABELS:
        raise ValueError(f"task must be one of {list(TASK_LABELS)}, got {task!r}")

    labels = TASK_LABELS[task]
    rubric = RUBRIC_DESCRIPTIONS[task]
    label_list = ", ".join(labels)

    prompt = (
        "You are an expert grader for short-answer questions.\n\n"
        f"Question: {question}\n\n"
        f"Reference Answer: {reference_answer}\n\n"
        f"Student Answer: {student_answer}\n\n"
        "Grading Rubric:\n"
        f"{rubric}\n\n"
        f"Based on the rubric above, classify the student answer as one of: {label_list}.\n"
        "Respond with only the label and nothing else."
    )
    return prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(response_text: str, task: str) -> str | None:
    """Extract the predicted label from the LLM response.

    Performs a case-insensitive search for each valid label in the response.
    Returns the first matching label, or None if no valid label is found.

    Args:
        response_text: Raw text returned by the LLM.
        task: One of "2way", "3way", "5way".

    Returns:
        The matched label string, or None if unparseable.
    """
    if task not in TASK_LABELS:
        raise ValueError(f"task must be one of {list(TASK_LABELS)}, got {task!r}")

    text_lower = response_text.lower().strip()
    labels = TASK_LABELS[task]

    # Check for exact match first (most common case)
    if text_lower in labels:
        return text_lower

    # Search for label substring in the response — check longer labels first
    # to avoid "correct" matching inside "partially_correct" or "incorrect"
    for label in sorted(labels, key=len, reverse=True):
        if label in text_lower:
            return label

    return None


# ---------------------------------------------------------------------------
# LLM API call with retry logic
# ---------------------------------------------------------------------------

def _get_openai_client(api_key: str, api_base: str | None):
    """Lazily import and instantiate the OpenAI client."""
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "openai is required for LLMZeroShotGrader. "
            "Install it with: pip install openai"
        ) from exc
    kwargs: dict = {"api_key": api_key}
    if api_base is not None:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def call_llm_api(
    prompt: str,
    api_key: str,
    model: str,
    api_base: str | None,
    max_retries: int,
    backoff_base: float,
) -> str:
    """Call the LLM API with exponential backoff retry logic.

    Args:
        prompt: The prompt to send.
        api_key: API key for authentication.
        model: Model name (e.g., "gpt-4o-mini").
        api_base: Base URL for the API (None → use OpenAI default).
        max_retries: Maximum number of retry attempts.
        backoff_base: Base seconds for exponential backoff (1s, 2s, 4s, ...).

    Returns:
        The raw response text from the LLM.

    Raises:
        Exception: If all retries are exhausted.
    """
    client = _get_openai_client(api_key, api_base)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries:
                wait = backoff_base * (2 ** attempt)
                logger.warning(
                    "LLM API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "LLM API call failed after %d attempts: %s",
                    max_retries + 1,
                    exc,
                )

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMZeroShotGrader
# ---------------------------------------------------------------------------

class LLMZeroShotGrader(GradingModel):
    """Zero-shot grading using a configurable LLM API.

    Args:
        api_key: API key for the LLM service.
        model: LLM model name (default: "gpt-4o-mini").
        api_base: Base URL for the API (default: OpenAI).
        task: "2way", "3way", or "5way" (determines valid labels).
        max_retries: Maximum number of retries on API failure (default: 3).
        backoff_base: Base seconds for exponential backoff (default: 1.0).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        api_base: str | None = None,
        task: str = "3way",
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ) -> None:
        if task not in TASK_LABELS:
            raise ValueError(f"task must be one of {list(TASK_LABELS)}, got {task!r}")
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.task = task
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        self._prediction_failures: int = 0

    @property
    def prediction_failures(self) -> int:
        """Number of responses that could not be parsed into a valid label."""
        return self._prediction_failures

    def fit(self, records: Iterable[UnifiedRecord], label_field: str) -> None:
        """No-op: zero-shot grading does not require training."""

    def predict(self, records: Iterable[UnifiedRecord]) -> list[str]:
        """Call the LLM API for each record and return predicted labels.

        Unparseable responses are logged, counted as failures, and replaced
        with the string "unparseable" in the output list.
        """
        self._prediction_failures = 0
        predictions: list[str] = []

        for record in records:
            prompt = build_prompt(
                question=record.question,
                reference_answer=record.reference_answer,
                student_answer=record.student_answer,
                task=self.task,
            )
            try:
                raw_response = call_llm_api(
                    prompt=prompt,
                    api_key=self.api_key,
                    model=self.model,
                    api_base=self.api_base,
                    max_retries=self.max_retries,
                    backoff_base=self.backoff_base,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("LLM API call failed for record %s: %s", record.sample_id, exc)
                self._prediction_failures += 1
                predictions.append("unparseable")
                continue

            label = parse_response(raw_response, self.task)
            if label is None:
                logger.warning(
                    "Unparseable LLM response for record %s: %r",
                    record.sample_id,
                    raw_response,
                )
                self._prediction_failures += 1
                predictions.append("unparseable")
            else:
                predictions.append(label)

        return predictions

    def predict_proba(self, records: Iterable[UnifiedRecord]) -> list[list[float]]:
        """Return one-hot probabilities based on the predicted label.

        The probability vector has length equal to the number of valid labels
        for the configured task. The predicted label gets probability 1.0;
        all others get 0.0. Unparseable predictions get a uniform distribution.
        """
        recs = list(records)
        labels = TASK_LABELS[self.task]
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        n_classes = len(labels)

        predictions = self.predict(recs)
        result: list[list[float]] = []

        for pred in predictions:
            proba = [0.0] * n_classes
            if pred in label_to_idx:
                proba[label_to_idx[pred]] = 1.0
            else:
                # Unparseable: uniform distribution
                uniform = 1.0 / n_classes
                proba = [uniform] * n_classes
            result.append(proba)

        return result


# ---------------------------------------------------------------------------
# Evaluation helper — integrate with EvaluationHarness
# ---------------------------------------------------------------------------

def evaluate_llm_zeroshot(
    model: LLMZeroShotGrader,
    records: Iterable[UnifiedRecord],
    label_field: str,
    bootstrap_n: int = 1000,
) -> dict:
    """Run the LLM zero-shot grader on records and return classification metrics.

    Unparseable predictions are excluded from metric computation.

    Args:
        model: A configured LLMZeroShotGrader instance.
        records: Iterable of UnifiedRecord instances.
        label_field: Name of the label attribute on UnifiedRecord.
        bootstrap_n: Number of bootstrap iterations for CI computation.

    Returns:
        dict with keys: macro_f1, accuracy, weighted_f1, per_class_f1,
        confusion_matrix — each metric (except confusion_matrix) has
        'value', 'ci_lower', 'ci_upper'. Also includes 'prediction_failures'
        with the count of unparseable responses.
    """
    recs = list(records)
    y_pred_all = model.predict(recs)
    y_true_all = [str(getattr(r, label_field)) for r in recs]

    # Exclude unparseable predictions from metric computation
    y_true_filtered: list[str] = []
    y_pred_filtered: list[str] = []
    for yt, yp in zip(y_true_all, y_pred_all):
        if yp != "unparseable":
            y_true_filtered.append(yt)
            y_pred_filtered.append(yp)

    harness = EvaluationHarness()
    metrics = harness.classification_metrics(
        y_true_filtered, y_pred_filtered, bootstrap_n=bootstrap_n
    )
    metrics["prediction_failures"] = model.prediction_failures
    return metrics
