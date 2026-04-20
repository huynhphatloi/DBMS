"""Unit tests for the LLM Zero-Shot baseline grader.

Tests use unittest.mock.patch to mock the LLM API call so no real API
requests are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.data.schema import UnifiedRecord
from src.grading.baselines.llm_zeroshot import (
    LLMZeroShotGrader,
    build_prompt,
    evaluate_llm_zeroshot,
    parse_response,
    TASK_LABELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    question: str = "What is photosynthesis?",
    reference_answer: str = "Photosynthesis converts light energy into chemical energy.",
    student_answer: str = "Plants use sunlight to make food.",
    sample_id: str = "TEST_001",
    label_3way: str = "correct",
) -> UnifiedRecord:
    return UnifiedRecord(
        sample_id=sample_id,
        source_dataset="scientsbank",
        original_id="001",
        question_id="Q001",
        domain="biology",
        subdomain="plants",
        difficulty="easy",
        question=question,
        reference_answer=reference_answer,
        student_answer=student_answer,
        label_3way=label_3way,
    )


# ---------------------------------------------------------------------------
# Task 15.1 — Prompt construction
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_prompt_contains_question(self):
        prompt = build_prompt(
            question="What is gravity?",
            reference_answer="A force of attraction.",
            student_answer="Things fall down.",
            task="3way",
        )
        assert "What is gravity?" in prompt

    def test_prompt_contains_reference_answer(self):
        prompt = build_prompt(
            question="What is gravity?",
            reference_answer="A force of attraction.",
            student_answer="Things fall down.",
            task="3way",
        )
        assert "A force of attraction." in prompt

    def test_prompt_contains_student_answer(self):
        prompt = build_prompt(
            question="What is gravity?",
            reference_answer="A force of attraction.",
            student_answer="Things fall down.",
            task="3way",
        )
        assert "Things fall down." in prompt

    def test_prompt_contains_rubric_labels_2way(self):
        prompt = build_prompt(
            question="Q", reference_answer="R", student_answer="S", task="2way"
        )
        assert "correct" in prompt
        assert "incorrect" in prompt

    def test_prompt_contains_rubric_labels_3way(self):
        prompt = build_prompt(
            question="Q", reference_answer="R", student_answer="S", task="3way"
        )
        assert "correct" in prompt
        assert "partially_correct" in prompt
        assert "incorrect" in prompt

    def test_prompt_contains_rubric_labels_5way(self):
        prompt = build_prompt(
            question="Q", reference_answer="R", student_answer="S", task="5way"
        )
        for label in TASK_LABELS["5way"]:
            assert label in prompt

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="task must be one of"):
            build_prompt("Q", "R", "S", task="invalid")


# ---------------------------------------------------------------------------
# Task 15.3 — Response parsing
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_parse_correct(self):
        assert parse_response("The answer is correct.", "3way") == "correct"

    def test_parse_incorrect(self):
        assert parse_response("This is incorrect.", "3way") == "incorrect"

    def test_parse_partially_correct(self):
        assert parse_response("The answer is partially_correct.", "3way") == "partially_correct"

    def test_parse_exact_match(self):
        assert parse_response("correct", "3way") == "correct"

    def test_parse_case_insensitive(self):
        assert parse_response("CORRECT", "3way") == "correct"

    def test_parse_unparseable_returns_none(self):
        assert parse_response("I don't know what to say here.", "3way") is None

    def test_parse_empty_response_returns_none(self):
        assert parse_response("", "3way") is None

    def test_parse_2way_correct(self):
        assert parse_response("correct", "2way") == "correct"

    def test_parse_2way_incorrect(self):
        assert parse_response("incorrect", "2way") == "incorrect"

    def test_parse_5way_contradictory(self):
        assert parse_response("contradictory", "5way") == "contradictory"

    def test_parse_invalid_task_raises(self):
        with pytest.raises(ValueError, match="task must be one of"):
            parse_response("correct", "invalid")


# ---------------------------------------------------------------------------
# Task 15.2 — Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Verify that the grader retries on API errors and succeeds on 3rd attempt."""

    def test_retries_on_api_error_then_succeeds(self):
        record = _make_record()

        # Mock the OpenAI client: fail twice, then succeed
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "correct"

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("API timeout")
            return mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = side_effect

        with patch("src.grading.baselines.llm_zeroshot.call_llm_api") as mock_call:
            mock_call.return_value = "correct"
            grader = LLMZeroShotGrader(api_key="test-key", task="3way")
            predictions = grader.predict([record])

        assert predictions == ["correct"]

    def test_retry_exhaustion_counts_as_failure(self):
        record = _make_record()

        with patch("src.grading.baselines.llm_zeroshot.call_llm_api") as mock_call:
            mock_call.side_effect = Exception("API down")
            grader = LLMZeroShotGrader(api_key="test-key", task="3way")
            predictions = grader.predict([record])

        assert predictions == ["unparseable"]
        assert grader.prediction_failures == 1

    def test_actual_retry_logic_in_call_llm_api(self):
        """Test that call_llm_api retries the correct number of times."""
        from src.grading.baselines.llm_zeroshot import call_llm_api

        call_count = 0

        def failing_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary error")
            # Return a mock response on 3rd attempt
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = "correct"
            return mock_resp

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = failing_create

        with patch("src.grading.baselines.llm_zeroshot._get_openai_client", return_value=mock_client):
            # Patch time.sleep to avoid actual delays
            with patch("src.grading.baselines.llm_zeroshot.time.sleep"):
                result = call_llm_api(
                    prompt="test",
                    api_key="key",
                    model="gpt-4o-mini",
                    api_base=None,
                    max_retries=3,
                    backoff_base=1.0,
                )

        assert result == "correct"
        assert call_count == 3


# ---------------------------------------------------------------------------
# Task 15.3 — Unparseable response logging and counting
# ---------------------------------------------------------------------------

class TestUnparseableResponses:
    def test_unparseable_response_counted_as_failure(self):
        record = _make_record()

        with patch("src.grading.baselines.llm_zeroshot.call_llm_api") as mock_call:
            mock_call.return_value = "I cannot determine the grade."
            grader = LLMZeroShotGrader(api_key="test-key", task="3way")
            predictions = grader.predict([record])

        assert predictions == ["unparseable"]
        assert grader.prediction_failures == 1

    def test_multiple_unparseable_responses_counted(self):
        records = [_make_record(sample_id=f"TEST_{i:03d}") for i in range(3)]

        with patch("src.grading.baselines.llm_zeroshot.call_llm_api") as mock_call:
            mock_call.return_value = "I don't know."
            grader = LLMZeroShotGrader(api_key="test-key", task="3way")
            predictions = grader.predict(records)

        assert all(p == "unparseable" for p in predictions)
        assert grader.prediction_failures == 3

    def test_failure_count_resets_on_new_predict_call(self):
        record = _make_record()

        with patch("src.grading.baselines.llm_zeroshot.call_llm_api") as mock_call:
            mock_call.return_value = "I don't know."
            grader = LLMZeroShotGrader(api_key="test-key", task="3way")
            grader.predict([record])
            assert grader.prediction_failures == 1

            # Second call resets the counter
            mock_call.return_value = "correct"
            grader.predict([record])
            assert grader.prediction_failures == 0


# ---------------------------------------------------------------------------
# Task 15.4 — EvaluationHarness integration
# ---------------------------------------------------------------------------

class TestEvaluateLLMZeroshot:
    def test_returns_macro_f1_with_ci_fields(self):
        records = [
            _make_record(sample_id="T001", label_3way="correct"),
            _make_record(sample_id="T002", label_3way="incorrect"),
            _make_record(sample_id="T003", label_3way="partially_correct"),
            _make_record(sample_id="T004", label_3way="correct"),
            _make_record(sample_id="T005", label_3way="incorrect"),
        ]
        responses = ["correct", "incorrect", "partially_correct", "correct", "incorrect"]

        with patch("src.grading.baselines.llm_zeroshot.call_llm_api") as mock_call:
            mock_call.side_effect = responses
            grader = LLMZeroShotGrader(api_key="test-key", task="3way")
            result = evaluate_llm_zeroshot(
                grader, records, label_field="label_3way", bootstrap_n=10
            )

        assert "macro_f1" in result
        assert "value" in result["macro_f1"]
        assert "ci_lower" in result["macro_f1"]
        assert "ci_upper" in result["macro_f1"]

    def test_prediction_failures_in_result(self):
        records = [
            _make_record(sample_id="T001", label_3way="correct"),
            _make_record(sample_id="T002", label_3way="incorrect"),
        ]
        responses = ["correct", "I cannot determine this."]

        with patch("src.grading.baselines.llm_zeroshot.call_llm_api") as mock_call:
            mock_call.side_effect = responses
            grader = LLMZeroShotGrader(api_key="test-key", task="3way")
            result = evaluate_llm_zeroshot(
                grader, records, label_field="label_3way", bootstrap_n=10
            )

        assert result["prediction_failures"] == 1

    def test_prediction_failure_count_tracked(self):
        records = [_make_record(sample_id=f"T{i:03d}") for i in range(4)]

        with patch("src.grading.baselines.llm_zeroshot.call_llm_api") as mock_call:
            mock_call.side_effect = ["correct", "unparseable text", "incorrect", "correct"]
            grader = LLMZeroShotGrader(api_key="test-key", task="3way")
            result = evaluate_llm_zeroshot(
                grader, records, label_field="label_3way", bootstrap_n=10
            )

        assert result["prediction_failures"] == 1


# ---------------------------------------------------------------------------
# fit() is a no-op
# ---------------------------------------------------------------------------

class TestFitNoOp:
    def test_fit_is_noop(self):
        grader = LLMZeroShotGrader(api_key="test-key", task="3way")
        records = [_make_record()]
        # Should not raise and should not change any state
        grader.fit(records, "label_3way")
        assert grader.prediction_failures == 0


# ---------------------------------------------------------------------------
# predict_proba() returns one-hot probabilities
# ---------------------------------------------------------------------------

class TestPredictProba:
    def test_predict_proba_one_hot_for_correct(self):
        record = _make_record()
        labels = TASK_LABELS["3way"]

        with patch("src.grading.baselines.llm_zeroshot.call_llm_api") as mock_call:
            mock_call.return_value = "correct"
            grader = LLMZeroShotGrader(api_key="test-key", task="3way")
            proba = grader.predict_proba([record])

        assert len(proba) == 1
        assert len(proba[0]) == len(labels)
        correct_idx = labels.index("correct")
        assert proba[0][correct_idx] == 1.0
        assert sum(proba[0]) == pytest.approx(1.0)

    def test_predict_proba_uniform_for_unparseable(self):
        record = _make_record()
        labels = TASK_LABELS["3way"]

        with patch("src.grading.baselines.llm_zeroshot.call_llm_api") as mock_call:
            mock_call.return_value = "I cannot determine this."
            grader = LLMZeroShotGrader(api_key="test-key", task="3way")
            proba = grader.predict_proba([record])

        assert len(proba[0]) == len(labels)
        expected_uniform = 1.0 / len(labels)
        for p in proba[0]:
            assert p == pytest.approx(expected_uniform)
