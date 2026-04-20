"""Tests for feedback evaluation metrics (Task 26).

Covers:
- ROUGE-L computation with known pairs
- BERTScore fallback (TF-IDF proxy)
- Concept coverage with known missing concepts
- Factual consistency with mock NLI pipeline
- Hallucination rate with mock NLI pipeline
- Human evaluation template generation

Validates: Requirement 21
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.evaluation.metrics import (
    EvaluationHarness,
    compute_bertscore,
    compute_concept_coverage,
    compute_factual_consistency,
    compute_hallucination_rate,
    compute_rouge_l,
    concept_coverage_single,
    export_human_eval_template_json,
    factual_consistency_single,
    generate_human_eval_template,
    has_hallucination,
    rouge_l_sentence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def harness():
    return EvaluationHarness()


class MockNLIPipeline:
    """Mock NLI pipeline that returns configurable labels per sentence."""

    def __init__(self, label_map: dict[str, str] | None = None,
                 default_label: str = "entailment"):
        self._label_map = label_map or {}
        self._default = default_label

    def __call__(self, inputs: dict, top_k: int = 1) -> list[dict]:
        sentence = inputs.get("text_pair", "")
        label = self._label_map.get(sentence, self._default)
        return [{"label": label, "score": 0.95}]


# ---------------------------------------------------------------------------
# 26.1 ROUGE-L tests
# ---------------------------------------------------------------------------

class TestRougeL:
    def test_identical_strings(self):
        scores = rouge_l_sentence("the cat sat on the mat",
                                  "the cat sat on the mat")
        assert scores["precision"] == pytest.approx(1.0)
        assert scores["recall"] == pytest.approx(1.0)
        assert scores["f1"] == pytest.approx(1.0)

    def test_no_overlap(self):
        scores = rouge_l_sentence("hello world", "foo bar baz")
        assert scores["f1"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        scores = rouge_l_sentence("the cat sat", "the cat sat on the mat")
        # LCS = "the cat sat" (3 tokens)
        # precision = 3/3 = 1.0, recall = 3/6 = 0.5
        assert scores["precision"] == pytest.approx(1.0)
        assert scores["recall"] == pytest.approx(0.5)
        expected_f1 = 2 * 1.0 * 0.5 / (1.0 + 0.5)
        assert scores["f1"] == pytest.approx(expected_f1)

    def test_empty_hypothesis(self):
        scores = rouge_l_sentence("", "the cat sat")
        assert scores["f1"] == pytest.approx(0.0)

    def test_empty_reference(self):
        scores = rouge_l_sentence("the cat sat", "")
        assert scores["f1"] == pytest.approx(0.0)

    def test_corpus_level(self):
        generated = ["the cat sat on the mat", "hello world"]
        references = ["the cat sat on the mat", "hello world"]
        result = compute_rouge_l(generated, references)
        assert result["f1"] == pytest.approx(1.0)

    def test_corpus_level_empty(self):
        result = compute_rouge_l([], [])
        assert result["f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 26.1 BERTScore tests (fallback mode)
# ---------------------------------------------------------------------------

class TestBERTScore:
    def test_identical_strings_high_similarity(self):
        generated = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        result = compute_bertscore(generated, references)
        # TF-IDF proxy: identical → cosine sim = 1.0
        assert result["f1"] == pytest.approx(1.0)

    def test_different_strings_lower_similarity(self):
        generated = ["the cat sat on the mat"]
        references = ["quantum physics is fascinating"]
        result = compute_bertscore(generated, references)
        assert result["f1"] < 1.0

    def test_empty_inputs(self):
        result = compute_bertscore([], [])
        assert result["f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 26.2 Concept coverage tests
# ---------------------------------------------------------------------------

class TestConceptCoverage:
    def test_all_concepts_mentioned(self):
        feedback = "You missed photosynthesis and cellular respiration."
        concepts = ["photosynthesis", "cellular respiration"]
        assert concept_coverage_single(feedback, concepts) == pytest.approx(1.0)

    def test_no_concepts_mentioned(self):
        feedback = "Good job on your answer."
        concepts = ["photosynthesis", "cellular respiration"]
        assert concept_coverage_single(feedback, concepts) == pytest.approx(0.0)

    def test_partial_coverage(self):
        feedback = "You should review photosynthesis."
        concepts = ["photosynthesis", "cellular respiration"]
        assert concept_coverage_single(feedback, concepts) == pytest.approx(0.5)

    def test_empty_concepts_list(self):
        feedback = "Some feedback text."
        assert concept_coverage_single(feedback, []) == pytest.approx(1.0)

    def test_case_insensitive(self):
        feedback = "Review PHOTOSYNTHESIS carefully."
        concepts = ["photosynthesis"]
        assert concept_coverage_single(feedback, concepts) == pytest.approx(1.0)

    def test_corpus_level(self):
        generated = [
            "You missed photosynthesis and cellular respiration.",
            "Good job.",
        ]
        gold_concepts = [
            ["photosynthesis", "cellular respiration"],
            ["mitosis"],
        ]
        result = compute_concept_coverage(generated, gold_concepts)
        assert result["mean"] == pytest.approx(0.5)
        assert len(result["per_record"]) == 2
        assert result["per_record"][0] == pytest.approx(1.0)
        assert result["per_record"][1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 26.3 Factual consistency tests
# ---------------------------------------------------------------------------

class TestFactualConsistency:
    def test_all_entailed(self):
        pipeline = MockNLIPipeline(default_label="entailment")
        score = factual_consistency_single(
            "Photosynthesis converts light to energy. Plants need sunlight.",
            "Photosynthesis is the process by which plants convert light.",
            pipeline,
        )
        assert score == pytest.approx(1.0)

    def test_all_contradicted(self):
        pipeline = MockNLIPipeline(default_label="contradiction")
        score = factual_consistency_single(
            "Plants do not need light. Photosynthesis is not real.",
            "Photosynthesis requires light.",
            pipeline,
        )
        assert score == pytest.approx(0.0)

    def test_mixed_labels(self):
        pipeline = MockNLIPipeline(label_map={
            "Photosynthesis converts light to energy.": "entailment",
            "Plants are made of metal.": "contradiction",
        })
        score = factual_consistency_single(
            "Photosynthesis converts light to energy. Plants are made of metal.",
            "Photosynthesis converts light energy.",
            pipeline,
        )
        assert score == pytest.approx(0.5)

    def test_empty_feedback(self):
        pipeline = MockNLIPipeline(default_label="entailment")
        score = factual_consistency_single("", "Some reference.", pipeline)
        assert score == pytest.approx(1.0)

    def test_corpus_level_no_pipeline(self):
        result = compute_factual_consistency(
            ["feedback"], ["reference"], nli_pipeline=None,
        )
        assert result["per_record"] == []

    def test_corpus_level_with_pipeline(self):
        pipeline = MockNLIPipeline(default_label="entailment")
        result = compute_factual_consistency(
            ["Good feedback. Another sentence."],
            ["Reference answer."],
            nli_pipeline=pipeline,
        )
        assert result["mean"] == pytest.approx(1.0)
        assert len(result["per_record"]) == 1


# ---------------------------------------------------------------------------
# 26.4 Hallucination rate tests
# ---------------------------------------------------------------------------

class TestHallucinationRate:
    def test_no_hallucination(self):
        pipeline = MockNLIPipeline(default_label="entailment")
        assert has_hallucination(
            "Correct claim. Another correct claim.",
            "Reference answer.",
            pipeline,
        ) is False

    def test_has_hallucination(self):
        pipeline = MockNLIPipeline(label_map={
            "This is wrong.": "contradiction",
        }, default_label="entailment")
        assert has_hallucination(
            "Correct claim. This is wrong.",
            "Reference answer.",
            pipeline,
        ) is True

    def test_empty_feedback_no_hallucination(self):
        pipeline = MockNLIPipeline(default_label="entailment")
        assert has_hallucination("", "Reference.", pipeline) is False

    def test_corpus_level(self):
        pipeline = MockNLIPipeline(label_map={
            "Bad claim.": "contradiction",
        }, default_label="entailment")
        result = compute_hallucination_rate(
            ["Good feedback.", "Bad claim."],
            ["Ref 1.", "Ref 2."],
            nli_pipeline=pipeline,
        )
        assert result["rate"] == pytest.approx(0.5)
        assert result["per_record"] == [False, True]

    def test_corpus_level_no_pipeline(self):
        result = compute_hallucination_rate(
            ["feedback"], ["reference"], nli_pipeline=None,
        )
        assert result["per_record"] == []


# ---------------------------------------------------------------------------
# 26.5 Human evaluation template tests
# ---------------------------------------------------------------------------

class TestHumanEvalTemplate:
    def _make_records(self, n: int = 10) -> list[dict[str, Any]]:
        labels = ["correct", "partially_correct", "incorrect"]
        return [
            {
                "question": f"Q{i}",
                "reference_answer": f"Ref{i}",
                "student_answer": f"Ans{i}",
                "generated_feedback": f"Feedback{i}",
                "predicted_label": labels[i % 3],
            }
            for i in range(n)
        ]

    def test_template_has_correct_rubric_dimensions(self):
        records = self._make_records(20)
        template = generate_human_eval_template(records, n_samples=5)
        assert len(template) <= 5
        for entry in template:
            assert "rubric" in entry
            for dim in ["accuracy", "specificity", "actionability",
                        "tone", "pedagogical_value"]:
                assert dim in entry["rubric"]
                assert entry["rubric"][dim]["score"] is None
                assert "1-5" in entry["rubric"][dim]["scale"]

    def test_template_stratified_sampling(self):
        records = self._make_records(30)
        template = generate_human_eval_template(
            records, n_samples=9, stratify_by="predicted_label",
        )
        # Should have samples from multiple labels.
        labels_in_template = {e["predicted_label"] for e in template}
        assert len(labels_in_template) >= 2

    def test_template_empty_records(self):
        template = generate_human_eval_template([], n_samples=5)
        assert template == []

    def test_template_json_export(self):
        records = self._make_records(5)
        template = generate_human_eval_template(records, n_samples=3)
        json_str = export_human_eval_template_json(template)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == len(template)

    def test_template_contains_required_fields(self):
        records = self._make_records(10)
        template = generate_human_eval_template(records, n_samples=3)
        for entry in template:
            assert "question" in entry
            assert "reference_answer" in entry
            assert "student_answer" in entry
            assert "generated_feedback" in entry
            assert "predicted_label" in entry
            assert "sample_index" in entry


# ---------------------------------------------------------------------------
# 26.6 Integration: feedback_metrics() on EvaluationHarness
# ---------------------------------------------------------------------------

class TestFeedbackMetricsIntegration:
    def test_full_feedback_metrics(self, harness):
        generated = [
            "You missed photosynthesis. Review the concept.",
            "Great job covering all concepts!",
        ]
        gold = [
            "You need to review photosynthesis. It is a key concept.",
            "Excellent work on all concepts!",
        ]
        gold_concepts = [
            ["photosynthesis"],
            ["cellular respiration"],
        ]
        reference_answers = [
            "Photosynthesis converts light to chemical energy.",
            "Cellular respiration produces ATP.",
        ]
        pipeline = MockNLIPipeline(default_label="entailment")

        result = harness.feedback_metrics(
            generated, gold,
            gold_missing_concepts=gold_concepts,
            reference_answers=reference_answers,
            nli_pipeline=pipeline,
        )

        assert "rouge_l" in result
        assert "bertscore" in result
        assert "concept_coverage" in result
        assert "factual_consistency" in result
        assert "hallucination_rate" in result

        assert 0.0 <= result["rouge_l"]["f1"] <= 1.0
        assert 0.0 <= result["bertscore"]["f1"] <= 1.0
        assert 0.0 <= result["concept_coverage"]["mean"] <= 1.0
        assert 0.0 <= result["factual_consistency"]["mean"] <= 1.0
        assert 0.0 <= result["hallucination_rate"]["rate"] <= 1.0

    def test_feedback_metrics_without_optional(self, harness):
        """Only ROUGE-L and BERTScore when no concepts/references."""
        generated = ["Some feedback."]
        gold = ["Some gold feedback."]

        result = harness.feedback_metrics(generated, gold)

        assert "rouge_l" in result
        assert "bertscore" in result
        assert "concept_coverage" not in result
        assert "factual_consistency" not in result
        assert "hallucination_rate" not in result
