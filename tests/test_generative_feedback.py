"""Tests for the T5 Generative Feedback Generator.

Covers:
- 24.1: T5-base fine-tuning with correct input format
- 24.2: Grounded vs. ungrounded modes produce different inputs
- 24.3: NLI-based factual consistency check / hallucination flagging
- 24.4: Integration tests: fine-tune on small dataset, verify modes differ

Validates: Requirement 20
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.data.schema import UnifiedRecord
from src.feedback.concept_gap import ConceptGapResult
from src.feedback.generative import (
    FactualConsistencyChecker,
    GenerativeFeedbackResult,
    T5GenerativeFeedbackGenerator,
    format_input,
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
        missing_concepts=["chemical energy"],
        label_3way="partially_correct",
        feedback_detailed=(
            "You should mention how light energy is "
            "converted to chemical energy in chloroplasts."
        ),
    )
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


def _make_mock_tokenizer():
    """Create a mock T5 tokenizer."""
    import torch

    mock = MagicMock()
    mock.pad_token_id = 0

    def _tokenize(
        texts, max_length=512, padding=True,
        truncation=True, return_tensors="pt",
    ):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        seq_len = min(max_length, 20)
        result = MagicMock()
        result.input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
        result.attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        return result

    mock.side_effect = _tokenize
    mock.__call__ = _tokenize

    def _decode(ids, skip_special_tokens=True):
        return "Generated feedback about the topic. Review missing concepts."

    mock.decode = MagicMock(side_effect=_decode)
    return mock


def _make_mock_model():
    """Create a mock T5 model that returns plausible outputs."""
    import torch

    mock = MagicMock()

    # Mock forward pass (training)
    def _forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
        result = MagicMock()
        result.loss = torch.tensor(1.5, requires_grad=True)
        result.logits = torch.randn(input_ids.shape[0], 20, 32128)
        return result

    mock.side_effect = _forward
    mock.__call__ = MagicMock(side_effect=_forward)

    # Mock generate
    def _generate(input_ids=None, attention_mask=None, **kwargs):
        return torch.ones(1, 15, dtype=torch.long)

    mock.generate = MagicMock(side_effect=_generate)

    # Mock train/eval mode
    mock.train = MagicMock(return_value=mock)
    mock.eval = MagicMock(return_value=mock)
    mock.to = MagicMock(return_value=mock)
    mock.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])

    return mock


def _make_mock_nli_pipeline(label_map: dict[str, str] | None = None):
    """Create a mock NLI pipeline for consistency checking.

    Parameters
    ----------
    label_map : dict[str, str] | None
        Maps sentence substrings to NLI labels.
        Default: all sentences are entailed (consistent).
    """
    if label_map is None:
        label_map = {}

    def _call(inputs, top_k=1):
        hypothesis = inputs.get("text_pair", "")
        for keyword, label in label_map.items():
            if keyword.lower() in hypothesis.lower():
                return [{"label": label, "score": 0.95}]
        return [{"label": "entailment", "score": 0.9}]

    return MagicMock(side_effect=_call)


# ------------------------------------------------------------------
# 24.1: Input format tests
# ------------------------------------------------------------------


class TestFormatInput:
    """Verify the T5 input format matches the design spec."""

    def test_grounded_format_with_missing_concepts(self):
        record = _make_record()
        gap = ConceptGapResult(
            missing_concepts=["chemical energy", "ATP"],
        )
        result = format_input(
            record, "partially_correct", gap, grounded=True,
        )

        assert "question: What is photosynthesis?" in result
        ref_text = (
            "reference: Photosynthesis converts light "
            "energy into chemical energy."
        )
        assert ref_text in result
        assert "answer: Plants make food from sunlight." in result
        assert "label: partially_correct" in result
        assert "missing: chemical energy, ATP" in result

    def test_grounded_format_no_gap_uses_record_missing(self):
        record = _make_record(missing_concepts=["chemical energy"])
        result = format_input(record, "incorrect", None, grounded=True)

        assert "missing: chemical energy" in result

    def test_grounded_format_no_missing_shows_none(self):
        record = _make_record(missing_concepts=[])
        gap = ConceptGapResult(missing_concepts=[])
        result = format_input(record, "correct", gap, grounded=True)

        assert "missing: none" in result

    def test_ungrounded_format_omits_missing(self):
        record = _make_record(missing_concepts=["chemical energy"])
        gap = ConceptGapResult(missing_concepts=["chemical energy"])
        result = format_input(record, "partially_correct", gap, grounded=False)

        assert "question:" in result
        assert "reference:" in result
        assert "answer:" in result
        assert "label:" in result
        assert "missing:" not in result

    def test_all_fields_present_in_order(self):
        record = _make_record()
        gap = ConceptGapResult(missing_concepts=["ATP"])
        result = format_input(record, "incorrect", gap, grounded=True)

        q_idx = result.index("question:")
        r_idx = result.index("reference:")
        a_idx = result.index("answer:")
        l_idx = result.index("label:")
        m_idx = result.index("missing:")

        assert q_idx < r_idx < a_idx < l_idx < m_idx


# ------------------------------------------------------------------
# 24.1: T5GenerativeFeedbackGenerator interface tests
# ------------------------------------------------------------------


class TestT5GeneratorInterface:
    """Verify the generator implements the FeedbackGenerator ABC."""

    def test_is_subclass_of_feedback_generator(self):
        assert issubclass(T5GenerativeFeedbackGenerator, FeedbackGenerator)

    def test_returns_tuple_of_two_strings(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            _model=mock_model, _tokenizer=mock_tokenizer
        )
        record = _make_record()
        gap = ConceptGapResult(missing_concepts=["chemical energy"])
        result = gen.generate(record, gap, "partially_correct")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_generate_with_metadata_returns_result(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            _model=mock_model, _tokenizer=mock_tokenizer
        )
        record = _make_record()
        gap = ConceptGapResult(missing_concepts=["chemical energy"])
        result = gen.generate_with_metadata(record, gap, "partially_correct")

        assert isinstance(result, GenerativeFeedbackResult)
        assert len(result.feedback_short) > 0
        assert len(result.feedback_detailed) > 0


# ------------------------------------------------------------------
# 24.1: Fine-tuning tests
# ------------------------------------------------------------------


class TestFineTuning:
    """Verify T5 fine-tuning mechanics."""

    def test_fine_tune_returns_summary(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            epochs=2,
            batch_size=2,
            _model=mock_model,
            _tokenizer=mock_tokenizer,
        )

        records = [_make_record(sample_id=f"GEN_{i:04d}") for i in range(4)]
        summary = gen.fine_tune(records)

        assert "epoch_losses" in summary
        assert len(summary["epoch_losses"]) == 2
        assert summary["num_records"] == 4
        assert summary["grounded"] is True

    def test_fine_tune_sets_fine_tuned_flag(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            epochs=1,
            _model=mock_model,
            _tokenizer=mock_tokenizer,
        )
        assert gen.is_fine_tuned is False

        gen.fine_tune([_make_record()])
        assert gen.is_fine_tuned is True

    def test_fine_tune_skips_records_without_feedback(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            epochs=1,
            _model=mock_model,
            _tokenizer=mock_tokenizer,
        )

        records = [
            _make_record(sample_id="GEN_0001", feedback_detailed="Good feedback."),
            _make_record(sample_id="GEN_0002", feedback_detailed=None),
            _make_record(sample_id="GEN_0003", feedback_detailed=""),
        ]
        summary = gen.fine_tune(records)

        # Only the first record has non-empty feedback_detailed
        assert summary["num_records"] == 1

    def test_fine_tune_empty_records(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            _model=mock_model, _tokenizer=mock_tokenizer
        )
        summary = gen.fine_tune([])

        assert summary["num_records"] == 0
        assert summary["epoch_losses"] == []

    def test_fine_tune_epoch_losses_decrease_or_exist(self):
        """Epoch losses should be recorded (values depend on mock)."""
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            epochs=3,
            batch_size=2,
            _model=mock_model,
            _tokenizer=mock_tokenizer,
        )
        records = [_make_record(sample_id=f"GEN_{i:04d}") for i in range(4)]
        summary = gen.fine_tune(records)

        assert len(summary["epoch_losses"]) == 3
        for loss in summary["epoch_losses"]:
            assert isinstance(loss, float)
            assert loss > 0


# ------------------------------------------------------------------
# 24.2: Grounded vs. ungrounded mode tests
# ------------------------------------------------------------------


class TestGroundedVsUngrounded:
    """Verify grounded and ungrounded modes produce different inputs."""

    def test_grounded_mode_includes_missing(self):
        record = _make_record(missing_concepts=["ATP", "chloroplast"])
        gap = ConceptGapResult(missing_concepts=["ATP", "chloroplast"])

        grounded_input = format_input(record, "incorrect", gap, grounded=True)
        assert "missing:" in grounded_input
        assert "ATP" in grounded_input

    def test_ungrounded_mode_excludes_missing(self):
        record = _make_record(missing_concepts=["ATP", "chloroplast"])
        gap = ConceptGapResult(missing_concepts=["ATP", "chloroplast"])

        ungrounded_input = format_input(record, "incorrect", gap, grounded=False)
        assert "missing:" not in ungrounded_input

    def test_grounded_and_ungrounded_differ(self):
        record = _make_record(missing_concepts=["ATP"])
        gap = ConceptGapResult(missing_concepts=["ATP"])

        grounded = format_input(record, "incorrect", gap, grounded=True)
        ungrounded = format_input(record, "incorrect", gap, grounded=False)

        assert grounded != ungrounded
        assert len(grounded) > len(ungrounded)

    def test_generator_grounded_property(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            grounded=True, _model=mock_model, _tokenizer=mock_tokenizer
        )
        assert gen.grounded is True

        gen.grounded = False
        assert gen.grounded is False

    def test_generator_metadata_reflects_mode(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen_grounded = T5GenerativeFeedbackGenerator(
            grounded=True, _model=mock_model, _tokenizer=mock_tokenizer
        )
        gen_ungrounded = T5GenerativeFeedbackGenerator(
            grounded=False, _model=mock_model, _tokenizer=mock_tokenizer
        )

        record = _make_record()
        gap = ConceptGapResult(missing_concepts=["ATP"])

        result_g = gen_grounded.generate_with_metadata(record, gap, "incorrect")
        result_u = gen_ungrounded.generate_with_metadata(record, gap, "incorrect")

        assert result_g.grounded is True
        assert result_u.grounded is False


# ------------------------------------------------------------------
# 24.3: NLI-based factual consistency check tests
# ------------------------------------------------------------------


class TestFactualConsistencyChecker:
    """Verify NLI-based hallucination detection."""

    def test_all_consistent_returns_score_1(self):
        mock_pipe = _make_mock_nli_pipeline()  # all entailment
        checker = FactualConsistencyChecker(_pipeline=mock_pipe)

        score, contradicting = checker.check(
            "Plants use sunlight. They produce oxygen.",
            "Photosynthesis uses sunlight to produce oxygen.",
        )

        assert score == 1.0
        assert contradicting == []

    def test_one_contradiction_flagged(self):
        mock_pipe = _make_mock_nli_pipeline(
            {"does not produce": "contradiction"}
        )
        checker = FactualConsistencyChecker(_pipeline=mock_pipe)

        score, contradicting = checker.check(
            "Plants use sunlight. Plants does not produce oxygen.",
            "Photosynthesis produces oxygen.",
        )

        assert score < 1.0
        assert len(contradicting) == 1
        assert "does not produce" in contradicting[0].lower()

    def test_all_contradictions(self):
        mock_pipe = _make_mock_nli_pipeline(
            {"plants": "contradiction"}
        )
        checker = FactualConsistencyChecker(_pipeline=mock_pipe)

        score, contradicting = checker.check(
            "Plants are bad. Plants don't work.",
            "Plants are essential for life.",
        )

        assert score == 0.0
        assert len(contradicting) == 2

    def test_empty_feedback_returns_perfect_score(self):
        mock_pipe = _make_mock_nli_pipeline()
        checker = FactualConsistencyChecker(_pipeline=mock_pipe)

        score, contradicting = checker.check("", "Some reference.")

        assert score == 1.0
        assert contradicting == []

    def test_single_sentence_consistent(self):
        mock_pipe = _make_mock_nli_pipeline()
        checker = FactualConsistencyChecker(_pipeline=mock_pipe)

        score, contradicting = checker.check(
            "This is correct.",
            "This is the reference.",
        )

        assert score == 1.0
        assert contradicting == []


class TestGeneratorWithConsistencyCheck:
    """Verify the generator integrates the consistency checker."""

    def test_no_checker_means_no_hallucination_flag(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            consistency_checker=None,
            _model=mock_model,
            _tokenizer=mock_tokenizer,
        )
        record = _make_record()
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "correct")

        assert result.is_potential_hallucination is False
        assert result.consistency_score == 1.0
        assert result.contradicting_claims == []

    def test_checker_flags_hallucination(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        # Mock tokenizer decode to return text with a contradiction keyword
        mock_tokenizer.decode = MagicMock(
            return_value="Good feedback. Plants do not need sunlight."
        )

        mock_pipe = _make_mock_nli_pipeline(
            {"do not need sunlight": "contradiction"}
        )
        checker = FactualConsistencyChecker(_pipeline=mock_pipe)

        gen = T5GenerativeFeedbackGenerator(
            consistency_checker=checker,
            _model=mock_model,
            _tokenizer=mock_tokenizer,
        )
        record = _make_record()
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "incorrect")

        assert result.is_potential_hallucination is True
        assert result.consistency_score < 1.0
        assert len(result.contradicting_claims) > 0

    def test_checker_no_hallucination(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        mock_pipe = _make_mock_nli_pipeline()  # all entailment
        checker = FactualConsistencyChecker(_pipeline=mock_pipe)

        gen = T5GenerativeFeedbackGenerator(
            consistency_checker=checker,
            _model=mock_model,
            _tokenizer=mock_tokenizer,
        )
        record = _make_record()
        gap = ConceptGapResult()
        result = gen.generate_with_metadata(record, gap, "correct")

        assert result.is_potential_hallucination is False
        assert result.consistency_score == 1.0


# ------------------------------------------------------------------
# 24.4: Integration tests — fine-tune + generate
# ------------------------------------------------------------------


class TestIntegrationFineTuneAndGenerate:
    """Integration: fine-tune on small dataset, then generate feedback."""

    def test_fine_tune_then_generate(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            epochs=1,
            batch_size=2,
            _model=mock_model,
            _tokenizer=mock_tokenizer,
        )

        # Fine-tune
        records = [
            _make_record(
                sample_id=f"GEN_{i:04d}",
                feedback_detailed=f"Feedback for record {i}.",
            )
            for i in range(4)
        ]
        summary = gen.fine_tune(records)
        assert summary["num_records"] == 4
        assert gen.is_fine_tuned is True

        # Generate
        record = _make_record()
        gap = ConceptGapResult(missing_concepts=["ATP"])
        short, detailed = gen.generate(record, gap, "partially_correct")

        assert isinstance(short, str)
        assert isinstance(detailed, str)
        assert len(detailed) > 0

    def test_grounded_vs_ungrounded_generate_different_inputs(self):
        """Grounded and ungrounded modes should produce different T5 inputs."""
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        # Track what inputs the tokenizer receives
        tokenizer_calls: list[str] = []
        original_side_effect = mock_tokenizer.side_effect

        def _tracking_tokenize(texts, **kwargs):
            if isinstance(texts, str):
                tokenizer_calls.append(texts)
            return original_side_effect(texts, **kwargs)

        mock_tokenizer.side_effect = _tracking_tokenize

        record = _make_record(missing_concepts=["ATP", "chloroplast"])
        gap = ConceptGapResult(missing_concepts=["ATP", "chloroplast"])

        # Grounded generation
        gen_g = T5GenerativeFeedbackGenerator(
            grounded=True, _model=mock_model, _tokenizer=mock_tokenizer
        )
        gen_g.generate(record, gap, "incorrect")

        # Ungrounded generation
        gen_u = T5GenerativeFeedbackGenerator(
            grounded=False, _model=mock_model, _tokenizer=mock_tokenizer
        )
        gen_u.generate(record, gap, "incorrect")

        # The tokenizer should have been called with different input strings
        grounded_calls = [
            c for c in tokenizer_calls if "missing:" in c
        ]
        ungrounded_calls = [
            c for c in tokenizer_calls
            if "missing:" not in c and "question:" in c
        ]

        assert len(grounded_calls) > 0, (
            "Grounded mode should include 'missing:' in input"
        )
        assert len(ungrounded_calls) > 0, (
            "Ungrounded mode should not include 'missing:'"
        )

    def test_fine_tune_grounded_summary(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            grounded=True,
            epochs=1,
            _model=mock_model,
            _tokenizer=mock_tokenizer,
        )
        summary = gen.fine_tune([_make_record()])
        assert summary["grounded"] is True

    def test_fine_tune_ungrounded_summary(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        gen = T5GenerativeFeedbackGenerator(
            grounded=False,
            epochs=1,
            _model=mock_model,
            _tokenizer=mock_tokenizer,
        )
        summary = gen.fine_tune([_make_record()])
        assert summary["grounded"] is False


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the generative feedback generator."""

    def test_short_feedback_extraction_no_punctuation(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()
        mock_tokenizer.decode = MagicMock(return_value="No punctuation here")

        gen = T5GenerativeFeedbackGenerator(
            _model=mock_model, _tokenizer=mock_tokenizer
        )
        record = _make_record()
        gap = ConceptGapResult()
        short, detailed = gen.generate(record, gap, "correct")

        assert short == "No punctuation here"

    def test_short_feedback_extraction_with_period(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()
        mock_tokenizer.decode = MagicMock(
            return_value="First sentence. Second sentence."
        )

        gen = T5GenerativeFeedbackGenerator(
            _model=mock_model, _tokenizer=mock_tokenizer
        )
        record = _make_record()
        gap = ConceptGapResult()
        short, detailed = gen.generate(record, gap, "correct")

        assert short == "First sentence."
        assert detailed == "First sentence. Second sentence."

    def test_empty_generated_text(self):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()
        mock_tokenizer.decode = MagicMock(return_value="")

        gen = T5GenerativeFeedbackGenerator(
            _model=mock_model, _tokenizer=mock_tokenizer
        )
        record = _make_record()
        gap = ConceptGapResult()
        short, detailed = gen.generate(record, gap, "correct")

        assert short == ""
        assert detailed == ""
