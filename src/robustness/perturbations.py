"""Perturbation Engine for robustness evaluation.

Applies programmatic perturbations to student answers for adversarial
testing of grading models. Each perturbation produces a new UnifiedRecord
with ``perturbation_type``, ``adversarial_variant_of``, and
``is_adversarial`` set appropriately.

Supported perturbations:
  - keyword_stuffing: append key_concepts as a list to the student answer
  - verbosity_attack: pad with 3× irrelevant filler text
  - copy_reference: replace student answer with reference_answer verbatim
  - empty_answer: replace student answer with empty string
  - random_text: replace student answer with random tokens
"""

from __future__ import annotations

import random
import string
from dataclasses import asdict

from src.data.schema import UnifiedRecord

# Filler sentences used by the verbosity_attack perturbation.
_FILLER_SENTENCES = [
    "This is additional context that does not relate to the question.",
    "Furthermore, it is important to consider many different perspectives.",
    "In general, there are numerous factors that could be relevant here.",
    "One might also note that the topic is quite broad and complex.",
    "Additionally, various studies have explored related themes.",
    "It should be mentioned that background knowledge varies widely.",
    "Moreover, the subject matter can be interpreted in multiple ways.",
    "There are several aspects worth considering in this discussion.",
    "Some researchers have pointed out tangential observations.",
    "Overall, the landscape of this field is diverse and evolving.",
]

VALID_PERTURBATION_TYPES = frozenset({
    "keyword_stuffing",
    "verbosity_attack",
    "copy_reference",
    "empty_answer",
    "random_text",
})


class PerturbationEngine:
    """Apply adversarial perturbations to ``UnifiedRecord`` instances.

    Parameters
    ----------
    verbosity_multiplier : int
        How many times the filler block is repeated for ``verbosity_attack``.
        Default is 3 (per the design doc).
    random_text_tokens : int
        Number of random tokens generated for ``random_text``. Default 50.
    seed : int | None
        Optional seed for reproducibility of random perturbations.
    """

    def __init__(
        self,
        verbosity_multiplier: int = 3,
        random_text_tokens: int = 50,
        seed: int | None = None,
    ) -> None:
        self.verbosity_multiplier = verbosity_multiplier
        self.random_text_tokens = random_text_tokens
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def perturb(
        self,
        record: UnifiedRecord,
        perturbation_type: str,
    ) -> UnifiedRecord:
        """Return a *new* ``UnifiedRecord`` with the perturbation applied.

        The returned record has:
        - ``perturbation_type`` set to *perturbation_type*
        - ``adversarial_variant_of`` set to the original ``sample_id``
        - ``is_adversarial`` set to ``True``
        - ``sample_id`` suffixed with ``_<perturbation_type>``
        """
        if perturbation_type not in VALID_PERTURBATION_TYPES:
            raise ValueError(
                f"perturbation_type must be one of "
                f"{sorted(VALID_PERTURBATION_TYPES)}, got {perturbation_type!r}"
            )

        handler = getattr(self, f"_apply_{perturbation_type}")
        new_answer: str = handler(record)

        data = asdict(record)
        data["student_answer"] = new_answer
        data["perturbation_type"] = perturbation_type
        data["adversarial_variant_of"] = record.sample_id
        data["is_adversarial"] = True
        data["sample_id"] = f"{record.sample_id}_{perturbation_type}"
        return UnifiedRecord(**data)

    def perturb_all(
        self,
        record: UnifiedRecord,
    ) -> list[UnifiedRecord]:
        """Apply every perturbation type and return the list of new records."""
        return [
            self.perturb(record, pt) for pt in sorted(VALID_PERTURBATION_TYPES)
        ]

    # ------------------------------------------------------------------
    # Individual perturbation implementations
    # ------------------------------------------------------------------

    def _apply_keyword_stuffing(self, record: UnifiedRecord) -> str:
        """Append ``key_concepts`` as a comma-separated list."""
        concepts = ", ".join(record.key_concepts) if record.key_concepts else ""
        if concepts:
            return f"{record.student_answer} {concepts}"
        return record.student_answer

    def _apply_verbosity_attack(self, record: UnifiedRecord) -> str:
        """Pad with filler text repeated ``verbosity_multiplier`` times."""
        filler_block = " ".join(_FILLER_SENTENCES)
        padding = " ".join([filler_block] * self.verbosity_multiplier)
        return f"{record.student_answer} {padding}"

    def _apply_copy_reference(self, record: UnifiedRecord) -> str:
        """Replace student answer with the verbatim ``reference_answer``."""
        return record.reference_answer

    def _apply_empty_answer(self, record: UnifiedRecord) -> str:
        """Replace student answer with an empty string."""
        return ""

    def _apply_random_text(self, record: UnifiedRecord) -> str:
        """Replace student answer with random alphabetic tokens."""
        tokens = [
            "".join(
                self._rng.choices(string.ascii_lowercase, k=self._rng.randint(3, 8))
            )
            for _ in range(self.random_text_tokens)
        ]
        return " ".join(tokens)
