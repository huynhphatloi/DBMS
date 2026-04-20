"""GradingTrainer — wraps HuggingFace Trainer with additional logic.

Features:
  - Saves checkpoints to `models/{model_name}/checkpoint-{step}/`
  - Handles CUDA OOM by reducing batch size by half and retrying once
  - Handles NaN loss by saving checkpoint and raising TrainingError
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Raised when training encounters an unrecoverable error (e.g. NaN loss)."""


class GradingTrainer:
    """Wraps the HuggingFace Trainer with checkpointing and error handling.

    Args:
        model: A HuggingFace model (or compatible object with .parameters()).
        model_name: Slug used for checkpoint directory naming.
        train_dataset: A torch Dataset instance.
        training_args_kwargs: Keyword arguments forwarded to TrainingArguments.
        checkpoint_dir: Root directory for checkpoints (default: "models").
        checkpoint_steps: Save checkpoint every N steps (0 = disabled).
    """

    def __init__(
        self,
        model: Any,
        model_name: str,
        train_dataset: Any,
        training_args_kwargs: dict | None = None,
        checkpoint_dir: str = "models",
        checkpoint_steps: int = 0,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.training_args_kwargs = training_args_kwargs or {}
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_steps = checkpoint_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run training with OOM and NaN-loss handling."""
        batch_size = self.training_args_kwargs.get("per_device_train_batch_size", 16)
        try:
            self._run_training(batch_size)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                reduced = max(1, batch_size // 2)
                logger.error(
                    "CUDA OOM with batch_size=%d. Retrying with batch_size=%d.",
                    batch_size,
                    reduced,
                )
                self._run_training(reduced)
            else:
                raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _checkpoint_output_dir(self) -> str:
        """Return the output_dir for TrainingArguments."""
        slug = self.model_name.replace("/", "_")
        return os.path.join(self.checkpoint_dir, slug)

    def _run_training(self, batch_size: int) -> None:
        """Execute one training run with the given batch size."""
        try:
            import torch  # noqa: PLC0415
            from transformers import TrainingArguments, Trainer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for GradingTrainer"
            ) from exc

        kwargs = dict(self.training_args_kwargs)
        kwargs["per_device_train_batch_size"] = batch_size
        kwargs.setdefault("output_dir", self._checkpoint_output_dir())
        kwargs.setdefault("report_to", "none")
        kwargs.setdefault("logging_steps", 50)

        # Configure checkpointing
        if self.checkpoint_steps > 0:
            kwargs["save_strategy"] = "steps"
            kwargs["save_steps"] = self.checkpoint_steps
        else:
            kwargs.setdefault("save_strategy", "no")

        training_args = TrainingArguments(**kwargs)

        trainer = _NaNAwareTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            checkpoint_dir=self.checkpoint_dir,
            model_name=self.model_name,
        )
        trainer.train()


# ---------------------------------------------------------------------------
# NaN-aware Trainer subclass
# ---------------------------------------------------------------------------

class _NaNAwareTrainer:
    """Minimal Trainer wrapper that detects NaN loss and saves a checkpoint."""

    def __init__(
        self,
        model,
        args,
        train_dataset,
        checkpoint_dir: str,
        model_name: str,
    ) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

    def train(self) -> None:
        """Delegate to HuggingFace Trainer, intercepting NaN loss."""
        try:
            from transformers import Trainer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("transformers is required for _NaNAwareTrainer") from exc

        import torch  # noqa: PLC0415

        # Wrap the model to intercept NaN loss
        original_model = self.model

        class _NaNCheckWrapper:
            """Wraps a model's forward pass to detect NaN loss."""

            def __init__(self, inner, checkpoint_dir, model_name):
                self._inner = inner
                self._checkpoint_dir = checkpoint_dir
                self._model_name = model_name
                self._step = 0
                # Expose attributes the Trainer needs
                self.config = getattr(inner, "config", None)
                self.training = getattr(inner, "training", True)

            def __call__(self, **kwargs):
                out = self._inner(**kwargs)
                self._step += 1
                loss = getattr(out, "loss", None)
                if loss is not None and torch.isnan(loss):
                    self._save_checkpoint()
                    raise TrainingError(
                        f"NaN loss detected at step {self._step}. "
                        "Checkpoint saved."
                    )
                return out

            def train(self, mode=True):
                self.training = mode
                if hasattr(self._inner, "train"):
                    self._inner.train(mode)
                return self

            def eval(self):
                return self.train(False)

            def parameters(self):
                if hasattr(self._inner, "parameters"):
                    return self._inner.parameters()
                return iter([])

            def state_dict(self):
                if hasattr(self._inner, "state_dict"):
                    return self._inner.state_dict()
                return {}

            def _save_checkpoint(self):
                slug = self._model_name.replace("/", "_")
                ckpt_path = os.path.join(
                    self._checkpoint_dir,
                    slug,
                    f"checkpoint-{self._step}",
                )
                os.makedirs(ckpt_path, exist_ok=True)
                try:
                    torch.save(
                        self._inner.state_dict()
                        if hasattr(self._inner, "state_dict")
                        else {},
                        os.path.join(ckpt_path, "model.pt"),
                    )
                    logger.info("NaN checkpoint saved to %s", ckpt_path)
                except Exception as save_exc:  # noqa: BLE001
                    logger.error("Failed to save NaN checkpoint: %s", save_exc)

        wrapped = _NaNCheckWrapper(original_model, self.checkpoint_dir, self.model_name)

        hf_trainer = Trainer(
            model=wrapped,
            args=self.args,
            train_dataset=self.train_dataset,
        )
        hf_trainer.train()
