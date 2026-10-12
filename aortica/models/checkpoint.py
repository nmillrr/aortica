"""Load an :class:`AorticaModel` checkpoint, with the metadata serving needs.

Several entry points used to load checkpoints their own way, with their own
guesses at the architecture (one defaulted ``feature_dim`` to 252; the model
default was 256, which the attention block rejects; checkpoints are 240).
This is the one loader: it rebuilds the architecture from the checkpoint's
own config, refuses a checkpoint trained at a sample rate serving cannot
feed, and attaches the checkpoint's per-class operating points so threshold
decisions use the values tuned for *these* weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from aortica.signal.preprocess import MODEL_SAMPLE_RATE_HZ

_ARCH_KEYS = ("feature_dim", "head_hidden_dim", "head_dropout", "enabled_tasks")


class CheckpointError(ValueError):
    """The file is not a checkpoint this version of Aortica can serve."""


def load_checkpoint(
    path: str | Path,
    *,
    enabled_tasks: Optional[Sequence[str]] = None,
) -> Any:
    """Load *path* into an eval-mode :class:`AorticaModel`.

    Accepts the release format written by ``scripts/train_reference.py``,
    the training format (``best_multitask_model.pt``), a pickled model, or a
    bare state dict.

    Args:
        path: Checkpoint file.
        enabled_tasks: Restrict to these heads (default: the checkpoint's).

    Returns:
        The model, with ``class_thresholds`` (dict, possibly empty),
        ``trained_outputs`` (list or ``None``) and ``version`` attributes set.

    Raises:
        CheckpointError: The checkpoint declares a different sample rate.
    """
    import torch

    from aortica.models.aortica_model import AorticaModel

    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    meta: Dict[str, Any] = {}
    if isinstance(raw, AorticaModel):
        model = raw
    else:
        if isinstance(raw, dict) and "model_state_dict" in raw:
            meta, state = raw, raw["model_state_dict"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            meta, state = raw, raw["state_dict"]
        else:
            state = raw
        config = dict(meta.get("config") or {})

        rate = meta.get("sampling_rate", config.get("sampling_rate"))
        if rate is not None and float(rate) != MODEL_SAMPLE_RATE_HZ:
            raise CheckpointError(
                f"{path} was trained at {rate} Hz; serving feeds "
                f"{MODEL_SAMPLE_RATE_HZ:g} Hz (aortica.signal.preprocess)."
            )

        kwargs = {k: config[k] for k in _ARCH_KEYS if k in config}
        if enabled_tasks is not None:
            kwargs["enabled_tasks"] = list(enabled_tasks)
        model = AorticaModel(in_channels=12, **kwargs)
        # strict=False only when the caller narrowed the heads.
        model.load_state_dict(state, strict=enabled_tasks is None)

    model.eval()
    model.class_thresholds = dict(meta.get("class_thresholds") or {})
    model.trained_outputs = meta.get("trained_outputs")
    model.version = meta.get("version")
    return model
