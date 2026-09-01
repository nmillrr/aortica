"""Default-on suppression of model outputs that were never trained.

No released checkpoint trains all 72 outputs — v0.3.0 covers 28.  The other
44 heads still emit a number: they sit at initialisation and drift with the
backbone, so their output is noise wearing the costume of a probability.
Two of them are ``mortality_1y`` and ``sudden_cardiac_death_risk``.

The model card has always said callers must gate these before showing them
to anyone.  Relying on callers to remember is the wrong shape for a safety
property, so gating happens here, in the shared inference path, and is on by
default.  Turning it off is an explicit argument, not an omission.

The allowlist ships inside the package (``aortica/models/artifacts/``) rather
than beside the weights, so it is present for any install and cannot go
missing when a checkpoint is loaded from an unusual path.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_TRAINED_OUTPUTS_PATH = _ARTIFACT_DIR / "trained_outputs.json"
_CLASS_THRESHOLDS_PATH = _ARTIFACT_DIR / "class_thresholds.json"


class MissingGatingArtifactError(RuntimeError):
    """Raised when the packaged gating sidecars cannot be read.

    Failing loudly is deliberate: silently falling back to "no gating" would
    turn a packaging error into untrained risk scores reaching a clinician.
    """


@lru_cache(maxsize=1)
def default_trained_outputs() -> frozenset[str]:
    """Return the allowlist of outputs carrying trained weights.

    Raises:
        MissingGatingArtifactError: The packaged sidecar is absent or invalid.
    """
    try:
        data = json.loads(_TRAINED_OUTPUTS_PATH.read_text(encoding="utf-8"))
        names = data["trained_outputs"]
    except (OSError, ValueError, KeyError) as exc:
        raise MissingGatingArtifactError(
            f"Could not read {_TRAINED_OUTPUTS_PATH}. Without it, untrained "
            "model outputs cannot be identified and must not be served. "
            "Reinstall the package or run scripts/sync_model_sidecars.py."
        ) from exc
    if not names:
        raise MissingGatingArtifactError(
            f"{_TRAINED_OUTPUTS_PATH} lists no trained outputs."
        )
    return frozenset(names)


@lru_cache(maxsize=1)
def default_class_thresholds() -> Dict[str, float]:
    """Return per-class operating points for probability calibration.

    Returns an empty mapping when the sidecar is absent — unlike the
    allowlist, missing calibration degrades tier sensitivity rather than
    letting untrained outputs through, so it is not fatal.
    """
    try:
        data = json.loads(_CLASS_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def is_trained(name: str, allowlist: Optional[Iterable[str]] = None) -> bool:
    """Return whether *name* is an output the active checkpoint trained."""
    names = default_trained_outputs() if allowlist is None else frozenset(allowlist)
    return name in names


def kept_indices(
    class_names: Sequence[str],
    *,
    allowlist: Optional[Iterable[str]] = None,
) -> Tuple[List[int], List[str]]:
    """Return the positions to keep for one task head, and what was dropped.

    Args:
        class_names: Output names in head order.
        allowlist: Trained-output names.  Defaults to
            :func:`default_trained_outputs`.

    Returns:
        ``(keep, suppressed_names)`` where *keep* holds the indices into
        *class_names* that survive gating, in head order.  An empty
        *class_names* keeps nothing, but callers that cannot name a head's
        outputs should skip gating rather than suppress it wholesale.
    """
    names = default_trained_outputs() if allowlist is None else frozenset(allowlist)
    keep = [i for i, n in enumerate(class_names) if n in names]
    suppressed = [n for n in class_names if n not in names]
    return keep, suppressed


def filter_task_outputs(
    class_names: Sequence[str],
    probabilities: Sequence[float],
    *,
    allowlist: Optional[Iterable[str]] = None,
) -> Tuple[List[str], List[float], List[str]]:
    """Split one task head's outputs into kept and suppressed.

    Args:
        class_names: Output names in head order.
        probabilities: Predicted values, positionally aligned with
            *class_names*.
        allowlist: Trained-output names.  Defaults to
            :func:`default_trained_outputs`.

    Returns:
        ``(kept_names, kept_probabilities, suppressed_names)``.  The two kept
        lists stay positionally aligned, so every consumer that zips them
        keeps working.
    """
    names = default_trained_outputs() if allowlist is None else frozenset(allowlist)
    kept_names: List[str] = []
    kept_probs: List[float] = []
    suppressed: List[str] = []
    for name, prob in zip(class_names, probabilities):
        if name in names:
            kept_names.append(name)
            kept_probs.append(prob)
        else:
            suppressed.append(name)
    return kept_names, kept_probs, suppressed


def activate_simplified_output_gating(
    *,
    allowlist: Optional[Iterable[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """Point :mod:`aortica.edge.simplified_output` at the packaged sidecars.

    The CHW tier logic keeps its own module-level allowlist and calibration
    table.  Calling this wires both to the packaged defaults so an untrained
    head cannot escalate a community health worker to 'urgent' on noise, and
    so rare classes are compared against their measured operating point
    rather than a clinical threshold they would never reach.
    """
    from aortica.edge.simplified_output import (
        set_class_thresholds,
        set_trained_outputs,
    )

    set_trained_outputs(
        default_trained_outputs() if allowlist is None else allowlist
    )
    resolved = default_class_thresholds() if thresholds is None else thresholds
    set_class_thresholds(resolved or None)
