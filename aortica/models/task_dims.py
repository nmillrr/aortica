"""Canonical per-task output dimensions — the single source of truth (US-129).

The multi-task model concatenates four task-head outputs in a fixed order:
``rhythm``, ``structural``, ``ischaemia``, ``risk``.  Their sizes are
**derived** from the head class-list constants so they can never drift out
of sync with the heads again — expanding a head automatically updates every
downstream consumer that imports :data:`TASK_NUM_OUTPUTS`.

This module deliberately imports only the class-list constants from the head
modules, whose ``torch``/``tf`` imports are guarded, so importing it does
**not** force a heavyweight ML dependency at load time.  This preserves the
lazy-import behaviour of modules such as ``federated/fl_client.py``.
"""

from __future__ import annotations

from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
from aortica.models.rhythm_head import RHYTHM_CLASSES
from aortica.models.risk_head import RISK_OUTPUTS
from aortica.models.structural_head import STRUCTURAL_CLASSES

#: Task order used when concatenating per-task label/prediction columns.
ALL_TASKS: list[str] = ["rhythm", "structural", "ischaemia", "risk"]

#: Classification tasks (multi-label sigmoid heads); ``risk`` is continuous.
CLASSIFICATION_TASKS: list[str] = ["rhythm", "structural", "ischaemia"]

#: Canonical per-task output dimensions, derived from the head class lists.
TASK_NUM_OUTPUTS: dict[str, int] = {
    "rhythm": len(RHYTHM_CLASSES),
    "structural": len(STRUCTURAL_CLASSES),
    "ischaemia": len(ISCHAEMIA_CLASSES),
    "risk": len(RISK_OUTPUTS),
}

#: Total concatenated output width across all task heads.
TOTAL_OUTPUTS: int = sum(TASK_NUM_OUTPUTS.values())
