"""The PWA's offline path names ONNX outputs by position.

``frontend/src/data/task_class_names.json`` is how it knows which index is
which class.  If a head gains, loses or reorders a class and this file is not
regenerated, every offline finding is attributed to the wrong condition.
"""

from __future__ import annotations

import json
from pathlib import Path

from aortica.models.task_dims import TASK_CLASS_NAMES

FRONTEND_NAMES = (
    Path(__file__).resolve().parents[2] / "frontend/src/data/task_class_names.json"
)


def test_frontend_class_names_match_model_heads() -> None:
    names = json.loads(FRONTEND_NAMES.read_text(encoding="utf-8"))
    assert names == TASK_CLASS_NAMES, (
        "frontend/src/data/task_class_names.json is stale; regenerate it from "
        "aortica.models.task_dims.TASK_CLASS_NAMES"
    )
