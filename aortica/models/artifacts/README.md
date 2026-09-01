# Packaged model sidecars

These two files ship **inside the Python package** so that output gating is
available from a plain `pip install aortica`, with no checkpoint download and
no filesystem lookup next to the weights. `aortica.models.output_gating` reads
them at import time.

| File | Purpose |
| --- | --- |
| `trained_outputs.json` | Allowlist of the 28 outputs that carry trained weights, plus the per-task loss mask used to train them. |
| `class_thresholds.json` | Per-class operating points, used to calibrate raw probabilities before CHW tier assignment. |

Both are copies of the v0.3.0 release sidecars in `artifacts_combined/`, which
remains the source of truth. `scripts/sync_model_sidecars.py` refreshes them and
CI fails if they drift.

Update these whenever a new checkpoint is released — a stale allowlist would
let genuinely untrained outputs through.
