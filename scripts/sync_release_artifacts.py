#!/usr/bin/env python3
"""Copy the released edge model and sidecars to every place that loads them.

``artifacts_combined/`` is the source of truth for a release.  Four other
places need a copy of part of it:

===================================================  ==========================
Destination                                          Loaded by
===================================================  ==========================
``frontend/public/models/aortica_edge_int8.onnx``    PWA WASM fallback
``mobile/android/app/src/main/assets/model.onnx``    Android OnnxInferenceEngine
``aortica/models/artifacts/trained_outputs.json``    Output gating
``aortica/models/artifacts/class_thresholds.json``   Probability calibration
===================================================  ==========================

Keeping those in sync by hand did not work: the PWA shipped a **two-byte**
``aortica_edge_int8.onnx`` for the entire life of the offline story, and the
Android assets directory held nothing but a README.  Both are the headline
"works without a network" claim.

Run with no arguments to sync.  Run with ``--check`` to verify — CI uses this
so a stale or stub artifact fails the build instead of shipping.

A stub is caught by size, not by content: the two-byte placeholder was
``08 09``, which is a syntactically valid start to an ONNX protobuf.  Nothing
cheap distinguishes a truncated model from a real one except being big
enough to be a model at all.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "artifacts_combined"

#: Refuse any ONNX smaller than this.  The real INT8 graph is ~418 KB; a
#: quantised MobileNet-1D cannot plausibly be under 100 KB.
MIN_ONNX_BYTES = 100_000

_VERSION_RE = re.compile(r"aortica_edge_int8_v(\d+)\.(\d+)\.(\d+)\.onnx$")


class SyncError(RuntimeError):
    """Raised when the release artifacts cannot be located or verified."""


def latest_edge_model(source_dir: Path = SOURCE_DIR) -> Path:
    """Return the highest-versioned INT8 ONNX in *source_dir*."""
    candidates: List[Tuple[Tuple[int, int, int], Path]] = []
    for path in source_dir.glob("aortica_edge_int8_v*.onnx"):
        match = _VERSION_RE.search(path.name)
        if match:
            candidates.append((tuple(int(g) for g in match.groups()), path))  # type: ignore[arg-type]
    if not candidates:
        raise SyncError(
            f"No aortica_edge_int8_v*.onnx found in {source_dir}. "
            "Export and quantise a model before syncing."
        )
    return max(candidates)[1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _targets() -> List[Tuple[Path, Path, int]]:
    """Return ``(source, destination, minimum_bytes)`` for every copy."""
    model = latest_edge_model()
    return [
        (model, REPO_ROOT / "frontend/public/models/aortica_edge_int8.onnx",
         MIN_ONNX_BYTES),
        (model, REPO_ROOT / "mobile/android/app/src/main/assets/model.onnx",
         MIN_ONNX_BYTES),
        (SOURCE_DIR / "trained_outputs.json",
         REPO_ROOT / "aortica/models/artifacts/trained_outputs.json", 1),
        (SOURCE_DIR / "class_thresholds.json",
         REPO_ROOT / "aortica/models/artifacts/class_thresholds.json", 1),
    ]


def _problem(source: Path, dest: Path, floor: int) -> Optional[str]:
    """Return why *dest* is unacceptable, or ``None`` if it is fine."""
    if not source.exists():
        return f"source missing: {source.relative_to(REPO_ROOT)}"
    if not dest.exists():
        return f"missing: {dest.relative_to(REPO_ROOT)}"
    size = dest.stat().st_size
    if size < floor:
        return (
            f"{dest.relative_to(REPO_ROOT)} is {size} bytes, under the "
            f"{floor}-byte floor — this is a placeholder, not a model"
        )
    if _digest(source) != _digest(dest):
        return (
            f"stale: {dest.relative_to(REPO_ROOT)} does not match "
            f"{source.relative_to(REPO_ROOT)}"
        )
    return None


def check() -> List[str]:
    """Return a list of problems; empty means everything is in sync."""
    return [p for p in (_problem(*t) for t in _targets()) if p is not None]


def sync() -> List[str]:
    """Copy every artifact into place.  Returns a list of actions taken."""
    actions: List[str] = []
    for source, dest, floor in _targets():
        if not source.exists():
            raise SyncError(f"Source artifact missing: {source}")
        if source.stat().st_size < floor:
            raise SyncError(
                f"Refusing to publish {source}: {source.stat().st_size} bytes "
                f"is below the {floor}-byte floor."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and _digest(dest) == _digest(source):
            continue
        shutil.copy2(source, dest)
        actions.append(
            f"{source.relative_to(REPO_ROOT)} → {dest.relative_to(REPO_ROOT)} "
            f"({dest.stat().st_size:,} bytes)"
        )
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify without writing; exit non-zero if anything is stale.",
    )
    args = parser.parse_args()

    try:
        if args.check:
            problems = check()
            if problems:
                print("Release artifacts are not in sync:", file=sys.stderr)
                for problem in problems:
                    print(f"  - {problem}", file=sys.stderr)
                print(
                    "\nRun: python scripts/sync_release_artifacts.py",
                    file=sys.stderr,
                )
                return 1
            print("Release artifacts are in sync.")
            return 0

        actions = sync()
        if actions:
            for action in actions:
                print(f"copied {action}")
        else:
            print("Already in sync; nothing to do.")
        return 0
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
