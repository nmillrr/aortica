"""Guards against the packaging mistakes that broke a fresh clone.

Two regressions motivate this file:

* ``.gitignore`` carried an unanchored ``data/`` rule.  Git matches such a
  pattern at any depth, so it silently excluded ``aortica/data/`` and the
  PTB-XL loaders in it were never committed.  The package stopped importing
  from a clean checkout and nothing caught it, because no workflow ran on
  push.
* The same shape of bug is one edit away for ``lib/``, ``env/``, ``build/``
  and friends — all common directory names inside a source tree.

These tests are cheap, need no dependencies beyond git, and fail loudly the
moment a source directory becomes invisible again.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Paths that must remain committable.  They need not exist — the point is
#: that .gitignore would not hide them if they did.
MUST_NOT_BE_IGNORED = [
    "aortica/data/ptbxl.py",
    "aortica/data/ptbxl_labels.py",
    "aortica/data/chapman.py",
    "aortica/models/artifacts/trained_outputs.json",
    "frontend/src/lib/util.ts",
    "frontend/src/env/config.ts",
    "landing/lib/analytics.js",
    "tests/data/fixture.py",
    "docs/data/schema.md",
    "mobile/android/app/src/main/java/io/aortica/ecg/lib/Codec.kt",
]


def _git_available() -> bool:
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _git_available(), reason="not a git checkout"
)


def _ignore_rule(relative_path: str) -> str | None:
    """Return the .gitignore rule hiding *relative_path*, or ``None``."""
    result = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", relative_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # Exit 0 means a rule matched; 1 means none did.
    return result.stdout.strip() or None if result.returncode == 0 else None


@pytest.mark.parametrize("relative_path", MUST_NOT_BE_IGNORED)
def test_source_paths_are_not_gitignored(relative_path: str) -> None:
    """No .gitignore rule may hide a source path."""
    rule = _ignore_rule(relative_path)
    assert rule is None, (
        f"{relative_path} is excluded by .gitignore — {rule}. "
        "Anchor the rule to the repository root (prefix it with '/') so it "
        "cannot match a directory of the same name inside the source tree."
    )


def test_data_package_modules_are_tracked() -> None:
    """Every module in aortica/data/ on disk is also known to git."""
    package = REPO_ROOT / "aortica" / "data"
    on_disk = {p.name for p in package.glob("*.py")}

    result = subprocess.run(
        ["git", "ls-files", "aortica/data"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = {Path(line).name for line in result.stdout.split() if line}

    untracked = sorted(on_disk - tracked)
    assert not untracked, (
        f"aortica/data modules exist but are not tracked: {untracked}. "
        "This is how the PTB-XL loaders were lost — commit them, and check "
        "why git did not offer to."
    )
