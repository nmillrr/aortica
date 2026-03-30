"""Aortica CLI — Click-based command-line interface.

Provides ``aortica predict``, ``aortica benchmark``, and ``aortica train``
commands for ECG analysis from the terminal.
"""

from __future__ import annotations

try:
    import click as _click  # noqa: F401

    HAS_CLICK = True
except ImportError:  # pragma: no cover
    HAS_CLICK = False

try:
    import rich as _rich  # noqa: F401

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False


def _check_cli_deps() -> None:
    """Raise *ImportError* if CLI dependencies are missing."""
    if not HAS_CLICK:
        raise ImportError(
            "Click is required for the Aortica CLI. "
            "Install it with: pip install aortica[cli]"
        )
    if not HAS_RICH:
        raise ImportError(
            "Rich is required for the Aortica CLI. "
            "Install it with: pip install aortica[cli]"
        )
