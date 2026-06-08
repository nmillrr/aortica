"""Aortica CLI — command-line interface for ECG analysis.

Provides ``aortica predict``, ``aortica benchmark``, and ``aortica train``
commands powered by Click with Rich-formatted terminal output.
"""

from __future__ import annotations

from typing import Any

try:
    import click

    HAS_CLICK = True
except ImportError:  # pragma: no cover
    HAS_CLICK = False

try:
    import rich  # noqa: F401

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False


def _check_cli_deps() -> None:
    """Raise *ImportError* if click or rich are not installed."""
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


def _build_cli() -> Any:
    """Build and return the Click CLI group (lazy to avoid import failures)."""
    _check_cli_deps()

    from aortica.cli.benchmark import benchmark_cmd
    from aortica.cli.build_index_cmd import build_index_cmd
    from aortica.cli.federated_cmd import federated_group
    from aortica.cli.info import info_cmd
    from aortica.cli.performance_card_cmd import performance_card_cmd
    from aortica.cli.predict import predict
    from aortica.cli.profile_cmd import profile_cmd
    from aortica.cli.train import train_cmd
    from aortica.cli.validation_cmd import validation_group

    @click.group()
    @click.version_option(package_name="aortica")
    def cli() -> None:
        """Aortica — AI-powered ECG analysis from the command line."""

    cli.add_command(predict)
    cli.add_command(benchmark_cmd)
    cli.add_command(train_cmd)
    cli.add_command(info_cmd)
    cli.add_command(profile_cmd)
    cli.add_command(federated_group)
    cli.add_command(performance_card_cmd)
    cli.add_command(build_index_cmd)
    cli.add_command(validation_group)
    return cli


def main() -> None:
    """Entry point for the ``aortica`` CLI."""
    cli = _build_cli()
    cli()


__all__ = ["main"]
