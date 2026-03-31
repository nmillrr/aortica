"""``aortica train <config.yaml>`` — train a multi-task model from the CLI.

Wraps :func:`aortica.models.train_multitask.train_multitask` (or
:func:`train_multitask_tf`) with a Click CLI that accepts a YAML config
file.
"""

from __future__ import annotations

import sys
from typing import Optional

import click

from aortica.models.train_multitask import (
    load_config,
    train_multitask,
    train_multitask_tf,
)


@click.command("train")
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--backend",
    type=click.Choice(["pytorch", "tensorflow"], case_sensitive=False),
    default=None,
    help="Override the training backend (default: read from config YAML).",
)
def train_cmd(
    config_path: str,
    backend: Optional[str],
) -> None:
    """Train the multi-task AorticaModel.

    CONFIG_PATH is the path to a YAML configuration file that specifies
    all training hyperparameters (learning rate, epochs, loss weights,
    dataset path, etc.).

    See ``aortica.models.train_multitask.MultiTaskTrainConfig`` for the
    list of supported configuration keys.

    Examples:

    \b
        aortica train config.yaml
        aortica train config.yaml --backend tensorflow
    """
    # Load config
    try:
        config = load_config(config_path)
    except Exception as exc:
        click.echo(f"Error loading config: {exc}", err=True)
        sys.exit(1)

    # Override backend if specified
    if backend is not None:
        config.backend = backend

    # Run training
    try:
        if config.backend == "tensorflow":
            train_multitask_tf(config)
        else:
            train_multitask(config)
    except Exception as exc:
        click.echo(f"Error during training: {exc}", err=True)
        sys.exit(1)

    click.echo("Training complete.")
