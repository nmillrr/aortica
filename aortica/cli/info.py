"""``aortica info`` — display model version, checkpoint source, and attribution.

Shows the currently loaded/cached model version, SHA-256 hash, checkpoint
source (hub vs. local), and training data attribution.
"""

from __future__ import annotations

from typing import Optional

import click


@click.command("info")
@click.option(
    "--cache-dir",
    type=click.Path(),
    default=None,
    help="Custom cache directory for pretrained models.",
)
def info_cmd(cache_dir: Optional[str]) -> None:
    """Display model version, checkpoint source, and data attribution.

    Shows the currently cached pretrained model information including
    version, variant, SHA-256 hash, and training data provenance.

    Examples:

    \b
        aortica info
        aortica info --cache-dir /custom/cache
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    import aortica
    from aortica.models.registry import DATA_PROVENANCE, get_model_info

    console = Console()

    # ── Package Info ──────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold cyan]Aortica[/bold cyan] v{aortica.__version__}\n"
            "[dim]Open-source AI ECG analysis platform[/dim]"
        ),
        title="[bold white]Package Info[/bold white]",
        border_style="cyan",
    ))

    # ── Model Info ────────────────────────────────────────────────────
    model_info = get_model_info(cache_dir=cache_dir)

    table = Table(
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
    )
    table.add_column("Property", style="bold")
    table.add_column("Value")

    if model_info is not None:
        table.add_row("Model Version", model_info.version)
        table.add_row("Variant", model_info.variant)
        table.add_row("Checkpoint Source", model_info.source)
        table.add_row("SHA-256", model_info.sha256[:16] + "…")
        table.add_row("Cache Path", model_info.cache_path)
        table.add_row("Data Attribution", model_info.data_attribution)
    else:
        table.add_row("Model Version", "[dim]No pretrained model cached[/dim]")
        table.add_row("Variant", "[dim]—[/dim]")
        table.add_row("Checkpoint Source", "[dim]—[/dim]")
        table.add_row("SHA-256", "[dim]—[/dim]")
        table.add_row("Cache Path", "[dim]—[/dim]")
        table.add_row("Data Attribution", DATA_PROVENANCE)

    console.print()
    console.print(Panel(
        table,
        title="[bold white]Model Info[/bold white]",
        border_style="cyan",
    ))
    console.print()
