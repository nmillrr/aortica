"""CLI commands for Aortica federated learning operations.

Provides:
- ``aortica federated test-connection <server_url>`` — verify FL server reachability
- ``aortica federated server <config.yaml>`` — start the FL server
- ``aortica federated client <config.yaml>`` — start the FL client
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    import click
except ImportError:  # pragma: no cover
    raise SystemExit(
        "Click is required for the Aortica CLI. "
        "Install it with: pip install aortica[cli]"
    )

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False


# ---------------------------------------------------------------------------
# Connection test helper
# ---------------------------------------------------------------------------


def check_fl_connection(
    server_url: str,
    timeout: float = 10.0,
) -> dict[str, object]:
    """Test TCP connectivity to a Flower FL server.

    Parses the ``host:port`` address, opens a TCP socket, and measures
    round-trip latency.

    Args:
        server_url: ``host:port`` string for the FL server.
        timeout: Connection timeout in seconds.

    Returns:
        Dict with keys: ``success`` (bool), ``host`` (str), ``port`` (int),
        ``latency_ms`` (float or None), ``error`` (str or None).
    """
    result: dict[str, object] = {
        "success": False,
        "host": "",
        "port": 0,
        "latency_ms": None,
        "error": None,
    }

    # Parse host:port
    try:
        host, port = _parse_server_url(server_url)
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    result["host"] = host
    result["port"] = port

    # Attempt TCP connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    start = time.monotonic()
    try:
        sock.connect((host, port))
        elapsed = time.monotonic() - start
        result["success"] = True
        result["latency_ms"] = round(elapsed * 1000, 1)
    except socket.timeout:
        result["error"] = (
            f"Connection timed out after {timeout:.0f}s"
        )
    except OSError as exc:
        result["error"] = str(exc)
    finally:
        sock.close()

    return result


def _parse_server_url(server_url: str) -> tuple[str, int]:
    """Parse a ``host:port`` string into (host, port).

    Args:
        server_url: String in ``host:port`` format.

    Returns:
        Tuple of (host, port).

    Raises:
        ValueError: If the format is invalid or port is not a valid number.
    """
    if not server_url or ":" not in server_url:
        raise ValueError(
            f"Invalid server address '{server_url}'. "
            "Expected format: host:port (e.g. 'fl.example.com:8080')"
        )

    # Handle IPv6 addresses like [::1]:8080
    if server_url.startswith("["):
        bracket_end = server_url.find("]")
        if bracket_end == -1:
            raise ValueError(
                f"Invalid IPv6 address '{server_url}'. "
                "Expected format: [host]:port"
            )
        host = server_url[1:bracket_end]
        remainder = server_url[bracket_end + 1 :]
        if not remainder.startswith(":"):
            raise ValueError(
                f"Invalid server address '{server_url}'. "
                "Expected format: [host]:port"
            )
        port_str = remainder[1:]
    else:
        parts = server_url.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid server address '{server_url}'. "
                "Expected format: host:port"
            )
        host, port_str = parts

    if not host:
        raise ValueError("Host cannot be empty")

    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(
            f"Invalid port '{port_str}'. Port must be a number."
        )

    if not (1 <= port <= 65535):
        raise ValueError(
            f"Invalid port {port}. Port must be between 1 and 65535."
        )

    return host, port


# ---------------------------------------------------------------------------
# Config validation helpers
# ---------------------------------------------------------------------------


def _validate_server_config(config_path: str) -> Any:
    """Load and validate an FL server YAML config.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        An ``FLServerConfig`` instance.

    Raises:
        click.ClickException: On file-not-found or validation error.
    """
    from aortica.federated.fl_server import FLServerConfig

    path = Path(config_path)
    if not path.exists():
        raise click.ClickException(f"Config file not found: {path}")

    try:
        return FLServerConfig.from_yaml(path)
    except (ValueError, TypeError) as exc:
        raise click.ClickException(f"Invalid server config: {exc}")


def _validate_client_config(config_path: str) -> Any:
    """Load and validate an FL client YAML config.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        An ``FLClientConfig`` instance.

    Raises:
        click.ClickException: On file-not-found or validation error.
    """
    from aortica.federated.fl_client import FLClientConfig

    path = Path(config_path)
    if not path.exists():
        raise click.ClickException(f"Config file not found: {path}")

    try:
        return FLClientConfig.from_yaml(path)
    except (ValueError, TypeError) as exc:
        raise click.ClickException(f"Invalid client config: {exc}")


def _print_config_summary(
    title: str,
    config_dict: dict[str, Any],
    output_format: str,
) -> None:
    """Print a config summary in text or JSON format."""
    if output_format == "json":
        click.echo(json.dumps(config_dict, indent=2))
        return

    if HAS_RICH:
        console = Console()
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Parameter", style="bold")
        table.add_column("Value")
        for key, val in config_dict.items():
            table.add_row(key, str(val))
        console.print(table)
    else:
        click.echo(f"\n{title}")
        click.echo("=" * len(title))
        for key, val in config_dict.items():
            click.echo(f"  {key}: {val}")
        click.echo()


# ---------------------------------------------------------------------------
# Click command group
# ---------------------------------------------------------------------------


@click.group(name="federated")
def federated_group() -> None:
    """Federated learning commands for collaborative model training."""


# ---------------------------------------------------------------------------
# test-connection
# ---------------------------------------------------------------------------


@federated_group.command(name="test-connection")
@click.argument("server_url")
@click.option(
    "--timeout",
    type=float,
    default=10.0,
    show_default=True,
    help="Connection timeout in seconds.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format: human-readable text or JSON.",
)
def test_connection_cmd(
    server_url: str,
    timeout: float,
    output_format: str,
) -> None:
    """Test connectivity to a Flower federated learning server.

    Verifies that the FL server at SERVER_URL (host:port) is reachable
    and accepting TCP connections. Reports connection latency on success.

    Example:

        aortica federated test-connection fl.example.com:8080
    """
    result = check_fl_connection(server_url, timeout=timeout)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        if not result["success"]:
            sys.exit(1)
        return

    # Rich text output
    if result["success"]:
        latency = result["latency_ms"]
        msg = (
            f"✓ Connection to {server_url} succeeded "
            f"(latency: {latency}ms)\n"
            f"  Server is reachable and accepting connections."
        )
        if HAS_RICH:
            console = Console()
            console.print(f"[bold green]{msg}[/bold green]")
        else:
            click.echo(msg)
    else:
        error = result["error"]
        msg = (
            f"✗ Connection to {server_url} failed\n"
            f"  Error: {error}\n"
            f"\n"
            f"  Troubleshooting:\n"
            f"  - Verify the server address and port\n"
            f"  - Check your firewall allows outbound TCP on the target port\n"
            f"  - Ensure the FL server is running\n"
            f"  - Contact the Coordinator for server status"
        )
        if HAS_RICH:
            console = Console(stderr=True)
            console.print(f"[bold red]{msg}[/bold red]")
        else:
            click.echo(msg, err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------


@federated_group.command(name="server")
@click.argument(
    "config_file",
    type=click.Path(exists=False),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate config without starting the server.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format for --dry-run summary.",
)
def server_cmd(
    config_file: str,
    dry_run: bool,
    output_format: str,
) -> None:
    """Start the Flower federated learning server.

    Loads FL server configuration from CONFIG_FILE (YAML) and starts the
    Flower server with the configured aggregation strategy and number of
    training rounds.

    \b
    YAML config fields:
      num_rounds            Number of federated rounds (default: 5)
      min_fit_clients       Min clients for fit round (default: 2)
      min_evaluate_clients  Min clients for evaluate round (default: 2)
      min_available_clients Min connected clients to start (default: 2)
      server_address        host:port to listen on (default: 0.0.0.0:8080)
      strategy              fedavg | fedprox | scaffold (default: fedavg)
      fraction_fit          Fraction sampled for fit (default: 1.0)
      fraction_evaluate     Fraction sampled for evaluate (default: 1.0)
      log_dir               Directory for metric logs (default: null)

    Example:

        aortica federated server fl_server_config.yaml

        aortica federated server fl_server_config.yaml --dry-run
    """
    config = _validate_server_config(config_file)

    if dry_run:
        if HAS_RICH and output_format == "text":
            console = Console()
            console.print(
                "[bold green]✓ Server config is valid[/bold green]"
            )
        elif output_format == "text":
            click.echo("✓ Server config is valid")

        _print_config_summary(
            "FL Server Configuration",
            config.to_dict(),
            output_format,
        )
        return

    # Actually start the FL server
    from aortica.federated.fl_server import FLServer

    if HAS_RICH:
        console = Console()
        console.print(
            f"[bold cyan]Starting FL server on "
            f"{config.server_address}…[/bold cyan]"
        )
    else:
        click.echo(f"Starting FL server on {config.server_address}…")

    server = FLServer(config)
    server.start()

    # Print summary after completion
    n_rounds = len(server.round_metrics)
    if HAS_RICH:
        console = Console()
        console.print(
            f"[bold green]✓ FL server completed "
            f"{n_rounds} rounds[/bold green]"
        )
    else:
        click.echo(f"✓ FL server completed {n_rounds} rounds")


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


@federated_group.command(name="client")
@click.argument(
    "config_file",
    type=click.Path(exists=False),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate config without starting the client.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format for --dry-run summary.",
)
def client_cmd(
    config_file: str,
    dry_run: bool,
    output_format: str,
) -> None:
    """Start the Flower federated learning client.

    Loads FL client configuration from CONFIG_FILE (YAML) and connects to
    the configured Flower server to participate in federated training rounds.

    \b
    YAML config fields:
      data_path         Path to site-local ECG dataset (required)
      server_address    FL server host:port (default: localhost:8080)
      local_epochs      Local training epochs per round (default: 1)
      batch_size        Training batch size (default: 32)
      lr                Learning rate (default: 0.001)
      sampling_rate     ECG sampling rate in Hz (default: 500)
      window_seconds    Signal window length in seconds (default: 10.0)
      weight_decay      AdamW weight decay (default: 0.0001)
      max_grad_norm     Gradient clipping max norm (default: 1.0)
      enabled_tasks     Task heads to train (default: [rhythm, structural,
                        ischaemia, risk])
      feature_dim       Backbone feature dimension (default: 256)
      head_hidden_dim   Task head hidden dimension (default: 128)
      head_dropout      Task head dropout rate (default: 0.3)
      seed              Random seed (default: 42)
      base_checkpoint   Path to local checkpoint (default: null — uses
                        pretrained from HuggingFace Hub)

    Example:

        aortica federated client fl_client_config.yaml

        aortica federated client fl_client_config.yaml --dry-run
    """
    config = _validate_client_config(config_file)

    if dry_run:
        if HAS_RICH and output_format == "text":
            console = Console()
            console.print(
                "[bold green]✓ Client config is valid[/bold green]"
            )
        elif output_format == "text":
            click.echo("✓ Client config is valid")

        _print_config_summary(
            "FL Client Configuration",
            config.to_dict(),
            output_format,
        )
        return

    # Actually start the FL client
    from aortica.federated.fl_client import AorticaFlowerClient

    if HAS_RICH:
        console = Console()
        console.print(
            f"[bold cyan]Starting FL client, connecting to "
            f"{config.server_address}…[/bold cyan]"
        )
    else:
        click.echo(
            f"Starting FL client, connecting to {config.server_address}…"
        )

    client = AorticaFlowerClient(config)
    client.start()
