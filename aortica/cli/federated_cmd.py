"""CLI commands for Aortica federated learning operations.

Provides ``aortica federated test-connection <server_url>`` for verifying
FL server reachability before joining a training round.
"""

from __future__ import annotations

import socket
import sys
import time
from typing import Optional

try:
    import click
except ImportError:  # pragma: no cover
    raise SystemExit(
        "Click is required for the Aortica CLI. "
        "Install it with: pip install aortica[cli]"
    )

try:
    from rich.console import Console

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
# Click command group
# ---------------------------------------------------------------------------


@click.group(name="federated")
def federated_group() -> None:
    """Federated learning commands for collaborative model training."""


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
        import json

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
