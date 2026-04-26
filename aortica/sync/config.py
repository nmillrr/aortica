"""Sync configuration, connectivity checking, bandwidth management, and anonymisation.

Provides a YAML-loadable :class:`SyncConfig` for controlling automatic
sync behaviour, a :func:`check_connectivity` helper for probing network
availability and estimating bandwidth, and an :func:`anonymise_result`
function that strips patient metadata before transmission.

Usage::

    from aortica.sync.config import SyncConfig, check_connectivity, anonymise_result

    cfg = SyncConfig.from_yaml("sync_config.yaml")
    status = check_connectivity(cfg.remote_url)
    if status.available and status.bandwidth_kbps >= cfg.min_bandwidth_kbps:
        # safe to sync
        ...

"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

# Default keys stripped from patient metadata during anonymisation.
_DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "patient_name",
        "patient_id",
        "name",
        "mrn",
        "medical_record_number",
        "date_of_birth",
        "dob",
        "address",
        "phone",
        "email",
        "ssn",
        "social_security_number",
        "insurance_id",
        "insurance",
    }
)


@dataclass
class SyncConfig:
    """Configuration for the automatic sync engine.

    Parameters
    ----------
    sync_interval_minutes:
        How often (in minutes) the auto-sync scheduler fires.
    min_bandwidth_kbps:
        Minimum estimated bandwidth (Kbps) required to proceed with sync.
    max_batch_size:
        Maximum number of results per sync batch.
    remote_url:
        Base URL of the central sync server.
    device_id:
        Unique identifier for this device / node.
    sensitive_keys:
        Extra metadata keys considered sensitive and stripped during
        anonymisation (merged with built-in defaults).
    """

    sync_interval_minutes: int = 30
    min_bandwidth_kbps: int = 256
    max_batch_size: int = 20
    remote_url: str = ""
    device_id: str = ""
    sensitive_keys: list[str] = field(default_factory=list)

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the config to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncConfig:
        """Create a :class:`SyncConfig` from a dict (e.g. parsed YAML)."""
        known_fields = {
            "sync_interval_minutes",
            "min_bandwidth_kbps",
            "max_batch_size",
            "remote_url",
            "device_id",
            "sensitive_keys",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    # -- YAML I/O -------------------------------------------------------------

    def to_yaml(self, path: str | Path) -> None:
        """Write the config to a YAML file.

        Falls back to a simple key-value writer when ``pyyaml`` is not
        installed.
        """
        try:
            import yaml  # type: ignore[import-untyped]

            with open(path, "w") as fh:
                yaml.safe_dump(self.to_dict(), fh, default_flow_style=False)
        except ImportError:
            # Minimal YAML-compatible writer (flat dict only)
            with open(path, "w") as fh:
                for key, value in self.to_dict().items():
                    if isinstance(value, list):
                        fh.write(f"{key}:\n")
                        for item in value:
                            fh.write(f"  - {item}\n")
                    else:
                        fh.write(f"{key}: {value}\n")

    @classmethod
    def from_yaml(cls, path: str | Path) -> SyncConfig:
        """Load a :class:`SyncConfig` from a YAML file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            import yaml  # type: ignore[import-untyped]

            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
        except ImportError:
            # Minimal YAML parser for flat configs (no nested structures
            # beyond simple lists).
            data = _simple_yaml_load(path)

        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Connectivity probe
# ---------------------------------------------------------------------------


@dataclass
class ConnectivityStatus:
    """Result of a connectivity probe."""

    available: bool
    latency_ms: float = 0.0
    bandwidth_kbps: float = 0.0
    error: str = ""


def check_connectivity(
    url: str,
    *,
    timeout: float = 10.0,
    probe_bytes: int = 4096,
) -> ConnectivityStatus:
    """Test network availability and estimate bandwidth to *url*.

    Sends a small ``GET`` request and measures the response time to
    derive a rough bandwidth estimate.

    Parameters
    ----------
    url:
        URL to probe (typically the remote sync server's ``/health``
        endpoint).
    timeout:
        Connection timeout in seconds.
    probe_bytes:
        Number of bytes to use for bandwidth estimation.  Defaults to
        4 KiB.
    """
    start = time.monotonic()
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            elapsed = time.monotonic() - start

            latency_ms = elapsed * 1000.0

            # Bandwidth estimation: bytes received / elapsed time
            received_bytes = len(body) if body else probe_bytes
            if elapsed > 0:
                bandwidth_kbps = (received_bytes * 8) / (elapsed * 1000)
            else:
                bandwidth_kbps = float("inf")

            return ConnectivityStatus(
                available=True,
                latency_ms=latency_ms,
                bandwidth_kbps=bandwidth_kbps,
            )
    except urllib.error.HTTPError as exc:
        # Server responded but with an error status — still "available"
        elapsed = time.monotonic() - start
        return ConnectivityStatus(
            available=True,
            latency_ms=elapsed * 1000.0,
            bandwidth_kbps=0.0,
            error=f"HTTP {exc.code}",
        )
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return ConnectivityStatus(
            available=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Auto-sync scheduler
# ---------------------------------------------------------------------------


class AutoSyncScheduler:
    """Periodically runs a sync function respecting bandwidth thresholds.

    The scheduler runs on a background daemon thread and fires at the
    interval specified in :class:`SyncConfig`.

    Parameters
    ----------
    config:
        Sync configuration controlling interval and bandwidth threshold.
    sync_fn:
        A callable ``(remote_url: str) -> Any`` that performs the actual
        sync (typically ``SyncEngine.sync_to_remote``).
    connectivity_url:
        URL to probe for connectivity (defaults to
        ``config.remote_url + '/health'``).
    """

    def __init__(
        self,
        config: SyncConfig,
        sync_fn: Callable[[str], Any],
        connectivity_url: str | None = None,
    ) -> None:
        self._config = config
        self._sync_fn = sync_fn
        self._connectivity_url = connectivity_url or f"{config.remote_url}/health"
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._last_sync_time: Optional[float] = None
        self._last_status: Optional[ConnectivityStatus] = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        """Return ``True`` if the scheduler is active."""
        return self._running

    @property
    def last_sync_time(self) -> Optional[float]:
        """Epoch timestamp of the last successful sync attempt."""
        return self._last_sync_time

    @property
    def last_status(self) -> Optional[ConnectivityStatus]:
        """The connectivity status from the most recent check."""
        return self._last_status

    def start(self) -> None:
        """Start the scheduler (idempotent)."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._schedule_next()

    def stop(self) -> None:
        """Stop the scheduler and cancel any pending timer."""
        with self._lock:
            self._running = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def tick(self) -> Optional[ConnectivityStatus]:
        """Execute one sync check cycle (public for testing).

        Returns the :class:`ConnectivityStatus` from the probe, or
        ``None`` if sync was skipped due to missing configuration.
        """
        if not self._config.remote_url:
            return None

        status = check_connectivity(self._connectivity_url)
        self._last_status = status

        if status.available and status.bandwidth_kbps >= self._config.min_bandwidth_kbps:
            try:
                self._sync_fn(self._config.remote_url)
                self._last_sync_time = time.time()
            except Exception:  # noqa: BLE001
                pass  # errors are logged inside SyncEngine

        return status

    # -- internal -------------------------------------------------------------

    def _schedule_next(self) -> None:
        """Schedule the next tick."""
        if not self._running:
            return
        interval_seconds = self._config.sync_interval_minutes * 60
        self._timer = threading.Timer(interval_seconds, self._run_cycle)
        self._timer.daemon = True
        self._timer.start()

    def _run_cycle(self) -> None:
        """Called by the timer thread — run one tick then reschedule."""
        self.tick()
        with self._lock:
            if self._running:
                self._schedule_next()


# ---------------------------------------------------------------------------
# Anonymisation
# ---------------------------------------------------------------------------


def anonymise_result(
    result_data: dict[str, Any],
    *,
    extra_sensitive_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Return a deep copy of *result_data* with patient metadata stripped.

    Removes keys from the ``metadata`` sub-dict that match the built-in
    sensitive key list or any additional keys in *extra_sensitive_keys*.

    The function is case-insensitive when matching key names.

    Parameters
    ----------
    result_data:
        A dict representing a sync payload (ecg_hash, predictions,
        quality, metadata, …).
    extra_sensitive_keys:
        Additional metadata keys to strip (merged with defaults).
    """
    # Build the full set of sensitive keys (lowercased for matching)
    sensitive = {k.lower() for k in _DEFAULT_SENSITIVE_KEYS}
    if extra_sensitive_keys:
        sensitive |= {k.lower() for k in extra_sensitive_keys}

    # Deep copy to avoid mutating the original
    result: dict[str, Any] = json.loads(json.dumps(result_data))

    # Strip from top-level metadata dict
    if "metadata" in result and isinstance(result["metadata"], dict):
        result["metadata"] = {
            k: v
            for k, v in result["metadata"].items()
            if k.lower() not in sensitive
        }

    # Also strip from nested patient_metadata if present
    if "patient_metadata" in result and isinstance(result["patient_metadata"], dict):
        result["patient_metadata"] = {
            k: v
            for k, v in result["patient_metadata"].items()
            if k.lower() not in sensitive
        }

    return result


# ---------------------------------------------------------------------------
# Simple YAML fallback parser
# ---------------------------------------------------------------------------


def _simple_yaml_load(path: Path) -> dict[str, Any]:
    """Minimal YAML parser for flat configs with optional simple lists.

    Handles:
        key: value
        key:
          - item1
          - item2
    """
    data: dict[str, Any] = {}
    current_list_key: Optional[str] = None

    with open(path) as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            # Skip comments and blank lines
            if not stripped or stripped.lstrip().startswith("#"):
                current_list_key = None
                continue

            # List item continuation
            if current_list_key and stripped.startswith("  - "):
                value_str = stripped[4:].strip()
                data[current_list_key].append(_coerce_value(value_str))
                continue

            # Key: value pairs
            if ":" in stripped:
                key, _, value_part = stripped.partition(":")
                key = key.strip()
                value_part = value_part.strip()

                if value_part == "" or value_part == "[]":
                    # Empty value — may be a list header or empty string
                    if value_part == "[]":
                        data[key] = []
                        current_list_key = None
                    else:
                        data[key] = []
                        current_list_key = key
                else:
                    data[key] = _coerce_value(value_part)
                    current_list_key = None

    return data


def _coerce_value(s: str) -> Any:
    """Coerce a YAML-style string value to an appropriate Python type."""
    # Booleans
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    # Integers
    try:
        return int(s)
    except ValueError:
        pass
    # Floats
    try:
        return float(s)
    except ValueError:
        pass
    # Strip surrounding quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s
