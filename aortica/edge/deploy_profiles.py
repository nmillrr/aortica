"""Deployment profiles for edge hardware targets.

Provides dataclass-based configuration profiles for deploying Aortica on
resource-constrained hardware such as Raspberry Pi.  Each profile captures
model path selection, quantisation mode, memory/latency budgets, and
hardware-specific settings.

The primary profile is :class:`RaspberryPiProfile` for ARM64 SBC deployment.

Example::

    from aortica.edge.deploy_profiles import RaspberryPiProfile

    profile = RaspberryPiProfile()
    print(profile.summary())

"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = "~/.cache/aortica"
DEFAULT_EDGE_MODEL_NAME = "aortica_edge_int8.onnx"

SUPPORTED_QUANTIZATIONS = ("INT8", "FP16", "FP32")

# Hardware TDP estimates (watts) for power estimation
HARDWARE_TDP: dict[str, float] = {
    "rpi4": 4.0,
    "rpi5": 6.0,
    "jetson_nano": 5.0,
    "jetson_orin_nano": 7.0,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RaspberryPiProfile:
    """Deployment configuration profile for Raspberry Pi ARM64 devices.

    Attributes:
        model_path: Path to the ONNX edge model.  Defaults to the cached
            INT8 model in ``~/.cache/aortica/``.
        quantization: Quantization level applied to the model.
            One of ``INT8``, ``FP16``, or ``FP32``.
        max_memory_mb: Maximum memory budget in megabytes.  The runtime
            will be configured to stay within this limit.
        target_latency_ms: Target inference latency in milliseconds.
            Used for profiling and validation, not enforced at runtime.
        device_name: Human-readable hardware identifier.
        num_threads: Number of ONNX Runtime intra-op threads.
            Defaults to 4 (RPi4 quad-core).
        enable_sync: Whether the automatic sync scheduler should run.
        sync_interval_minutes: How often the sync scheduler fires.
        data_dir: Directory for local encrypted result storage.
        log_dir: Directory for application logs.
        service_user: Unix user under which the systemd service runs.
        service_group: Unix group for the systemd service.
        watchdog_timeout_sec: systemd watchdog timeout for restart.
    """

    model_path: str = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_EDGE_MODEL_NAME)
    quantization: str = "INT8"
    max_memory_mb: int = 512
    target_latency_ms: int = 350
    device_name: str = "raspberry_pi_4"
    num_threads: int = 4
    enable_sync: bool = True
    sync_interval_minutes: int = 30
    data_dir: str = "/var/lib/aortica/data"
    log_dir: str = "/var/log/aortica"
    service_user: str = "aortica"
    service_group: str = "aortica"
    watchdog_timeout_sec: int = 60
    # ---- Duty-cycling / power optimisation (US-061b) ----
    # When enabled, the ONNX session is loaded on-demand per ECG rather than
    # kept resident, so the device returns to idle draw between acquisitions.
    duty_cycle_enabled: bool = True
    inference_interval_seconds: int = 300
    model_unload_after_seconds: int = 30

    def __post_init__(self) -> None:
        """Validate profile fields on construction."""
        if self.quantization not in SUPPORTED_QUANTIZATIONS:
            raise ValueError(
                f"Unsupported quantization '{self.quantization}'. "
                f"Must be one of {SUPPORTED_QUANTIZATIONS}"
            )
        if self.max_memory_mb <= 0:
            raise ValueError(
                f"max_memory_mb must be positive, got {self.max_memory_mb}"
            )
        if self.target_latency_ms <= 0:
            raise ValueError(
                f"target_latency_ms must be positive, got {self.target_latency_ms}"
            )
        if self.num_threads <= 0:
            raise ValueError(
                f"num_threads must be positive, got {self.num_threads}"
            )
        if self.sync_interval_minutes <= 0:
            raise ValueError(
                f"sync_interval_minutes must be positive, got {self.sync_interval_minutes}"
            )
        if self.watchdog_timeout_sec <= 0:
            raise ValueError(
                f"watchdog_timeout_sec must be positive, got {self.watchdog_timeout_sec}"
            )
        if self.inference_interval_seconds <= 0:
            raise ValueError(
                f"inference_interval_seconds must be positive, "
                f"got {self.inference_interval_seconds}"
            )
        if self.model_unload_after_seconds < 0:
            raise ValueError(
                f"model_unload_after_seconds must be non-negative, "
                f"got {self.model_unload_after_seconds}"
            )

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        """Serialise profile to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RaspberryPiProfile:
        """Deserialise from a dictionary, ignoring unknown keys."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)

    def to_json(self, path: str | Path) -> None:
        """Write profile to a JSON file."""
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> RaspberryPiProfile:
        """Load profile from a JSON file."""
        with open(path) as fh:
            data: dict[str, Any] = json.load(fh)
        return cls.from_dict(data)

    # ---- Helpers ----

    def resolved_model_path(self) -> str:
        """Return the model path with ``~`` expanded."""
        return os.path.expanduser(self.model_path)

    def tdp_watts(self) -> float:
        """Return estimated TDP (watts) for the configured device.

        Falls back to 4.0 W (RPi4 default) for unknown devices.
        """
        return HARDWARE_TDP.get(self.device_name, 4.0)

    def onnxruntime_session_options(self) -> dict[str, Any]:
        """Return ONNX Runtime ``SessionOptions``-compatible settings.

        These can be applied to an ``onnxruntime.SessionOptions`` object
        for consistent edge deployment behaviour.
        """
        return {
            "intra_op_num_threads": self.num_threads,
            "inter_op_num_threads": 1,
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "enable_mem_pattern": True,
            "enable_cpu_mem_arena": self.max_memory_mb > 256,
        }

    def summary(self) -> str:
        """Return a human-readable summary of the profile."""
        lines = [
            "Raspberry Pi Deployment Profile",
            "=" * 40,
            f"  Device:           {self.device_name}",
            f"  Model:            {self.model_path}",
            f"  Quantization:     {self.quantization}",
            f"  Max memory:       {self.max_memory_mb} MB",
            f"  Target latency:   {self.target_latency_ms} ms",
            f"  ORT threads:      {self.num_threads}",
            f"  TDP estimate:     {self.tdp_watts():.1f} W",
            f"  Sync enabled:     {self.enable_sync}",
            f"  Sync interval:    {self.sync_interval_minutes} min",
            f"  Data directory:   {self.data_dir}",
            f"  Log directory:    {self.log_dir}",
            f"  Service user:     {self.service_user}",
            f"  Watchdog timeout: {self.watchdog_timeout_sec} s",
            f"  Duty cycling:     {self.duty_cycle_enabled}",
            f"  ECG interval:     {self.inference_interval_seconds} s",
            f"  Unload after:     {self.model_unload_after_seconds} s idle",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Systemd service file generator
# ---------------------------------------------------------------------------


def generate_systemd_service(
    profile: Optional[RaspberryPiProfile] = None,
) -> str:
    """Generate the content of an ``aortica-edge.service`` systemd unit file.

    Args:
        profile: Deployment profile to use for service parameters.
            Uses default ``RaspberryPiProfile()`` when ``None``.

    Returns:
        String content of the systemd unit file.
    """
    if profile is None:
        profile = RaspberryPiProfile()

    return (
        "[Unit]\n"
        "Description=Aortica Edge ECG Analysis Service\n"
        "After=network.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"User={profile.service_user}\n"
        f"Group={profile.service_group}\n"
        f"Environment=AORTICA_DATA_DIR={profile.data_dir}\n"
        f"Environment=AORTICA_LOG_DIR={profile.log_dir}\n"
        f"Environment=ORT_NUM_THREADS={profile.num_threads}\n"
        f"ExecStart=/usr/bin/env aortica predict --watch {profile.data_dir}/inbox"
        f" --model {profile.resolved_model_path()}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        f"WatchdogSec={profile.watchdog_timeout_sec}\n"
        "\n"
        "# Resource limits\n"
        f"MemoryMax={profile.max_memory_mb}M\n"
        "CPUQuota=90%\n"
        "\n"
        "# Security hardening\n"
        "ProtectSystem=strict\n"
        "ProtectHome=read-only\n"
        f"ReadWritePaths={profile.data_dir} {profile.log_dir}\n"
        "NoNewPrivileges=yes\n"
        "PrivateTmp=yes\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


# ---------------------------------------------------------------------------
# Shell script generator
# ---------------------------------------------------------------------------


def generate_pi_image_script(
    profile: Optional[RaspberryPiProfile] = None,
) -> str:
    """Generate the content of ``create_pi_image_script.sh``.

    The script installs Python, creates a virtualenv, installs
    ``aortica[cli,edge]``, downloads the edge model, sets up directories,
    installs the systemd service, and enables it for auto-start.

    Args:
        profile: Deployment profile to use.  Defaults to
            ``RaspberryPiProfile()``.

    Returns:
        Shell script content as a string.
    """
    if profile is None:
        profile = RaspberryPiProfile()

    service_content = generate_systemd_service(profile).rstrip()
    lines = [
        "#!/usr/bin/env bash",
        "# ----------------------------------------------------------------",
        "# Aortica Edge \u2014 Raspberry Pi Image Setup Script",
        "# Generated by aortica.edge.deploy_profiles",
        "# ----------------------------------------------------------------",
        "set -euo pipefail",
        "",
        f'echo "=== Aortica Edge Setup for {profile.device_name} ==="',
        "",
        "# ---- 1. System dependencies ----",
        'echo "[1/7] Installing system dependencies..."',
        "sudo apt-get update -qq",
        "sudo apt-get install -y -qq python3 python3-pip python3-venv libatlas-base-dev",
        "",
        "# ---- 2. Create service user ----",
        f'echo "[2/7] Creating service user \'{profile.service_user}\'..."',
        f"if ! id -u {profile.service_user} > /dev/null 2>&1; then",
        f"    sudo useradd --system --create-home --shell /usr/sbin/nologin {profile.service_user}",
        "fi",
        "",
        "# ---- 3. Python virtual environment ----",
        'echo "[3/7] Setting up Python virtual environment..."',
        'VENV_DIR="/opt/aortica/venv"',
        "sudo mkdir -p /opt/aortica",
        'sudo python3 -m venv "$VENV_DIR"',
        'sudo "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel',
        "",
        "# ---- 4. Install Aortica ----",
        'echo "[4/7] Installing aortica[cli,edge]..."',
        'sudo "$VENV_DIR/bin/pip" install "aortica[cli,edge]"',
        'sudo ln -sf "$VENV_DIR/bin/aortica" /usr/local/bin/aortica',
        "",
        "# ---- 5. Download edge model ----",
        'echo "[5/7] Downloading INT8 edge model..."',
        f'sudo -u {profile.service_user} "$VENV_DIR/bin/python" -c \\',
        '    "from aortica.models.registry import load_pretrained;'
        " load_pretrained('latest', variant='edge')\"",
        "",
        "# ---- 6. Create data/log directories ----",
        'echo "[6/7] Creating data and log directories..."',
        f"sudo mkdir -p {profile.data_dir}/inbox",
        f"sudo mkdir -p {profile.log_dir}",
        f"sudo chown -R {profile.service_user}:{profile.service_group} {profile.data_dir}",
        f"sudo chown -R {profile.service_user}:{profile.service_group} {profile.log_dir}",
        "",
        "# ---- 7. Install systemd service ----",
        'echo "[7/7] Installing systemd service..."',
        "cat > /tmp/aortica-edge.service << 'SYSTEMD_EOF'",
        service_content,
        "SYSTEMD_EOF",
        "sudo mv /tmp/aortica-edge.service /etc/systemd/system/aortica-edge.service",
        "sudo systemctl daemon-reload",
        "sudo systemctl enable aortica-edge.service",
        "",
        'echo ""',
        'echo "=== Setup complete ==="',
        'echo "Start the service:  sudo systemctl start aortica-edge"',
        'echo "Check status:       sudo systemctl status aortica-edge"',
        'echo "View logs:          sudo journalctl -u aortica-edge -f"',
        'echo ""',
        f'echo "Drop ECG files into {profile.data_dir}/inbox for analysis."',
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def write_systemd_service(
    output_path: str | Path,
    profile: Optional[RaspberryPiProfile] = None,
) -> Path:
    """Write the systemd service file to disk.

    Args:
        output_path: Destination file path.
        profile: Deployment profile.  Defaults to ``RaspberryPiProfile()``.

    Returns:
        Resolved ``Path`` of the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_systemd_service(profile))
    return path.resolve()


def write_pi_image_script(
    output_path: str | Path,
    profile: Optional[RaspberryPiProfile] = None,
) -> Path:
    """Write the Pi image setup script to disk and make it executable.

    Args:
        output_path: Destination file path.
        profile: Deployment profile.  Defaults to ``RaspberryPiProfile()``.

    Returns:
        Resolved ``Path`` of the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_pi_image_script(profile))
    path.chmod(0o755)
    return path.resolve()


# ---------------------------------------------------------------------------
# Duty-cycled on-demand model loader (US-061b power optimisation)
# ---------------------------------------------------------------------------


class DutyCycledModelLoader:
    """Load an edge model on-demand and release it after idle to save power.

    Rather than keeping an ONNX session resident (which holds memory and keeps
    the device from returning to its lowest idle state), the loader materialises
    the model only when an ECG needs inference and releases it after
    ``unload_after_seconds`` of inactivity.  This implements the duty-cycling
    power optimisation for rural / battery-constrained deployments.

    The loader is backend-agnostic: it takes a ``loader`` callable that returns
    a model/session object, so it can wrap ``onnxruntime.InferenceSession`` in
    production and a stub in tests.

    Example::

        import onnxruntime as ort
        loader = DutyCycledModelLoader(lambda: ort.InferenceSession("m.onnx"))
        with loader.session() as sess:
            sess.run(None, {...})
        # session released here; reloaded lazily on next use

    Args:
        loader: Zero-argument callable returning the loaded model/session.
        unload_after_seconds: Idle seconds after which :meth:`maybe_unload`
            releases the session.  ``0`` releases immediately when the context
            manager exits.
        time_fn: Injectable clock (defaults to :func:`time.monotonic`) for tests.
    """

    def __init__(
        self,
        loader: Callable[[], Any],
        unload_after_seconds: float = 30.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if unload_after_seconds < 0:
            raise ValueError(
                f"unload_after_seconds must be non-negative, got {unload_after_seconds}"
            )
        self._loader = loader
        self._unload_after_seconds = unload_after_seconds
        self._time_fn = time_fn
        self._model: Optional[Any] = None
        self._last_used: float = 0.0
        self.load_count: int = 0

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently resident in memory."""
        return self._model is not None

    def load(self) -> Any:
        """Load the model if not already resident and return it."""
        if self._model is None:
            self._model = self._loader()
            self.load_count += 1
        self._last_used = self._time_fn()
        return self._model

    def release(self) -> None:
        """Release the resident model, allowing the device to return to idle."""
        self._model = None

    def maybe_unload(self) -> bool:
        """Release the model if it has been idle beyond the unload threshold.

        Returns:
            ``True`` if the model was released, ``False`` otherwise.
        """
        if self._model is None:
            return False
        idle = self._time_fn() - self._last_used
        if idle >= self._unload_after_seconds:
            self.release()
            return True
        return False

    def session(self) -> "_LoadedModelContext":
        """Context manager that loads the model on entry and unloads on exit."""
        return _LoadedModelContext(self)


class _LoadedModelContext:
    """Context manager returned by :meth:`DutyCycledModelLoader.session`."""

    def __init__(self, loader: DutyCycledModelLoader) -> None:
        self._loader = loader

    def __enter__(self) -> Any:
        return self._loader.load()

    def __exit__(self, *exc: Any) -> None:
        # Immediate unload when threshold is 0; otherwise defer to maybe_unload.
        if self._loader._unload_after_seconds == 0:
            self._loader.release()
        else:
            self._loader.maybe_unload()
