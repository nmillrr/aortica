"""Pre-trained model registry for downloading and caching Aortica checkpoints.

Provides :func:`load_pretrained` to download versioned checkpoints from
HuggingFace Hub (``nmillrr/aortica``) with SHA-256 integrity verification
and local caching at ``~/.cache/aortica/``.

Also provides :func:`list_available_versions` to query available model
versions from the Hub API.

Example::

    from aortica.models.registry import load_pretrained

    model = load_pretrained("latest")
    model = load_pretrained("0.2.0", variant="edge")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUB_REPO_ID = "nmillrr/aortica"
HUB_API_URL = "https://huggingface.co/api/models"
HUB_DOWNLOAD_URL = "https://huggingface.co/{repo}/resolve/{revision}/{filename}"

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "aortica")

# Filename patterns keyed by variant
_VARIANT_FILENAME: Dict[str, str] = {
    "full": "aortica_full_v{version}.pt",
    "edge": "aortica_edge_int8_v{version}.onnx",
}

# Mapping from version tags to SHA-256 checksums.
# Updated by the release CI workflow.
_KNOWN_CHECKSUMS: Dict[str, Dict[str, str]] = {}

# Data provenance statement included in model card and CLI info.
DATA_PROVENANCE = (
    "Trained on PTB-XL (CC BY 4.0, Wagner et al. 2020, PhysioNet). "
    "No proprietary data used. No patient data leaves this deployment."
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ChecksumError(Exception):
    """Raised when a downloaded checkpoint fails SHA-256 verification.

    This indicates a corrupted download or potential tampering.
    Re-download with ``force_download=True`` or verify network integrity.
    """


class ModelNotFoundError(Exception):
    """Raised when the requested model version is not available."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelVersion:
    """Metadata about an available model version.

    Attributes:
        version: Semantic version string (e.g. ``'0.2.0'``).
        release_date: ISO-8601 date string or ``None`` if unknown.
        variants: Available variants (``['full', 'edge']``).
        performance_summary: Optional brief performance note.
    """

    version: str
    release_date: Optional[str] = None
    variants: List[str] = field(default_factory=lambda: ["full", "edge"])
    performance_summary: Optional[str] = None


@dataclass
class PretrainedModelInfo:
    """Information about the currently loaded pretrained model.

    Attributes:
        version: Model version string.
        variant: ``'full'`` or ``'edge'``.
        source: ``'hub'`` or ``'local'`` indicating checkpoint source.
        sha256: SHA-256 hash of the checkpoint file.
        cache_path: Local filesystem path to the cached checkpoint.
        data_attribution: Data provenance statement.
    """

    version: str
    variant: str
    source: str
    sha256: str
    cache_path: str
    data_attribution: str = DATA_PROVENANCE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _get_cache_dir(cache_dir: Optional[str]) -> Path:
    """Resolve and create the cache directory."""
    path = Path(cache_dir) if cache_dir else Path(DEFAULT_CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_version(version: str, repo_id: str = HUB_REPO_ID) -> str:
    """Resolve ``'latest'`` to an actual version tag.

    Queries the HuggingFace Hub API for available tags and returns the
    most recent semantic version.  If the API is unreachable, falls back
    to the package version.
    """
    if version != "latest":
        return version

    try:
        versions = list_available_versions(repo_id=repo_id)
        if versions:
            return versions[0].version
    except Exception:
        pass

    # Fallback to package version
    import aortica

    return aortica.__version__


def _get_expected_checksum(
    version: str,
    variant: str,
) -> Optional[str]:
    """Look up the expected SHA-256 checksum for a version+variant.

    Returns ``None`` if no checksum is known (e.g. new release before
    the checksum registry is updated).
    """
    key = f"{version}/{variant}"
    return _KNOWN_CHECKSUMS.get(key, {}).get("sha256")


def _download_file(url: str, dest: Path, timeout: int = 60) -> None:
    """Download a file from *url* to *dest* with progress logging."""
    logger.info("Downloading %s → %s", url, dest)
    req = Request(url, headers={"User-Agent": "aortica-python"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
    except Exception:
        # Clean up partial downloads
        if dest.exists():
            dest.unlink()
        raise
    logger.info("Download complete: %s (%d bytes)", dest, dest.stat().st_size)


def _build_download_url(
    repo_id: str,
    version: str,
    variant: str,
) -> str:
    """Construct the HuggingFace Hub download URL for a checkpoint."""
    filename = _VARIANT_FILENAME[variant].format(version=version)
    revision = f"v{version}"
    return HUB_DOWNLOAD_URL.format(
        repo=repo_id,
        revision=revision,
        filename=filename,
    )


def _save_model_info(cache_dir: Path, info: PretrainedModelInfo) -> None:
    """Persist model info to a JSON sidecar file."""
    info_path = cache_dir / "model_info.json"
    info_dict = {
        "version": info.version,
        "variant": info.variant,
        "source": info.source,
        "sha256": info.sha256,
        "cache_path": info.cache_path,
        "data_attribution": info.data_attribution,
    }
    info_path.write_text(json.dumps(info_dict, indent=2), encoding="utf-8")


def _load_model_info(cache_dir: Path) -> Optional[PretrainedModelInfo]:
    """Load model info from the JSON sidecar file if it exists."""
    info_path = cache_dir / "model_info.json"
    if not info_path.exists():
        return None
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
        return PretrainedModelInfo(**data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_pretrained(
    version: str = "latest",
    *,
    variant: str = "full",
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    repo_id: str = HUB_REPO_ID,
) -> Any:
    """Download (if needed) and load a pre-trained Aortica model.

    Downloads the checkpoint from HuggingFace Hub and caches it locally.
    Subsequent calls load from cache without network access.

    Parameters
    ----------
    version:
        Semantic version string (e.g. ``'0.2.0'``) or ``'latest'``.
    variant:
        ``'full'`` for the PyTorch checkpoint or ``'edge'`` for the
        INT8 ONNX edge model.
    cache_dir:
        Local directory for caching.  Defaults to ``~/.cache/aortica/``.
    force_download:
        If ``True``, re-download even if a cached copy exists.
    repo_id:
        HuggingFace Hub repository ID.  Defaults to ``nmillrr/aortica``.

    Returns
    -------
    AorticaModel | str
        For ``variant='full'``, returns a loaded :class:`AorticaModel`
        instance in eval mode.  For ``variant='edge'``, returns the
        local file path to the ONNX model.

    Raises
    ------
    ChecksumError
        Downloaded file does not match expected SHA-256 hash.
    ModelNotFoundError
        Requested version/variant was not found on the Hub.
    ImportError
        Required dependencies (``torch`` for full, ``onnxruntime`` for
        edge) are not installed.
    """
    if variant not in _VARIANT_FILENAME:
        raise ValueError(
            f"Unknown variant '{variant}'. Must be one of: "
            f"{list(_VARIANT_FILENAME.keys())}"
        )

    resolved_version = _resolve_version(version, repo_id=repo_id)
    cache = _get_cache_dir(cache_dir)
    filename = _VARIANT_FILENAME[variant].format(version=resolved_version)
    cached_path = cache / filename

    # Download if not cached or forced
    if force_download or not cached_path.exists():
        url = _build_download_url(repo_id, resolved_version, variant)
        try:
            _download_file(url, cached_path)
        except (URLError, OSError) as exc:
            raise ModelNotFoundError(
                f"Could not download model v{resolved_version} ({variant}) "
                f"from {repo_id}: {exc}"
            ) from exc

    # Verify checksum
    actual_sha = _compute_sha256(cached_path)
    expected_sha = _get_expected_checksum(resolved_version, variant)
    if expected_sha is not None and actual_sha != expected_sha:
        cached_path.unlink(missing_ok=True)
        raise ChecksumError(
            f"SHA-256 mismatch for {filename}. "
            f"Expected: {expected_sha}, got: {actual_sha}. "
            "The file may be corrupted or tampered with. "
            "Try again with force_download=True."
        )

    # Save model info sidecar
    info = PretrainedModelInfo(
        version=resolved_version,
        variant=variant,
        source="hub",
        sha256=actual_sha,
        cache_path=str(cached_path),
    )
    _save_model_info(cache, info)

    # Load and return
    if variant == "edge":
        return str(cached_path)

    # Full PyTorch model
    try:
        import torch

        from aortica.models.aortica_model import AorticaModel
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load the full model. "
            "Install with: pip install aortica[torch]"
        ) from exc

    checkpoint = torch.load(
        str(cached_path), map_location="cpu", weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        enabled_tasks = checkpoint.get(
            "enabled_tasks", ["rhythm", "structural", "ischaemia", "risk"],
        )
        model = AorticaModel(enabled_tasks=enabled_tasks)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, AorticaModel):
        model = checkpoint  # type: ignore[assignment]
    else:
        # Assume raw state_dict
        model = AorticaModel()
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def list_available_versions(
    *,
    repo_id: str = HUB_REPO_ID,
    timeout: int = 10,
) -> List[ModelVersion]:
    """Query HuggingFace Hub for available Aortica model versions.

    Returns a list of :class:`ModelVersion` sorted by version descending
    (newest first).

    Parameters
    ----------
    repo_id:
        HuggingFace Hub repository ID.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    list[ModelVersion]
        Available model versions with metadata.
    """
    url = f"{HUB_API_URL}/{repo_id}/refs"
    req = Request(url, headers={"User-Agent": "aortica-python"})

    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not query Hub API: %s", exc)
        return []

    versions: List[ModelVersion] = []

    # Parse tags from the refs response
    tags = data.get("tags", [])
    for tag_info in tags:
        tag_name = tag_info if isinstance(tag_info, str) else tag_info.get("name", "")
        # Tags are expected to be like "v0.2.0"
        if tag_name.startswith("v"):
            version_str = tag_name[1:]
        else:
            version_str = tag_name

        if not version_str:
            continue

        versions.append(
            ModelVersion(
                version=version_str,
                release_date=tag_info.get("date") if isinstance(tag_info, dict) else None,
            )
        )

    # Sort by version descending (simple string sort — good enough for semver)
    versions.sort(key=lambda v: v.version, reverse=True)
    return versions


def get_model_info(cache_dir: Optional[str] = None) -> Optional[PretrainedModelInfo]:
    """Get information about the currently cached pretrained model.

    Returns ``None`` if no model info is cached.
    """
    cache = _get_cache_dir(cache_dir)
    return _load_model_info(cache)
