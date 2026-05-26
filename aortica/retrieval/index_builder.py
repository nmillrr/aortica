"""Latent space index construction for case-based ECG retrieval.

Encodes ECGs through the :class:`~aortica.models.AorticaModel` backbone,
extracts feature vectors, and builds an approximate nearest-neighbor (ANN)
index for fast similarity search.

Supports **Annoy** (default) and **FAISS** backends.  A JSON metadata sidecar
maps each index entry to anonymized record ID, verified diagnoses, and
demographic information.

Example::

    from aortica.retrieval import build_index

    report = build_index(
        model=model,
        dataset=ecg_records,
        output_path="./index",
        labels=label_list,
    )
    print(report)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IndexMetadataEntry:
    """Metadata for a single ECG in the index.

    Attributes:
        index_id: Position in the ANN index.
        record_id: Anonymised record identifier.
        diagnoses: Verified diagnosis labels from the dataset.
        age: Patient age (if available).
        sex: Patient sex (if available).
    """

    index_id: int
    record_id: str
    diagnoses: list[str] = field(default_factory=list)
    age: Optional[int] = None
    sex: Optional[str] = None


@dataclass
class IndexBuildReport:
    """Report produced by :func:`build_index`.

    Attributes:
        index_path: Filesystem path to the saved index file.
        metadata_path: Filesystem path to the JSON metadata sidecar.
        num_vectors: Total number of feature vectors indexed.
        feature_dim: Dimensionality of each feature vector.
        backend: ANN backend used (``"annoy"`` or ``"faiss"``).
        num_trees: Number of trees (Annoy) or nlist (FAISS).
        build_time_seconds: Wall-clock time to build the index.
    """

    index_path: str
    metadata_path: str
    num_vectors: int
    feature_dim: int
    backend: str
    num_trees: int
    build_time_seconds: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_features(
    model: Any,
    ecg_signals: NDArray[np.float64],
    batch_size: int = 64,
) -> NDArray[np.float64]:
    """Run *ecg_signals* through the model backbone and return feature vectors.

    Args:
        model: An :class:`~aortica.models.AorticaModel` (PyTorch ``nn.Module``).
        ecg_signals: Array of shape ``[N, leads, samples]``.
        batch_size: Number of ECGs to feed per forward pass.

    Returns:
        Feature matrix of shape ``[N, feature_dim]``.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for feature extraction. "
            "Install with: pip install aortica[torch]"
        ) from exc

    model.eval()
    device = next(model.parameters()).device

    all_features: list[NDArray[np.float64]] = []
    n_samples = ecg_signals.shape[0]

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_np = ecg_signals[start:end]
            batch_t = torch.tensor(batch_np, dtype=torch.float32, device=device)

            # Use the backbone + attention to get feature vectors
            features = model.backbone(batch_t)
            features = model.attention(features)
            all_features.append(features.cpu().numpy().astype(np.float64))

    return np.concatenate(all_features, axis=0)


def _build_annoy_index(
    vectors: NDArray[np.float64],
    output_path: Path,
    num_trees: int,
    metric: str,
) -> str:
    """Build an Annoy index and save to disk.

    Returns:
        Path to the saved index file.
    """
    try:
        from annoy import AnnoyIndex  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "annoy is required for ANN index construction. "
            "Install with: pip install annoy"
        ) from exc

    feature_dim = vectors.shape[1]
    index = AnnoyIndex(feature_dim, metric)

    for i in range(vectors.shape[0]):
        index.add_item(i, vectors[i].tolist())

    index.build(num_trees)
    index_file = str(output_path / "index.ann")
    index.save(index_file)
    return index_file


def _build_faiss_index(
    vectors: NDArray[np.float64],
    output_path: Path,
    nlist: int,
) -> str:
    """Build a FAISS index and save to disk.

    Returns:
        Path to the saved index file.
    """
    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "faiss is required when backend='faiss'. "
            "Install with: pip install faiss-cpu"
        ) from exc

    feature_dim = vectors.shape[1]
    vectors_f32 = vectors.astype(np.float32)

    # Use IVFFlat for large datasets, Flat for small ones
    if vectors.shape[0] < 1000:
        index = faiss.IndexFlatL2(feature_dim)
    else:
        effective_nlist = min(nlist, vectors.shape[0] // 2)
        quantiser = faiss.IndexFlatL2(feature_dim)
        index = faiss.IndexIVFFlat(quantiser, feature_dim, effective_nlist)
        index.train(vectors_f32)

    index.add(vectors_f32)
    index_file = str(output_path / "index.faiss")
    faiss.write_index(index, index_file)
    return index_file


def _build_metadata(
    num_vectors: int,
    record_ids: Optional[Sequence[str]],
    labels: Optional[Sequence[Sequence[str]]],
    demographics: Optional[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build the metadata sidecar list."""
    entries: list[dict[str, Any]] = []
    for i in range(num_vectors):
        record_id = record_ids[i] if record_ids is not None else f"record_{i:06d}"
        diagnoses: list[str] = list(labels[i]) if labels is not None else []
        age: Optional[int] = None
        sex: Optional[str] = None
        if demographics is not None and i < len(demographics):
            demo = demographics[i]
            age = demo.get("age")  # type: ignore[assignment]
            sex = demo.get("sex")  # type: ignore[assignment]

        entry = IndexMetadataEntry(
            index_id=i,
            record_id=record_id,
            diagnoses=diagnoses,
            age=age,
            sex=sex,
        )
        entries.append(asdict(entry))
    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_index(
    model: Any,
    dataset: NDArray[np.float64],
    output_path: str | Path,
    *,
    labels: Optional[Sequence[Sequence[str]]] = None,
    record_ids: Optional[Sequence[str]] = None,
    demographics: Optional[Sequence[dict[str, Any]]] = None,
    backend: str = "annoy",
    num_trees: int = 100,
    metric: str = "angular",
    batch_size: int = 64,
) -> IndexBuildReport:
    """Encode ECGs through the model backbone and build an ANN index.

    Args:
        model: An :class:`~aortica.models.AorticaModel` instance.
        dataset: ECG signal array of shape ``[N, leads, samples]``.
        output_path: Directory to save the index and metadata sidecar.
        labels: Per-ECG list of verified diagnosis label strings.
        record_ids: Per-ECG anonymised record identifiers.  If ``None``,
            auto-generated as ``record_000000``, ``record_000001``, etc.
        demographics: Per-ECG dicts with optional ``"age"`` (int) and
            ``"sex"`` (str) keys.
        backend: ANN library to use: ``"annoy"`` (default) or ``"faiss"``.
        num_trees: Number of Annoy trees (or FAISS *nlist*).  Default ``100``.
        metric: Distance metric for Annoy.  Default ``"angular"``.
        batch_size: Batch size for feature extraction.  Default ``64``.

    Returns:
        :class:`IndexBuildReport` with paths and summary statistics.

    Raises:
        ImportError: If the chosen ANN library is not installed.
        ValueError: If *backend* is not ``"annoy"`` or ``"faiss"``.
    """
    if backend not in ("annoy", "faiss"):
        raise ValueError(f"backend must be 'annoy' or 'faiss', got '{backend}'")

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Extracting features from %d ECGs (batch_size=%d)…",
        dataset.shape[0],
        batch_size,
    )
    t0 = time.monotonic()
    vectors = _extract_features(model, dataset, batch_size=batch_size)
    feature_dim = vectors.shape[1]
    logger.info("Extracted %d feature vectors of dim %d.", vectors.shape[0], feature_dim)

    # Build index
    if backend == "annoy":
        index_file = _build_annoy_index(vectors, out, num_trees, metric)
    else:
        index_file = _build_faiss_index(vectors, out, nlist=num_trees)

    # Build and save metadata sidecar
    metadata = _build_metadata(
        num_vectors=vectors.shape[0],
        record_ids=record_ids,
        labels=labels,
        demographics=demographics,
    )
    metadata_file = str(out / "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    build_time = time.monotonic() - t0
    logger.info("Index built in %.2fs (%s backend, %d trees).", build_time, backend, num_trees)

    return IndexBuildReport(
        index_path=index_file,
        metadata_path=metadata_file,
        num_vectors=vectors.shape[0],
        feature_dim=feature_dim,
        backend=backend,
        num_trees=num_trees,
        build_time_seconds=build_time,
    )
