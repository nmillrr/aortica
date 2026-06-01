"""Top-K similar ECG retrieval from a latent-space ANN index.

Given a query ECG, encodes it through the model backbone and queries the
pre-built ANN index for the most similar historical cases.

Example::

    from aortica.retrieval import retrieve_similar

    results = retrieve_similar(
        model=model,
        ecg_record=record,
        index_path="./index",
        k=3,
    )
    for r in results:
        print(r.record_id, r.similarity_score, r.diagnoses)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SimilarCaseResult:
    """A single similar-case result from retrieval.

    Attributes:
        similarity_score: Cosine similarity score (0–1, higher = more similar).
        record_id: Anonymised record identifier from the index metadata.
        diagnoses: Verified diagnosis labels for this historical case.
        age: Patient age (if available in metadata).
        sex: Patient sex (if available in metadata).
        outcome_summary: Brief outcome text (if available).
        index_id: Position of this item in the ANN index.
    """

    similarity_score: float
    record_id: str
    diagnoses: list[str] = field(default_factory=list)
    age: Optional[int] = None
    sex: Optional[str] = None
    outcome_summary: Optional[str] = None
    index_id: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _encode_query(
    model: Any,
    ecg_signals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Encode a single ECG through the model backbone.

    Args:
        model: An :class:`~aortica.models.AorticaModel` instance.
        ecg_signals: Array of shape ``[leads, samples]`` or ``[1, leads, samples]``.

    Returns:
        Feature vector of shape ``[feature_dim]``.
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

    # Ensure 3D: [1, leads, samples]
    if ecg_signals.ndim == 2:
        ecg_signals = ecg_signals[np.newaxis, :, :]

    with torch.no_grad():
        batch_t = torch.tensor(ecg_signals, dtype=torch.float32, device=device)
        features = model.backbone(batch_t)
        features = model.attention(features)

    result: NDArray[np.float64] = features.cpu().numpy().astype(np.float64).squeeze(0)
    return result


def _load_metadata(index_path: Path) -> list[dict[str, Any]]:
    """Load the metadata sidecar JSON."""
    metadata_file = index_path / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata sidecar not found at {metadata_file}. "
            "Was the index built with build_index()?"
        )
    with open(metadata_file) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _query_annoy(
    index_file: str,
    query_vector: NDArray[np.float64],
    feature_dim: int,
    k: int,
    metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"] = "angular",
) -> tuple[list[int], list[float]]:
    """Query an Annoy index and return (indices, distances)."""
    try:
        from annoy import AnnoyIndex  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "annoy is required for index querying. "
            "Install with: pip install annoy"
        ) from exc

    index = AnnoyIndex(feature_dim, metric)
    index.load(index_file)
    indices, distances = index.get_nns_by_vector(
        query_vector.tolist(), k, include_distances=True
    )
    return indices, distances


def _query_faiss(
    index_file: str,
    query_vector: NDArray[np.float64],
    k: int,
) -> tuple[list[int], list[float]]:
    """Query a FAISS index and return (indices, distances)."""
    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "faiss is required when using FAISS backend. "
            "Install with: pip install faiss-cpu"
        ) from exc

    index = faiss.read_index(index_file)
    query_f32 = query_vector.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_f32, k)
    return indices[0].tolist(), distances[0].tolist()


def _angular_distance_to_cosine_similarity(distance: float) -> float:
    """Convert Annoy angular distance to cosine similarity (0–1 range)."""
    # Annoy angular distance = sqrt(2 * (1 - cos(u, v)))
    # So cos(u, v) = 1 - distance^2 / 2
    cos_sim = 1.0 - (distance ** 2) / 2.0
    return max(0.0, min(1.0, cos_sim))


def _l2_distance_to_similarity(distance: float, scale: float = 1.0) -> float:
    """Convert L2 distance to a similarity score (0–1 range)."""
    return 1.0 / (1.0 + distance / scale)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve_similar(
    model: Any,
    ecg_record: Any,
    index_path: str | Path,
    *,
    k: int = 3,
    backend: str = "annoy",
    metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"] = "angular",
    filter_unlabeled: bool = True,
) -> list[SimilarCaseResult]:
    """Retrieve the top-K most similar historical ECGs for a query ECG.

    Encodes the query ECG through the model backbone, queries the ANN index,
    and returns similar cases with metadata.

    Args:
        model: An :class:`~aortica.models.AorticaModel` instance.
        ecg_record: An :class:`~aortica.io.ECGRecord` or numpy array
            of shape ``[leads, samples]``.
        index_path: Directory containing the ANN index and metadata sidecar
            (produced by :func:`build_index`).
        k: Number of similar cases to retrieve.  Default ``3``.
        backend: ANN backend: ``"annoy"`` (default) or ``"faiss"``.
        metric: Distance metric (Annoy only).  Default ``"angular"``.
        filter_unlabeled: If ``True`` (default), skip matches that have
            no verified diagnoses in the metadata.

    Returns:
        List of :class:`SimilarCaseResult` sorted by decreasing similarity.

    Raises:
        FileNotFoundError: If the index or metadata file is not found.
        ImportError: If the ANN library is not installed.
        ValueError: If *backend* is not ``"annoy"`` or ``"faiss"``.
    """
    if backend not in ("annoy", "faiss"):
        raise ValueError(f"backend must be 'annoy' or 'faiss', got '{backend}'")

    idx_path = Path(index_path)

    # Extract signal array
    if hasattr(ecg_record, "signals"):
        signals = ecg_record.signals
    elif isinstance(ecg_record, np.ndarray):
        signals = ecg_record
    else:
        raise TypeError(
            f"ecg_record must be an ECGRecord or numpy array, got {type(ecg_record)}"
        )

    # Encode query
    t0 = time.monotonic()
    query_vector = _encode_query(model, signals)
    feature_dim = query_vector.shape[0]

    # Load metadata
    metadata = _load_metadata(idx_path)

    # Determine how many extra candidates to fetch if filtering
    fetch_k = k * 3 if filter_unlabeled else k

    # Query index
    if backend == "annoy":
        index_file = str(idx_path / "index.ann")
        if not Path(index_file).exists():
            raise FileNotFoundError(f"Annoy index not found at {index_file}")
        indices, distances = _query_annoy(index_file, query_vector, feature_dim, fetch_k, metric)
    else:
        index_file = str(idx_path / "index.faiss")
        if not Path(index_file).exists():
            raise FileNotFoundError(f"FAISS index not found at {index_file}")
        indices, distances = _query_faiss(index_file, query_vector, fetch_k)

    query_time = time.monotonic() - t0
    logger.info("Query completed in %.1fms (backend=%s, k=%d)", query_time * 1000, backend, k)

    # Build results with metadata
    results: list[SimilarCaseResult] = []
    for idx, dist in zip(indices, distances):
        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]
        diagnoses = meta.get("diagnoses", [])

        # Filter unlabeled matches
        if filter_unlabeled and not diagnoses:
            continue

        # Convert distance to similarity
        if backend == "annoy" and metric == "angular":
            similarity = _angular_distance_to_cosine_similarity(dist)
        elif backend == "annoy" and metric == "euclidean":
            similarity = _l2_distance_to_similarity(dist)
        else:
            # FAISS uses L2 by default
            similarity = _l2_distance_to_similarity(dist)

        results.append(
            SimilarCaseResult(
                similarity_score=round(similarity, 4),
                record_id=meta.get("record_id", f"unknown_{idx}"),
                diagnoses=diagnoses,
                age=meta.get("age"),
                sex=meta.get("sex"),
                outcome_summary=meta.get("outcome_summary"),
                index_id=idx,
            )
        )

        if len(results) >= k:
            break

    # Sort by similarity descending (should already be, but ensure)
    results.sort(key=lambda r: r.similarity_score, reverse=True)

    return results
