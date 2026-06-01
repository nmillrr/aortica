"""Tests for aortica.retrieval.retriever — top-K similar ECG retrieval.

Tests cover:
- SimilarCaseResult dataclass
- Query encoding
- Annoy retrieval with metadata
- Filtering unlabeled matches
- Cosine similarity conversion
- API endpoint
- Edge cases (missing index, invalid backend, etc.)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from aortica.retrieval.index_builder import build_index
from aortica.retrieval.retriever import (
    SimilarCaseResult,
    _angular_distance_to_cosine_similarity,
    _encode_query,
    _l2_distance_to_similarity,
    retrieve_similar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(feature_dim: int = 16) -> MagicMock:
    """Create a mock AorticaModel with deterministic backbone output."""
    import torch

    model = MagicMock()
    model.eval = MagicMock()
    dummy_param = torch.zeros(1)
    model.parameters = MagicMock(side_effect=lambda: iter([dummy_param]))

    def backbone_fn(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        # Use input mean as seed for reproducible features
        torch.manual_seed(42)
        return torch.randn(batch, feature_dim)

    def attention_fn(x: torch.Tensor) -> torch.Tensor:
        return x

    model.backbone = MagicMock(side_effect=backbone_fn)
    model.attention = MagicMock(side_effect=attention_fn)
    return model


def _build_test_index(
    model: MagicMock, tmp_path: Path, n: int = 20, feature_dim: int = 16
) -> str:
    """Build a test index and return the output directory path."""
    rng = np.random.default_rng(42)
    dataset = rng.standard_normal((n, 12, 500)).astype(np.float64)

    labels: list[list[str]] = []
    demographics: list[dict[str, object]] = []
    for i in range(n):
        if i % 3 == 0:
            labels.append([])  # unlabeled
        elif i % 3 == 1:
            labels.append(["AF", "LBBB"])
        else:
            labels.append(["NSR"])
        demographics.append({
            "age": 50 + i,
            "sex": "M" if i % 2 == 0 else "F",
        })

    out = str(tmp_path / "test_idx")
    build_index(
        model=model,
        dataset=dataset,
        output_path=out,
        labels=labels,
        demographics=demographics,
        backend="annoy",
        num_trees=5,
    )
    return out


# ---------------------------------------------------------------------------
# SimilarCaseResult dataclass
# ---------------------------------------------------------------------------


class TestSimilarCaseResult:
    def test_defaults(self) -> None:
        result = SimilarCaseResult(
            similarity_score=0.95,
            record_id="rec_001",
        )
        assert result.similarity_score == 0.95
        assert result.record_id == "rec_001"
        assert result.diagnoses == []
        assert result.age is None
        assert result.sex is None
        assert result.outcome_summary is None
        assert result.index_id == 0

    def test_with_all_fields(self) -> None:
        result = SimilarCaseResult(
            similarity_score=0.87,
            record_id="ptbxl_12345",
            diagnoses=["AF", "LBBB"],
            age=65,
            sex="M",
            outcome_summary="Stable, no events at 1 year",
            index_id=42,
        )
        assert result.diagnoses == ["AF", "LBBB"]
        assert result.age == 65
        assert result.outcome_summary == "Stable, no events at 1 year"

    def test_serializable(self) -> None:
        result = SimilarCaseResult(
            similarity_score=0.92,
            record_id="rec",
            diagnoses=["AF"],
        )
        d = asdict(result)
        assert d["similarity_score"] == 0.92
        assert d["record_id"] == "rec"
        # JSON-serializable
        json.dumps(d)


# ---------------------------------------------------------------------------
# Distance conversion functions
# ---------------------------------------------------------------------------


class TestDistanceConversions:
    def test_angular_zero_distance(self) -> None:
        """Zero distance = identical = similarity 1.0."""
        assert _angular_distance_to_cosine_similarity(0.0) == 1.0

    def test_angular_max_distance(self) -> None:
        """Distance = sqrt(2) ≈ 1.414 means cos = 0."""
        sim = _angular_distance_to_cosine_similarity(2.0**0.5)
        assert abs(sim) < 1e-6

    def test_angular_clamps(self) -> None:
        """Large distances should clamp to 0, not go negative."""
        assert _angular_distance_to_cosine_similarity(10.0) == 0.0

    def test_l2_zero_distance(self) -> None:
        assert _l2_distance_to_similarity(0.0) == 1.0

    def test_l2_large_distance(self) -> None:
        sim = _l2_distance_to_similarity(1000.0)
        assert 0.0 < sim < 0.01


# ---------------------------------------------------------------------------
# _encode_query
# ---------------------------------------------------------------------------


class TestEncodeQuery:
    def test_2d_input(self) -> None:
        model = _make_mock_model(feature_dim=16)
        signals = np.random.randn(12, 500).astype(np.float64)
        vec = _encode_query(model, signals)
        assert vec.shape == (16,)
        assert vec.dtype == np.float64

    def test_3d_input(self) -> None:
        model = _make_mock_model(feature_dim=16)
        signals = np.random.randn(1, 12, 500).astype(np.float64)
        vec = _encode_query(model, signals)
        assert vec.shape == (16,)


# ---------------------------------------------------------------------------
# retrieve_similar — Annoy backend
# ---------------------------------------------------------------------------


class TestRetrieveSimilarAnnoy:
    @pytest.fixture()
    def model(self) -> MagicMock:
        return _make_mock_model(feature_dim=16)

    @pytest.fixture()
    def index_dir(self, model: MagicMock, tmp_path: Path) -> str:
        return _build_test_index(model, tmp_path)

    def test_basic_retrieval(self, model: MagicMock, index_dir: str) -> None:
        query = np.random.randn(12, 500).astype(np.float64)
        results = retrieve_similar(
            model=model,
            ecg_record=query,
            index_path=index_dir,
            k=3,
        )
        assert len(results) <= 3
        for r in results:
            assert isinstance(r, SimilarCaseResult)
            assert 0.0 <= r.similarity_score <= 1.0
            assert r.record_id != ""

    def test_results_sorted_by_similarity(
        self, model: MagicMock, index_dir: str
    ) -> None:
        query = np.random.randn(12, 500).astype(np.float64)
        results = retrieve_similar(
            model=model,
            ecg_record=query,
            index_path=index_dir,
            k=5,
            filter_unlabeled=False,
        )
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].similarity_score >= results[i + 1].similarity_score

    def test_filter_unlabeled(self, model: MagicMock, index_dir: str) -> None:
        """With filter_unlabeled=True, no result should have empty diagnoses."""
        query = np.random.randn(12, 500).astype(np.float64)
        results = retrieve_similar(
            model=model,
            ecg_record=query,
            index_path=index_dir,
            k=5,
            filter_unlabeled=True,
        )
        for r in results:
            assert len(r.diagnoses) > 0

    def test_no_filter_unlabeled(self, model: MagicMock, index_dir: str) -> None:
        """With filter_unlabeled=False, unlabeled results can appear."""
        query = np.random.randn(12, 500).astype(np.float64)
        results = retrieve_similar(
            model=model,
            ecg_record=query,
            index_path=index_dir,
            k=10,
            filter_unlabeled=False,
        )
        assert len(results) == 10

    def test_demographics_included(self, model: MagicMock, index_dir: str) -> None:
        query = np.random.randn(12, 500).astype(np.float64)
        results = retrieve_similar(
            model=model,
            ecg_record=query,
            index_path=index_dir,
            k=3,
            filter_unlabeled=False,
        )
        for r in results:
            assert r.age is not None
            assert r.sex in ("M", "F")

    def test_ecg_record_object(self, model: MagicMock, index_dir: str) -> None:
        """Works with an object that has a .signals attribute."""
        mock_record = MagicMock()
        mock_record.signals = np.random.randn(12, 500).astype(np.float64)
        results = retrieve_similar(
            model=model,
            ecg_record=mock_record,
            index_path=index_dir,
            k=2,
            filter_unlabeled=False,
        )
        assert len(results) == 2

    def test_k_equals_1(self, model: MagicMock, index_dir: str) -> None:
        query = np.random.randn(12, 500).astype(np.float64)
        results = retrieve_similar(
            model=model,
            ecg_record=query,
            index_path=index_dir,
            k=1,
            filter_unlabeled=False,
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRetrieveSimilarEdgeCases:
    @pytest.fixture()
    def model(self) -> MagicMock:
        return _make_mock_model(feature_dim=16)

    def test_missing_index_file(self, model: MagicMock, tmp_path: Path) -> None:
        query = np.random.randn(12, 500).astype(np.float64)
        with pytest.raises(FileNotFoundError):
            retrieve_similar(
                model=model,
                ecg_record=query,
                index_path=str(tmp_path / "nonexistent"),
                k=3,
            )

    def test_missing_metadata(self, model: MagicMock, tmp_path: Path) -> None:
        """Index dir exists but no metadata.json."""
        idx_dir = tmp_path / "no_meta"
        idx_dir.mkdir()
        # Create a dummy index file
        (idx_dir / "index.ann").touch()
        query = np.random.randn(12, 500).astype(np.float64)
        with pytest.raises(FileNotFoundError, match="Metadata sidecar"):
            retrieve_similar(
                model=model,
                ecg_record=query,
                index_path=str(idx_dir),
                k=3,
            )

    def test_invalid_backend(self, model: MagicMock, tmp_path: Path) -> None:
        query = np.random.randn(12, 500).astype(np.float64)
        with pytest.raises(ValueError, match="backend must be"):
            retrieve_similar(
                model=model,
                ecg_record=query,
                index_path=str(tmp_path),
                backend="invalid",
            )

    def test_invalid_ecg_type(self, model: MagicMock, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="ecg_record must be"):
            retrieve_similar(
                model=model,
                ecg_record="not an array",
                index_path=str(tmp_path),
            )


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


class TestRetrievalAPIEndpoint:
    def test_endpoint_exists(self) -> None:
        """The /api/v1/retrieve/similar endpoint is registered."""
        from aortica.api.app import create_app

        app = create_app(enable_auth=False)
        routes = [r.path for r in app.routes]
        assert "/api/v1/retrieve/similar" in routes

    def test_endpoint_no_index(self) -> None:
        """Returns 422 when no index is loaded."""
        from fastapi.testclient import TestClient

        from aortica.api.app import create_app

        app = create_app(enable_auth=False)
        client = TestClient(app, raise_server_exceptions=False)

        # Create a minimal file to upload
        import io
        file_bytes = io.BytesIO(b"dummy data")
        response = client.post(
            "/api/v1/retrieve/similar",
            files={"file": ("test.csv", file_bytes, "text/csv")},
        )
        assert response.status_code == 422
        assert "No retrieval index" in response.json()["detail"]

    def test_endpoint_no_model(self) -> None:
        """Returns 422 when no model is loaded but index is set."""
        from fastapi.testclient import TestClient

        from aortica.api.app import create_app

        app = create_app(enable_auth=False, model=None)
        app.state.retrieval_index_path = "/tmp/fake_index"
        client = TestClient(app, raise_server_exceptions=False)

        import io
        file_bytes = io.BytesIO(b"dummy data")
        response = client.post(
            "/api/v1/retrieve/similar",
            files={"file": ("test.csv", file_bytes, "text/csv")},
        )
        assert response.status_code == 422
        assert "No model loaded" in response.json()["detail"]
