"""Tests for aortica.retrieval.index_builder — latent space index construction.

Tests cover:
- Feature extraction from model backbone
- Annoy index build and query
- FAISS index build and query
- Metadata sidecar generation
- CLI command invocation
- Edge cases (invalid backend, empty dataset, etc.)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from aortica.retrieval.index_builder import (
    IndexBuildReport,
    IndexMetadataEntry,
    _build_metadata,
    _extract_features,
    build_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(feature_dim: int = 256) -> MagicMock:
    """Create a mock AorticaModel that returns deterministic features."""
    import torch

    model = MagicMock()
    model.eval = MagicMock()

    # Mock parameters() so next(model.parameters()).device works
    dummy_param = torch.zeros(1)
    model.parameters = MagicMock(return_value=iter([dummy_param]))

    def backbone_fn(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        # Deterministic features based on input shape
        torch.manual_seed(batch)
        features = torch.randn(batch, feature_dim)
        return features

    def attention_fn(x: torch.Tensor) -> torch.Tensor:
        return x  # pass-through

    model.backbone = MagicMock(side_effect=backbone_fn)
    model.attention = MagicMock(side_effect=attention_fn)

    return model


def _make_dataset(n: int = 20, leads: int = 12, samples: int = 5000) -> np.ndarray:
    """Create a synthetic ECG dataset."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, leads, samples)).astype(np.float64)


# ---------------------------------------------------------------------------
# IndexMetadataEntry dataclass
# ---------------------------------------------------------------------------


class TestIndexMetadataEntry:
    def test_defaults(self) -> None:
        entry = IndexMetadataEntry(index_id=0, record_id="rec_000")
        assert entry.index_id == 0
        assert entry.record_id == "rec_000"
        assert entry.diagnoses == []
        assert entry.age is None
        assert entry.sex is None

    def test_with_all_fields(self) -> None:
        entry = IndexMetadataEntry(
            index_id=5,
            record_id="ptbxl_12345",
            diagnoses=["AF", "LBBB"],
            age=65,
            sex="M",
        )
        assert entry.diagnoses == ["AF", "LBBB"]
        assert entry.age == 65
        assert entry.sex == "M"


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    def test_auto_record_ids(self) -> None:
        metadata = _build_metadata(3, record_ids=None, labels=None, demographics=None)
        assert len(metadata) == 3
        assert metadata[0]["record_id"] == "record_000000"
        assert metadata[1]["record_id"] == "record_000001"
        assert metadata[2]["record_id"] == "record_000002"

    def test_custom_record_ids(self) -> None:
        metadata = _build_metadata(
            2,
            record_ids=["a", "b"],
            labels=None,
            demographics=None,
        )
        assert metadata[0]["record_id"] == "a"
        assert metadata[1]["record_id"] == "b"

    def test_with_labels(self) -> None:
        metadata = _build_metadata(
            2,
            record_ids=None,
            labels=[["AF", "SVT"], ["NSR"]],
            demographics=None,
        )
        assert metadata[0]["diagnoses"] == ["AF", "SVT"]
        assert metadata[1]["diagnoses"] == ["NSR"]

    def test_with_demographics(self) -> None:
        metadata = _build_metadata(
            2,
            record_ids=None,
            labels=None,
            demographics=[
                {"age": 70, "sex": "F"},
                {"age": 45, "sex": "M"},
            ],
        )
        assert metadata[0]["age"] == 70
        assert metadata[0]["sex"] == "F"
        assert metadata[1]["age"] == 45

    def test_empty_demographics(self) -> None:
        metadata = _build_metadata(
            1,
            record_ids=None,
            labels=None,
            demographics=[{}],
        )
        assert metadata[0]["age"] is None
        assert metadata[0]["sex"] is None


# ---------------------------------------------------------------------------
# _extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    @pytest.fixture()
    def model(self) -> MagicMock:
        return _make_mock_model(feature_dim=256)

    def test_output_shape(self, model: MagicMock) -> None:
        dataset = _make_dataset(n=10)
        features = _extract_features(model, dataset, batch_size=4)
        assert features.shape == (10, 256)

    def test_dtype(self, model: MagicMock) -> None:
        dataset = _make_dataset(n=5)
        features = _extract_features(model, dataset, batch_size=5)
        assert features.dtype == np.float64

    def test_single_batch(self, model: MagicMock) -> None:
        dataset = _make_dataset(n=3)
        features = _extract_features(model, dataset, batch_size=100)
        assert features.shape == (3, 256)

    def test_exact_batch(self, model: MagicMock) -> None:
        """N exactly divisible by batch_size."""
        dataset = _make_dataset(n=8)
        features = _extract_features(model, dataset, batch_size=4)
        assert features.shape == (8, 256)


# ---------------------------------------------------------------------------
# build_index — Annoy backend
# ---------------------------------------------------------------------------


class TestBuildIndexAnnoy:
    @pytest.fixture()
    def model(self) -> MagicMock:
        return _make_mock_model(feature_dim=16)

    def test_basic_build(self, model: MagicMock, tmp_path: Path) -> None:
        dataset = _make_dataset(n=10, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=tmp_path / "idx",
            backend="annoy",
            num_trees=5,
        )
        assert isinstance(report, IndexBuildReport)
        assert report.num_vectors == 10
        assert report.feature_dim == 16
        assert report.backend == "annoy"
        assert report.num_trees == 5
        assert report.build_time_seconds > 0
        assert Path(report.index_path).exists()
        assert Path(report.metadata_path).exists()

    def test_metadata_sidecar_valid_json(
        self, model: MagicMock, tmp_path: Path
    ) -> None:
        dataset = _make_dataset(n=5, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=tmp_path / "idx",
            backend="annoy",
            num_trees=3,
            record_ids=[f"rec_{i}" for i in range(5)],
            labels=[["AF"], ["NSR"], ["VT", "LBBB"], [], ["SVT"]],
            demographics=[
                {"age": 60, "sex": "M"},
                {"age": 45, "sex": "F"},
                {},
                {"age": 70},
                {"sex": "F"},
            ],
        )
        with open(report.metadata_path) as f:
            metadata = json.load(f)

        assert len(metadata) == 5
        assert metadata[0]["record_id"] == "rec_0"
        assert metadata[0]["diagnoses"] == ["AF"]
        assert metadata[0]["age"] == 60
        assert metadata[0]["sex"] == "M"
        assert metadata[2]["diagnoses"] == ["VT", "LBBB"]
        assert metadata[2]["age"] is None
        assert metadata[3]["diagnoses"] == []

    def test_annoy_index_queryable(self, model: MagicMock, tmp_path: Path) -> None:
        """Verify the saved Annoy index can be loaded and queried."""
        dataset = _make_dataset(n=20, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=tmp_path / "idx",
            backend="annoy",
            num_trees=10,
        )
        from annoy import AnnoyIndex

        index = AnnoyIndex(report.feature_dim, "angular")
        index.load(report.index_path)
        assert index.get_n_items() == 20

        # Query for neighbours of item 0
        neighbours, distances = index.get_nns_by_item(0, 3, include_distances=True)
        assert len(neighbours) == 3
        assert len(distances) == 3
        assert neighbours[0] == 0  # Closest to itself

    def test_auto_record_ids(self, model: MagicMock, tmp_path: Path) -> None:
        dataset = _make_dataset(n=3, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=tmp_path / "idx",
            backend="annoy",
            num_trees=3,
        )
        with open(report.metadata_path) as f:
            metadata = json.load(f)
        assert metadata[0]["record_id"] == "record_000000"
        assert metadata[2]["record_id"] == "record_000002"

    def test_creates_output_directory(self, model: MagicMock, tmp_path: Path) -> None:
        out = tmp_path / "deeply" / "nested" / "path"
        dataset = _make_dataset(n=3, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=out,
            backend="annoy",
            num_trees=3,
        )
        assert Path(report.index_path).exists()

    def test_custom_metric(self, model: MagicMock, tmp_path: Path) -> None:
        dataset = _make_dataset(n=5, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=tmp_path / "idx",
            backend="annoy",
            num_trees=3,
            metric="euclidean",
        )
        assert report.num_vectors == 5


# ---------------------------------------------------------------------------
# build_index — FAISS backend
# ---------------------------------------------------------------------------


class TestBuildIndexFAISS:
    @pytest.fixture()
    def model(self) -> MagicMock:
        return _make_mock_model(feature_dim=16)

    def test_basic_build(self, model: MagicMock, tmp_path: Path) -> None:
        pytest.importorskip("faiss")
        dataset = _make_dataset(n=10, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=tmp_path / "idx",
            backend="faiss",
            num_trees=5,
        )
        assert isinstance(report, IndexBuildReport)
        assert report.num_vectors == 10
        assert report.backend == "faiss"
        assert Path(report.index_path).exists()

    def test_faiss_index_queryable(self, model: MagicMock, tmp_path: Path) -> None:
        """Verify the saved FAISS index can be loaded and queried."""
        faiss = pytest.importorskip("faiss")
        dataset = _make_dataset(n=20, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=tmp_path / "idx",
            backend="faiss",
            num_trees=5,
        )
        index = faiss.read_index(report.index_path)
        assert index.ntotal == 20

        # Query
        query = np.random.randn(1, 16).astype(np.float32)
        distances, indices = index.search(query, 3)
        assert indices.shape == (1, 3)

    def test_faiss_small_dataset(self, model: MagicMock, tmp_path: Path) -> None:
        """Small dataset uses IndexFlatL2 instead of IVF."""
        pytest.importorskip("faiss")
        dataset = _make_dataset(n=5, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=tmp_path / "idx",
            backend="faiss",
            num_trees=5,
        )
        assert report.num_vectors == 5


# ---------------------------------------------------------------------------
# build_index — edge cases
# ---------------------------------------------------------------------------


class TestBuildIndexEdgeCases:
    @pytest.fixture()
    def model(self) -> MagicMock:
        return _make_mock_model(feature_dim=16)

    def test_invalid_backend(self, model: MagicMock, tmp_path: Path) -> None:
        dataset = _make_dataset(n=3, samples=500)
        with pytest.raises(ValueError, match="backend must be"):
            build_index(
                model=model,
                dataset=dataset,
                output_path=tmp_path,
                backend="invalid",
            )

    def test_single_record(self, model: MagicMock, tmp_path: Path) -> None:
        dataset = _make_dataset(n=1, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=tmp_path / "idx",
            backend="annoy",
            num_trees=3,
        )
        assert report.num_vectors == 1

    def test_custom_batch_size(self, model: MagicMock, tmp_path: Path) -> None:
        dataset = _make_dataset(n=7, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=tmp_path / "idx",
            backend="annoy",
            num_trees=3,
            batch_size=2,
        )
        assert report.num_vectors == 7

    def test_large_num_trees(self, model: MagicMock, tmp_path: Path) -> None:
        dataset = _make_dataset(n=5, samples=500)
        report = build_index(
            model=model,
            dataset=dataset,
            output_path=tmp_path / "idx",
            backend="annoy",
            num_trees=200,
        )
        assert report.num_trees == 200


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


class TestBuildIndexCLI:
    def test_cli_command_exists(self) -> None:
        """Verify the build-index command is registered."""
        from click.testing import CliRunner

        from aortica.cli import _build_cli

        cli = _build_cli()
        runner = CliRunner()
        result = runner.invoke(cli, ["build-index", "--help"])
        assert result.exit_code == 0
        assert "Build a latent-space ANN index" in result.output

    def test_cli_requires_dataset(self) -> None:
        """--dataset is required."""
        from click.testing import CliRunner

        from aortica.cli import _build_cli

        cli = _build_cli()
        runner = CliRunner()
        result = runner.invoke(cli, ["build-index", "--output", "/tmp/idx"])
        assert result.exit_code != 0

    def test_cli_requires_output(self) -> None:
        """--output is required."""
        from click.testing import CliRunner

        from aortica.cli import _build_cli

        cli = _build_cli()
        runner = CliRunner()
        result = runner.invoke(cli, ["build-index", "--dataset", "/tmp/data"])
        assert result.exit_code != 0

    def test_cli_help_shows_options(self) -> None:
        """All key options appear in --help."""
        from click.testing import CliRunner

        from aortica.cli import _build_cli

        cli = _build_cli()
        runner = CliRunner()
        result = runner.invoke(cli, ["build-index", "--help"])
        assert "--dataset" in result.output
        assert "--model" in result.output
        assert "--output" in result.output
        assert "--backend" in result.output
        assert "--num-trees" in result.output
        assert "--batch-size" in result.output
        assert "--max-records" in result.output


# ---------------------------------------------------------------------------
# IndexBuildReport
# ---------------------------------------------------------------------------


class TestIndexBuildReport:
    def test_fields(self) -> None:
        report = IndexBuildReport(
            index_path="/tmp/index.ann",
            metadata_path="/tmp/metadata.json",
            num_vectors=1000,
            feature_dim=256,
            backend="annoy",
            num_trees=100,
            build_time_seconds=12.5,
        )
        assert report.num_vectors == 1000
        assert report.feature_dim == 256
        assert report.backend == "annoy"
        assert report.build_time_seconds == 12.5

    def test_report_attributes(self) -> None:
        report = IndexBuildReport(
            index_path="/data/index.faiss",
            metadata_path="/data/metadata.json",
            num_vectors=50000,
            feature_dim=256,
            backend="faiss",
            num_trees=64,
            build_time_seconds=45.3,
        )
        assert report.index_path == "/data/index.faiss"
        assert report.metadata_path == "/data/metadata.json"
        assert report.backend == "faiss"
