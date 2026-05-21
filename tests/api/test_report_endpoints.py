"""Tests for aortica.api.report_endpoints — Report generation API endpoints."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from aortica.api.app import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Mock ResultStore and StoredResult
# ---------------------------------------------------------------------------


class MockStoredResult:
    """Minimal mock of StoredResult for testing."""

    def __init__(
        self,
        id: int,
        ecg_hash: str = "abc123",
        predictions: Optional[Dict[str, Any]] = None,
        quality: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: float = 1700000000.0,
        synced: bool = False,
    ) -> None:
        self.id = id
        self.ecg_hash = ecg_hash
        self.predictions = predictions or _default_predictions()
        self.quality = quality or {"overall": 85}
        self.metadata = metadata or _default_metadata()
        self.timestamp = timestamp
        self.synced = synced


def _default_predictions() -> Dict[str, Any]:
    """Return realistic default predictions for testing."""
    return {
        "rhythm": {
            "AF": 0.92,
            "normal_sinus_rhythm": 0.05,
            "VT": 0.01,
            "sinus_brady": 0.02,
        },
        "structural": {
            "LVH": 0.78,
            "LVSD": 0.15,
        },
        "ischaemia": {
            "STEMI": 0.03,
            "QTc_prolongation": 0.45,
        },
        "risk": {
            "mortality_1y": 0.22,
            "hf_hosp_12m": 0.18,
            "af_onset_12m": 0.85,
            "ecg_predicted_ef": 0.55,
            "conduction_disease_trajectory": 0.10,
            "sudden_cardiac_death_risk": 0.08,
        },
    }


def _default_metadata() -> Dict[str, Any]:
    """Return default ECG metadata."""
    return {
        "num_leads": 12,
        "sample_rate": 500,
        "duration_seconds": 10.0,
        "source_format": "wfdb",
        "patient_metadata": {"patient_id": "P001", "age": 65, "sex": "M"},
    }


class MockResultStore:
    """Mock ResultStore that returns pre-configured results."""

    def __init__(
        self, results: Optional[Dict[int, MockStoredResult]] = None
    ) -> None:
        self._results = results or {}

    def get_result_by_id(self, result_id: int) -> Optional[MockStoredResult]:
        return self._results.get(result_id)


# ---------------------------------------------------------------------------
# Helper: create a mock generate_pdf that writes fake content
# ---------------------------------------------------------------------------


def _make_mock_generate_pdf(
    fake_content: bytes = b"%PDF-1.4 fake",
    expected_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Return a mock generate_pdf function that writes fake PDF bytes."""

    def _mock(
        predictions: Any,
        ecg_record: Any,
        output_path: Any = "report.pdf",
        **kwargs: Any,
    ) -> Any:
        from pathlib import Path

        p = Path(output_path)
        p.write_bytes(fake_content)
        if expected_kwargs:
            for k, v in expected_kwargs.items():
                assert kwargs.get(k) == v, (
                    f"Expected {k}={v!r}, got {kwargs.get(k)!r}"
                )
        return p

    return _mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_store() -> MockResultStore:
    """Return a mock ResultStore with two default results."""
    return MockResultStore(
        results={
            1: MockStoredResult(id=1),
            2: MockStoredResult(
                id=2,
                predictions=_default_predictions(),
                metadata={
                    "num_leads": 12,
                    "sample_rate": 500,
                    "duration_seconds": 10.0,
                    "source_format": "dicom",
                    "acquisition_datetime": "2026-01-15T10:30:00Z",
                },
            ),
        }
    )


@pytest.fixture()
def app(mock_store: MockResultStore) -> fastapi.FastAPI:
    """Return a FastAPI app with auth disabled and a mock result store."""
    application = create_app(enable_auth=False)
    application.state.result_store = mock_store  # type: ignore[attr-defined]
    return application


@pytest.fixture()
def client(app: fastapi.FastAPI) -> TestClient:
    """Synchronous test client."""
    return TestClient(app)


@pytest.fixture()
def app_no_store() -> fastapi.FastAPI:
    """Return a FastAPI app with NO result store attached."""
    return create_app(enable_auth=False)


@pytest.fixture()
def client_no_store(app_no_store: fastapi.FastAPI) -> TestClient:
    """Test client for app without result store."""
    return TestClient(app_no_store)


# ---------------------------------------------------------------------------
# POST /api/v1/report/pdf/{result_id}
# ---------------------------------------------------------------------------


class TestPDFReportEndpoint:
    """Tests for the PDF report generation endpoint."""

    def test_pdf_404_unknown_id(self, client: TestClient) -> None:
        resp = client.post("/api/v1/report/pdf/999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_pdf_422_no_store(self, client_no_store: TestClient) -> None:
        resp = client_no_store.post("/api/v1/report/pdf/1")
        assert resp.status_code == 422
        assert "not configured" in resp.json()["detail"].lower()

    def test_pdf_success_returns_pdf(self, client: TestClient) -> None:
        """Test PDF generation with mocked WeasyPrint."""
        fake_pdf_content = b"%PDF-1.4 fake pdf content"
        mock_fn = _make_mock_generate_pdf(fake_content=fake_pdf_content)

        with patch(
            "aortica.reports.pdf_report.generate_pdf",
            side_effect=mock_fn,
        ):
            resp = client.post("/api/v1/report/pdf/1")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/pdf"
            assert "attachment" in resp.headers.get("content-disposition", "")
            assert resp.content == fake_pdf_content

    def test_pdf_custom_model_version(self, client: TestClient) -> None:
        mock_fn = _make_mock_generate_pdf(
            expected_kwargs={"model_version": "v2.1.0"},
        )
        with patch(
            "aortica.reports.pdf_report.generate_pdf",
            side_effect=mock_fn,
        ):
            resp = client.post("/api/v1/report/pdf/1?model_version=v2.1.0")
            assert resp.status_code == 200

    def test_pdf_custom_threshold(self, client: TestClient) -> None:
        mock_fn = _make_mock_generate_pdf(
            expected_kwargs={"finding_threshold": 0.8},
        )
        with patch(
            "aortica.reports.pdf_report.generate_pdf",
            side_effect=mock_fn,
        ):
            resp = client.post("/api/v1/report/pdf/1?finding_threshold=0.8")
            assert resp.status_code == 200

    def test_pdf_filename_includes_result_id(self, client: TestClient) -> None:
        mock_fn = _make_mock_generate_pdf()
        with patch(
            "aortica.reports.pdf_report.generate_pdf",
            side_effect=mock_fn,
        ):
            resp = client.post("/api/v1/report/pdf/1")
            assert "report_1.pdf" in resp.headers.get(
                "content-disposition", ""
            )

    def test_pdf_422_on_import_error(self, client: TestClient) -> None:
        """If weasyprint is not installed, endpoint returns 422."""
        with patch(
            "aortica.reports.pdf_report.generate_pdf",
            side_effect=ImportError("weasyprint not found"),
        ):
            resp = client.post("/api/v1/report/pdf/1")
            assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/report/jsonld/{result_id}
# ---------------------------------------------------------------------------


class TestJSONLDReportEndpoint:
    """Tests for the JSON-LD report generation endpoint."""

    def test_jsonld_404_unknown_id(self, client: TestClient) -> None:
        resp = client.post("/api/v1/report/jsonld/999")
        assert resp.status_code == 404

    def test_jsonld_422_no_store(self, client_no_store: TestClient) -> None:
        resp = client_no_store.post("/api/v1/report/jsonld/1")
        assert resp.status_code == 422

    def test_jsonld_success(self, client: TestClient) -> None:
        mock_jsonld = {
            "@context": {"snomed": "http://snomed.info/id/"},
            "@type": "MedicalTest",
            "findings": [],
        }
        with patch(
            "aortica.reports.jsonld_report.generate_jsonld",
            return_value=mock_jsonld,
        ):
            resp = client.post("/api/v1/report/jsonld/1")
            assert resp.status_code == 200
            body = resp.json()
            assert "@context" in body
            assert body["@type"] == "MedicalTest"

    def test_jsonld_custom_model_version(self, client: TestClient) -> None:
        def mock_generate(
            predictions: Any,
            ecg_metadata: Any = None,
            model_version: str = "unknown",
            **kwargs: Any,
        ) -> Dict[str, Any]:
            assert model_version == "v3.0.0"
            return {"@context": {}, "@type": "MedicalTest"}

        with patch(
            "aortica.reports.jsonld_report.generate_jsonld",
            side_effect=mock_generate,
        ):
            resp = client.post(
                "/api/v1/report/jsonld/1?model_version=v3.0.0"
            )
            assert resp.status_code == 200

    def test_jsonld_returns_json_content_type(
        self, client: TestClient
    ) -> None:
        with patch(
            "aortica.reports.jsonld_report.generate_jsonld",
            return_value={"@context": {}},
        ):
            resp = client.post("/api/v1/report/jsonld/1")
            assert "application/json" in resp.headers["content-type"]

    def test_jsonld_422_on_import_error(self, client: TestClient) -> None:
        with patch(
            "aortica.reports.jsonld_report.generate_jsonld",
            side_effect=ImportError("pyld not found"),
        ):
            resp = client.post("/api/v1/report/jsonld/1")
            assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/report/fhir/{result_id}
# ---------------------------------------------------------------------------


class TestFHIRReportEndpoint:
    """Tests for the FHIR R4 DiagnosticReport endpoint."""

    def _mock_fhir_output(self, entry: Optional[list] = None) -> MagicMock:
        mock = MagicMock()
        mock.bundle_json = json.dumps(
            {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": entry or [],
            }
        )
        return mock

    def test_fhir_404_unknown_id(self, client: TestClient) -> None:
        resp = client.post("/api/v1/report/fhir/999")
        assert resp.status_code == 404

    def test_fhir_422_no_store(self, client_no_store: TestClient) -> None:
        resp = client_no_store.post("/api/v1/report/fhir/1")
        assert resp.status_code == 422

    def test_fhir_success(self, client: TestClient) -> None:
        mock_out = self._mock_fhir_output(
            entry=[
                {
                    "resource": {
                        "resourceType": "DiagnosticReport",
                        "status": "final",
                    }
                }
            ]
        )
        with patch(
            "aortica.integration.fhir.to_diagnostic_report",
            return_value=mock_out,
        ):
            resp = client.post("/api/v1/report/fhir/1")
            assert resp.status_code == 200
            body = resp.json()
            assert body["resourceType"] == "Bundle"
            assert body["type"] == "collection"

    def test_fhir_with_patient_ref(self, client: TestClient) -> None:
        mock_out = self._mock_fhir_output()

        def mock_to_diag(
            predictions: Any,
            patient_ref: Any = None,
            ecg_metadata: Any = None,
            **kwargs: Any,
        ) -> Any:
            assert patient_ref == "Patient/12345"
            return mock_out

        with patch(
            "aortica.integration.fhir.to_diagnostic_report",
            side_effect=mock_to_diag,
        ):
            resp = client.post(
                "/api/v1/report/fhir/1?patient_ref=Patient/12345"
            )
            assert resp.status_code == 200

    def test_fhir_custom_confidence_threshold(
        self, client: TestClient
    ) -> None:
        mock_out = self._mock_fhir_output()

        def mock_to_diag(
            predictions: Any,
            patient_ref: Any = None,
            ecg_metadata: Any = None,
            confidence_threshold: float = 0.30,
            **kwargs: Any,
        ) -> Any:
            assert confidence_threshold == 0.5
            return mock_out

        with patch(
            "aortica.integration.fhir.to_diagnostic_report",
            side_effect=mock_to_diag,
        ):
            resp = client.post(
                "/api/v1/report/fhir/1?confidence_threshold=0.5"
            )
            assert resp.status_code == 200

    def test_fhir_returns_json(self, client: TestClient) -> None:
        mock_out = self._mock_fhir_output()
        with patch(
            "aortica.integration.fhir.to_diagnostic_report",
            return_value=mock_out,
        ):
            resp = client.post("/api/v1/report/fhir/1")
            assert "application/json" in resp.headers["content-type"]

    def test_fhir_422_on_import_error(self, client: TestClient) -> None:
        with patch(
            "aortica.integration.fhir.to_diagnostic_report",
            side_effect=ImportError("fhir.resources not found"),
        ):
            resp = client.post("/api/v1/report/fhir/1")
            assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/report/hl7/{result_id}
# ---------------------------------------------------------------------------


class TestHL7ReportEndpoint:
    """Tests for the HL7 v2.x ORU^R01 endpoint."""

    _MOCK_HL7 = "MSH|^~\\&|Aortica|AorticaSite|EHR|Hospital||ORU^R01\r"

    def test_hl7_404_unknown_id(self, client: TestClient) -> None:
        resp = client.post("/api/v1/report/hl7/999")
        assert resp.status_code == 404

    def test_hl7_422_no_store(self, client_no_store: TestClient) -> None:
        resp = client_no_store.post("/api/v1/report/hl7/1")
        assert resp.status_code == 422

    def test_hl7_success(self, client: TestClient) -> None:
        with patch(
            "aortica.integration.hl7v2.to_oru_r01",
            return_value=self._MOCK_HL7,
        ):
            resp = client.post("/api/v1/report/hl7/1")
            assert resp.status_code == 200
            assert "text/plain" in resp.headers["content-type"]
            assert "MSH" in resp.text

    def test_hl7_with_patient_id(self, client: TestClient) -> None:
        def mock_to_oru(
            predictions: Any,
            patient_id: Any = None,
            order_id: Any = None,
            **kwargs: Any,
        ) -> str:
            assert patient_id == "P12345"
            return self._MOCK_HL7

        with patch(
            "aortica.integration.hl7v2.to_oru_r01",
            side_effect=mock_to_oru,
        ):
            resp = client.post("/api/v1/report/hl7/1?patient_id=P12345")
            assert resp.status_code == 200

    def test_hl7_with_order_id(self, client: TestClient) -> None:
        def mock_to_oru(
            predictions: Any,
            patient_id: Any = None,
            order_id: Any = None,
            **kwargs: Any,
        ) -> str:
            assert order_id == "ORD-987"
            return self._MOCK_HL7

        with patch(
            "aortica.integration.hl7v2.to_oru_r01",
            side_effect=mock_to_oru,
        ):
            resp = client.post("/api/v1/report/hl7/1?order_id=ORD-987")
            assert resp.status_code == 200

    def test_hl7_custom_confidence_threshold(
        self, client: TestClient
    ) -> None:
        def mock_to_oru(
            predictions: Any,
            patient_id: Any = None,
            order_id: Any = None,
            confidence_threshold: float = 0.30,
            **kwargs: Any,
        ) -> str:
            assert confidence_threshold == 0.7
            return self._MOCK_HL7

        with patch(
            "aortica.integration.hl7v2.to_oru_r01",
            side_effect=mock_to_oru,
        ):
            resp = client.post(
                "/api/v1/report/hl7/1?confidence_threshold=0.7"
            )
            assert resp.status_code == 200

    def test_hl7_filename_includes_result_id(
        self, client: TestClient
    ) -> None:
        with patch(
            "aortica.integration.hl7v2.to_oru_r01",
            return_value=self._MOCK_HL7,
        ):
            resp = client.post("/api/v1/report/hl7/1")
            assert "report_1.hl7" in resp.headers.get(
                "content-disposition", ""
            )

    def test_hl7_422_on_import_error(self, client: TestClient) -> None:
        with patch(
            "aortica.integration.hl7v2.to_oru_r01",
            side_effect=ImportError("hl7apy not found"),
        ):
            resp = client.post("/api/v1/report/hl7/1")
            assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Cross-endpoint tests
# ---------------------------------------------------------------------------


class TestReportEndpointAuth:
    """Tests verifying authentication is required when enabled."""

    @pytest.fixture()
    def auth_app(self, mock_store: MockResultStore) -> fastapi.FastAPI:
        """App with auth enabled."""
        application = create_app(enable_auth=True)
        application.state.result_store = mock_store  # type: ignore[attr-defined]
        return application

    @pytest.fixture()
    def auth_client(self, auth_app: fastapi.FastAPI) -> TestClient:
        return TestClient(auth_app, raise_server_exceptions=False)

    def test_pdf_requires_auth(self, auth_client: TestClient) -> None:
        resp = auth_client.post("/api/v1/report/pdf/1")
        assert resp.status_code == 401

    def test_jsonld_requires_auth(self, auth_client: TestClient) -> None:
        resp = auth_client.post("/api/v1/report/jsonld/1")
        assert resp.status_code == 401

    def test_fhir_requires_auth(self, auth_client: TestClient) -> None:
        resp = auth_client.post("/api/v1/report/fhir/1")
        assert resp.status_code == 401

    def test_hl7_requires_auth(self, auth_client: TestClient) -> None:
        resp = auth_client.post("/api/v1/report/hl7/1")
        assert resp.status_code == 401


class TestReportEndpointRouting:
    """Tests verifying routes are registered correctly."""

    def test_all_report_routes_exist(self, app: fastapi.FastAPI) -> None:
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/api/v1/report/pdf/{result_id}" in route_paths
        assert "/api/v1/report/jsonld/{result_id}" in route_paths
        assert "/api/v1/report/fhir/{result_id}" in route_paths
        assert "/api/v1/report/hl7/{result_id}" in route_paths

    def test_report_routes_are_post(self, app: fastapi.FastAPI) -> None:
        for route in app.routes:
            if hasattr(route, "path") and "/api/v1/report/" in route.path:
                assert "POST" in route.methods  # type: ignore[attr-defined]


class TestReportEndpointImports:
    """Tests verifying the report_endpoints module imports correctly."""

    def test_import_create_report_router(self) -> None:
        from aortica.api.report_endpoints import create_report_router

        assert callable(create_report_router)

    def test_import_error_response(self) -> None:
        from aortica.api.report_endpoints import ReportErrorResponse

        resp = ReportErrorResponse(detail="test error")
        assert resp.detail == "test error"
