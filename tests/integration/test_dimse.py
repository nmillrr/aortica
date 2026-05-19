"""Tests for aortica.integration.dimse — DICOM DIMSE C-STORE/C-FIND (US-083)."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import generate_uid

from aortica.integration.dimse import (
    CFindResult,
    CStoreMultiResult,
    CStoreResult,
    DicomClient,
    _build_find_dataset,
    _dataset_to_dict,
    _status_category,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.33"
_ECG_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.9.1.1"

# Patch target — AE is imported lazily inside each method
_AE_PATCH = "pynetdicom.AE"


@pytest.fixture
def client() -> DicomClient:
    return DicomClient(local_ae_title="TEST_AE", local_port=11114, timeout=5)


@pytest.fixture
def sample_sr_dataset() -> Dataset:
    ds = Dataset()
    ds.SOPClassUID = _SR_SOP_CLASS
    ds.SOPInstanceUID = generate_uid()
    ds.PatientName = "DOE^JOHN"
    ds.PatientID = "PAT001"
    ds.Modality = "SR"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = _SR_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
    ds.file_meta = file_meta  # type: ignore[assignment]
    return ds


@pytest.fixture
def sample_ecg_dataset() -> Dataset:
    ds = Dataset()
    ds.SOPClassUID = _ECG_SOP_CLASS
    ds.SOPInstanceUID = generate_uid()
    ds.PatientName = "SMITH^JANE"
    ds.PatientID = "PAT002"
    ds.Modality = "ECG"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = _ECG_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
    ds.file_meta = file_meta  # type: ignore[assignment]
    return ds


# ---------------------------------------------------------------------------
# Test: DicomClient construction
# ---------------------------------------------------------------------------


class TestDicomClientInit:
    def test_default_init(self) -> None:
        c = DicomClient()
        assert c.local_ae_title == "AORTICA"
        assert c.local_port == 11113
        assert c.timeout == 30

    def test_custom_init(self) -> None:
        c = DicomClient(local_ae_title="MY_AE", local_port=5000, timeout=60)
        assert c.local_ae_title == "MY_AE"
        assert c.local_port == 5000
        assert c.timeout == 60

    def test_additional_sop_classes(self) -> None:
        c = DicomClient(additional_sop_classes=["1.2.3.4"])
        assert "1.2.3.4" in c._additional_sop_classes


# ---------------------------------------------------------------------------
# Test: CStoreResult dataclass
# ---------------------------------------------------------------------------


class TestCStoreResult:
    def test_defaults(self) -> None:
        r = CStoreResult()
        assert r.success is False
        assert r.status_code == 0
        assert r.error_message == ""

    def test_success(self) -> None:
        r = CStoreResult(success=True, status_code=0x0000, status_category="Success")
        assert r.success is True
        assert r.status_category == "Success"


# ---------------------------------------------------------------------------
# Test: CFindResult dataclass
# ---------------------------------------------------------------------------


class TestCFindResult:
    def test_defaults(self) -> None:
        r = CFindResult()
        assert r.success is False
        assert r.matches == []
        assert r.match_count == 0

    def test_with_matches(self) -> None:
        r = CFindResult(success=True, matches=[{"PatientID": "123"}], match_count=1)
        assert r.match_count == 1


# ---------------------------------------------------------------------------
# Test: Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_status_category_success(self) -> None:
        assert _status_category(0x0000) == "Success"

    def test_status_category_warning(self) -> None:
        assert _status_category(0xB000) == "Warning"
        assert _status_category(0xBFFF) == "Warning"

    def test_status_category_pending(self) -> None:
        assert _status_category(0xFF00) == "Pending"
        assert _status_category(0xFF01) == "Pending"

    def test_status_category_failure_c(self) -> None:
        assert _status_category(0xC000) == "Failure"

    def test_status_category_failure_a(self) -> None:
        assert _status_category(0xA000) == "Failure"

    def test_status_category_cancel(self) -> None:
        assert _status_category(0xFE00) == "Cancel"

    def test_status_category_unknown(self) -> None:
        assert _status_category(0x1234) == "Unknown"

    def test_dataset_to_dict(self) -> None:
        ds = Dataset()
        ds.PatientID = "PAT001"
        ds.PatientName = "DOE^JOHN"
        result = _dataset_to_dict(ds)
        assert result["PatientID"] == "PAT001"
        assert "DOE^JOHN" in result["PatientName"]

    def test_dataset_to_dict_skips_sequences(self) -> None:
        ds = Dataset()
        ds.PatientID = "PAT001"
        ds.ReferencedStudySequence = DicomSequence([])
        result = _dataset_to_dict(ds)
        assert "PatientID" in result
        assert "ReferencedStudySequence" not in result

    def test_build_find_dataset_basic(self) -> None:
        ds = _build_find_dataset({"PatientID": "123", "Modality": "ECG"})
        assert ds.PatientID == "123"
        assert ds.Modality == "ECG"
        assert ds.QueryRetrieveLevel == "STUDY"

    def test_build_find_dataset_return_keys(self) -> None:
        ds = _build_find_dataset({"PatientID": "123"})
        assert hasattr(ds, "StudyDate")
        assert hasattr(ds, "StudyInstanceUID")
        assert hasattr(ds, "AccessionNumber")

    def test_build_find_dataset_skips_unknown(self) -> None:
        ds = _build_find_dataset({"PatientID": "123", "FakeField": "abc"})
        assert ds.PatientID == "123"
        assert not hasattr(ds, "FakeField")

    def test_build_find_dataset_custom_level(self) -> None:
        ds = _build_find_dataset({"PatientID": "123"}, "SERIES")
        assert ds.QueryRetrieveLevel == "SERIES"


# ---------------------------------------------------------------------------
# Test: C-STORE with mock SCP
# ---------------------------------------------------------------------------


def _make_mock_ae(
    is_established: bool = True,
    store_status: int | None = 0x0000,
    store_exception: Exception | None = None,
    assoc_exception: Exception | None = None,
) -> MagicMock:
    """Create a mock AE with pre-configured association and responses."""
    mock_ae = MagicMock()
    mock_assoc = MagicMock()
    mock_assoc.is_established = is_established

    if assoc_exception:
        mock_ae.associate.side_effect = assoc_exception
    else:
        mock_ae.associate.return_value = mock_assoc

    if store_exception:
        mock_assoc.send_c_store.side_effect = store_exception
    elif store_status is not None:
        status_ds = Dataset()
        status_ds.Status = store_status
        mock_assoc.send_c_store.return_value = status_ds
    else:
        mock_assoc.send_c_store.return_value = None

    mock_ae._assoc = mock_assoc  # store ref for assertions
    return mock_ae


class TestCStore:
    def test_no_sop_class_uid(self, client: DicomClient) -> None:
        ds = Dataset()
        ds.SOPInstanceUID = generate_uid()
        result = client.c_store(ds, "REMOTE", "localhost", 4006)
        assert result.success is False
        assert "SOPClassUID" in result.error_message

    @patch(_AE_PATCH)
    def test_association_rejected(
        self, mock_ae_cls: Any, client: DicomClient, sample_sr_dataset: Dataset
    ) -> None:
        mock_ae_cls.return_value = _make_mock_ae(is_established=False)
        result = client.c_store(sample_sr_dataset, "REMOTE", "localhost", 4006)
        assert result.success is False
        assert "rejected" in result.error_message.lower()

    @patch(_AE_PATCH)
    def test_association_exception(
        self, mock_ae_cls: Any, client: DicomClient, sample_sr_dataset: Dataset
    ) -> None:
        mock_ae_cls.return_value = _make_mock_ae(
            assoc_exception=ConnectionRefusedError("refused")
        )
        result = client.c_store(sample_sr_dataset, "REMOTE", "localhost", 4006)
        assert result.success is False
        assert "Association failed" in result.error_message

    @patch(_AE_PATCH)
    def test_successful_store(
        self, mock_ae_cls: Any, client: DicomClient, sample_sr_dataset: Dataset
    ) -> None:
        mock_ae = _make_mock_ae(store_status=0x0000)
        mock_ae_cls.return_value = mock_ae
        result = client.c_store(sample_sr_dataset, "REMOTE", "localhost", 4006)
        assert result.success is True
        assert result.status_code == 0x0000
        assert result.status_category == "Success"
        assert result.sop_instance_uid == str(sample_sr_dataset.SOPInstanceUID)
        mock_ae._assoc.release.assert_called_once()

    @patch(_AE_PATCH)
    def test_store_warning_status(
        self, mock_ae_cls: Any, client: DicomClient, sample_sr_dataset: Dataset
    ) -> None:
        mock_ae_cls.return_value = _make_mock_ae(store_status=0xB000)
        result = client.c_store(sample_sr_dataset, "REMOTE", "localhost", 4006)
        assert result.success is False
        assert result.status_category == "Warning"

    @patch(_AE_PATCH)
    def test_store_no_response(
        self, mock_ae_cls: Any, client: DicomClient, sample_sr_dataset: Dataset
    ) -> None:
        mock_ae_cls.return_value = _make_mock_ae(store_status=None)
        result = client.c_store(sample_sr_dataset, "REMOTE", "localhost", 4006)
        assert result.success is False
        assert "No response" in result.error_message

    @patch(_AE_PATCH)
    def test_store_releases_on_error(
        self, mock_ae_cls: Any, client: DicomClient, sample_sr_dataset: Dataset
    ) -> None:
        mock_ae = _make_mock_ae(store_exception=RuntimeError("oops"))
        mock_ae_cls.return_value = mock_ae
        result = client.c_store(sample_sr_dataset, "REMOTE", "localhost", 4006)
        assert result.success is False
        mock_ae._assoc.release.assert_called_once()


# ---------------------------------------------------------------------------
# Test: C-STORE batch
# ---------------------------------------------------------------------------


class TestCStoreBatch:
    @patch(_AE_PATCH)
    def test_batch_all_success(
        self,
        mock_ae_cls: Any,
        client: DicomClient,
        sample_sr_dataset: Dataset,
        sample_ecg_dataset: Dataset,
    ) -> None:
        mock_ae = _make_mock_ae(store_status=0x0000)
        mock_ae_cls.return_value = mock_ae
        result = client.c_store_batch(
            [sample_sr_dataset, sample_ecg_dataset], "REMOTE", "localhost", 4006
        )
        assert result.total == 2
        assert result.succeeded == 2
        assert result.failed == 0
        assert len(result.results) == 2

    @patch(_AE_PATCH)
    def test_batch_association_failure(
        self, mock_ae_cls: Any, client: DicomClient, sample_sr_dataset: Dataset
    ) -> None:
        mock_ae_cls.return_value = _make_mock_ae(
            assoc_exception=ConnectionRefusedError("refused")
        )
        result = client.c_store_batch([sample_sr_dataset], "REMOTE", "localhost", 4006)
        assert result.failed == 1
        assert result.succeeded == 0

    @patch(_AE_PATCH)
    def test_batch_empty(self, mock_ae_cls: Any, client: DicomClient) -> None:
        result = client.c_store_batch([], "REMOTE", "localhost", 4006)
        assert result.total == 0

    @patch(_AE_PATCH)
    def test_batch_partial_failure(
        self,
        mock_ae_cls: Any,
        client: DicomClient,
        sample_sr_dataset: Dataset,
        sample_ecg_dataset: Dataset,
    ) -> None:
        mock_ae = _make_mock_ae()
        ok = Dataset()
        ok.Status = 0x0000
        fail = Dataset()
        fail.Status = 0xC000
        mock_ae._assoc.send_c_store.side_effect = [ok, fail]
        mock_ae_cls.return_value = mock_ae
        result = client.c_store_batch(
            [sample_sr_dataset, sample_ecg_dataset], "REMOTE", "localhost", 4006
        )
        assert result.succeeded == 1
        assert result.failed == 1


# ---------------------------------------------------------------------------
# Test: C-FIND with mock SCP
# ---------------------------------------------------------------------------


class TestCFind:
    @patch(_AE_PATCH)
    def test_association_rejected(
        self, mock_ae_cls: Any, client: DicomClient
    ) -> None:
        mock_ae_cls.return_value = _make_mock_ae(is_established=False)
        result = client.c_find({"PatientID": "123"}, "REMOTE", "localhost", 4006)
        assert result.success is False
        assert "rejected" in result.error_message.lower()

    @patch(_AE_PATCH)
    def test_association_exception(
        self, mock_ae_cls: Any, client: DicomClient
    ) -> None:
        mock_ae_cls.return_value = _make_mock_ae(
            assoc_exception=ConnectionRefusedError("refused")
        )
        result = client.c_find({"PatientID": "123"}, "REMOTE", "localhost", 4006)
        assert result.success is False

    @patch(_AE_PATCH)
    def test_successful_find_with_matches(
        self, mock_ae_cls: Any, client: DicomClient
    ) -> None:
        mock_ae = _make_mock_ae()
        match_ds = Dataset()
        match_ds.PatientID = "PAT001"
        match_ds.PatientName = "DOE^JOHN"
        match_ds.Modality = "ECG"
        match_ds.StudyDate = "20240101"

        pending = Dataset()
        pending.Status = 0xFF00
        success = Dataset()
        success.Status = 0x0000
        mock_ae._assoc.send_c_find.return_value = iter(
            [(pending, match_ds), (success, None)]
        )
        mock_ae_cls.return_value = mock_ae

        result = client.c_find(
            {"PatientID": "PAT001", "Modality": "ECG"}, "REMOTE", "localhost", 4006
        )
        assert result.success is True
        assert result.match_count == 1
        assert result.matches[0]["PatientID"] == "PAT001"
        mock_ae._assoc.release.assert_called_once()

    @patch(_AE_PATCH)
    def test_find_no_matches(
        self, mock_ae_cls: Any, client: DicomClient
    ) -> None:
        mock_ae = _make_mock_ae()
        success = Dataset()
        success.Status = 0x0000
        mock_ae._assoc.send_c_find.return_value = iter([(success, None)])
        mock_ae_cls.return_value = mock_ae
        result = client.c_find({"PatientID": "NOBODY"}, "REMOTE", "localhost", 4006)
        assert result.success is True
        assert result.match_count == 0

    @patch(_AE_PATCH)
    def test_find_multiple_matches(
        self, mock_ae_cls: Any, client: DicomClient
    ) -> None:
        mock_ae = _make_mock_ae()
        ds1 = Dataset()
        ds1.PatientID = "PAT001"
        ds2 = Dataset()
        ds2.PatientID = "PAT002"
        p1 = Dataset()
        p1.Status = 0xFF00
        p2 = Dataset()
        p2.Status = 0xFF00
        done = Dataset()
        done.Status = 0x0000
        mock_ae._assoc.send_c_find.return_value = iter(
            [(p1, ds1), (p2, ds2), (done, None)]
        )
        mock_ae_cls.return_value = mock_ae
        result = client.c_find({"Modality": "ECG"}, "REMOTE", "localhost", 4006)
        assert result.success is True
        assert result.match_count == 2

    @patch(_AE_PATCH)
    def test_find_error_status(
        self, mock_ae_cls: Any, client: DicomClient
    ) -> None:
        mock_ae = _make_mock_ae()
        err = Dataset()
        err.Status = 0xC000
        mock_ae._assoc.send_c_find.return_value = iter([(err, None)])
        mock_ae_cls.return_value = mock_ae
        result = client.c_find({"PatientID": "123"}, "REMOTE", "localhost", 4006)
        assert result.success is False
        assert "C-FIND error" in result.error_message

    @patch(_AE_PATCH)
    def test_find_null_status(
        self, mock_ae_cls: Any, client: DicomClient
    ) -> None:
        mock_ae = _make_mock_ae()
        mock_ae._assoc.send_c_find.return_value = iter([(None, None)])
        mock_ae_cls.return_value = mock_ae
        result = client.c_find({"PatientID": "123"}, "REMOTE", "localhost", 4006)
        assert result.success is False
        assert "Connection lost" in result.error_message

    @patch(_AE_PATCH)
    def test_find_patient_model(
        self, mock_ae_cls: Any, client: DicomClient
    ) -> None:
        mock_ae = _make_mock_ae()
        done = Dataset()
        done.Status = 0x0000
        mock_ae._assoc.send_c_find.return_value = iter([(done, None)])
        mock_ae_cls.return_value = mock_ae
        result = client.c_find(
            {"PatientID": "123"}, "REMOTE", "localhost", 4006, query_model="patient"
        )
        assert result.success is True

    @patch(_AE_PATCH)
    def test_find_releases_on_error(
        self, mock_ae_cls: Any, client: DicomClient
    ) -> None:
        mock_ae = _make_mock_ae()
        mock_ae._assoc.send_c_find.side_effect = RuntimeError("oops")
        mock_ae_cls.return_value = mock_ae
        result = client.c_find({"PatientID": "123"}, "REMOTE", "localhost", 4006)
        assert result.success is False
        mock_ae._assoc.release.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Verify (C-ECHO)
# ---------------------------------------------------------------------------


class TestVerify:
    @patch(_AE_PATCH)
    def test_verify_success(self, mock_ae_cls: Any, client: DicomClient) -> None:
        mock_ae = _make_mock_ae()
        echo_status = Dataset()
        echo_status.Status = 0x0000
        mock_ae._assoc.send_c_echo.return_value = echo_status
        mock_ae_cls.return_value = mock_ae
        assert client.verify("REMOTE", "localhost", 4006) is True

    @patch(_AE_PATCH)
    def test_verify_rejected(self, mock_ae_cls: Any, client: DicomClient) -> None:
        mock_ae_cls.return_value = _make_mock_ae(is_established=False)
        assert client.verify("REMOTE", "localhost", 4006) is False

    @patch(_AE_PATCH)
    def test_verify_exception(self, mock_ae_cls: Any, client: DicomClient) -> None:
        mock_ae_cls.return_value = _make_mock_ae(
            assoc_exception=ConnectionRefusedError()
        )
        assert client.verify("REMOTE", "localhost", 4006) is False

    @patch(_AE_PATCH)
    def test_verify_no_response(self, mock_ae_cls: Any, client: DicomClient) -> None:
        mock_ae = _make_mock_ae()
        mock_ae._assoc.send_c_echo.return_value = None
        mock_ae_cls.return_value = mock_ae
        assert client.verify("REMOTE", "localhost", 4006) is False


# ---------------------------------------------------------------------------
# Test: Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_configurable_port(self) -> None:
        c = DicomClient(local_port=9999)
        assert c.local_port == 9999

    def test_configurable_timeout(self) -> None:
        c = DicomClient(timeout=120)
        assert c.timeout == 120

    @patch(_AE_PATCH)
    def test_ae_title_passed_to_pynetdicom(
        self, mock_ae_cls: Any, sample_sr_dataset: Dataset
    ) -> None:
        c = DicomClient(local_ae_title="MY_TITLE", timeout=42)
        mock_ae_cls.return_value = _make_mock_ae(is_established=False)
        c.c_store(sample_sr_dataset, "REMOTE", "localhost", 4006)
        mock_ae_cls.assert_called_with(ae_title="MY_TITLE")
