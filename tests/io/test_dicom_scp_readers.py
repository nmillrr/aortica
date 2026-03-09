"""Tests for DICOM and SCP-ECG format readers.

All tests use synthetically constructed files — no external downloads or
real patient data are required.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pydicom
import pydicom.uid
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from aortica.io.dicom_reader import read_dicom
from aortica.io.ecg_record import ECGRecord
from aortica.io.scp_reader import read_scp

# =====================================================================
# Helpers — synthetic DICOM ECG builder
# =====================================================================


def _make_dicom_ecg(
    path: Path,
    signals: np.ndarray,
    sample_rate: float = 500.0,
    lead_names: list[str] | None = None,
    bits: int = 16,
    patient_name: str | None = None,
    patient_id: str | None = None,
    patient_sex: str | None = None,
) -> Path:
    """Create a minimal DICOM ECG waveform file for testing.

    Parameters
    ----------
    signals:
        Array of shape ``[leads, samples]`` with ECG signal data (in µV).
    """
    num_leads, num_samples = signals.shape
    if lead_names is None:
        lead_names = [f"Lead_{i}" for i in range(num_leads)]

    ds = Dataset()
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID

    if patient_name:
        ds.PatientName = patient_name
    if patient_id:
        ds.PatientID = patient_id
    if patient_sex:
        ds.PatientSex = patient_sex

    # Waveform sequence
    wf = Dataset()
    wf.WaveformOriginality = "ORIGINAL"
    wf.NumberOfWaveformChannels = num_leads
    wf.NumberOfWaveformSamples = num_samples
    wf.SamplingFrequency = str(sample_rate)
    wf.WaveformBitsAllocated = bits
    wf.WaveformSampleInterpretation = "SS"

    # Channel definitions
    channel_defs = []
    for i, name in enumerate(lead_names):
        ch = Dataset()
        ch.ChannelLabel = name

        # Source sequence
        src = Dataset()
        src.CodeValue = "5.6.3-9-1"
        src.CodingSchemeDesignator = "SCPECG"
        src.CodeMeaning = name
        ch.ChannelSourceSequence = Sequence([src])

        # Sensitivity = 1.0 µV per unit (signals are already in µV)
        ch.ChannelSensitivity = "1.0"
        ch.ChannelSensitivityCorrectionFactor = "1.0"
        ch.ChannelBaseline = "0"

        # Sensitivity units
        unit = Dataset()
        unit.CodeValue = "uV"
        unit.CodingSchemeDesignator = "UCUM"
        unit.CodeMeaning = "uV"
        ch.ChannelSensitivityUnitsSequence = Sequence([unit])

        ch.WaveformBitsStored = bits
        channel_defs.append(ch)

    wf.ChannelDefinitionSequence = Sequence(channel_defs)

    # Waveform data — interleaved [sample0_ch0, sample0_ch1, ..., sample1_ch0, ...]
    interleaved = signals.T.astype(np.int16).flatten()
    wf.WaveformData = interleaved.tobytes()

    ds.WaveformSequence = Sequence([wf])
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.save_as(str(path), write_like_original=False)
    return path


# =====================================================================
# Helpers — synthetic SCP-ECG builder
# =====================================================================


def _make_scp_ecg(
    path: Path,
    signals: np.ndarray,
    sample_rate: float = 500.0,
    lead_ids: list[int] | None = None,
    avm_nv: int = 1000,
    patient_id: str | None = None,
) -> Path:
    """Create a minimal SCP-ECG file for testing.

    Parameters
    ----------
    signals:
        Array of shape ``[leads, samples]`` with ECG signal data (in µV).
        Data will be stored as int16 after dividing by (avm_nv / 1000).
    """
    num_leads, num_samples = signals.shape
    if lead_ids is None:
        lead_ids = list(range(1, num_leads + 1))  # 1=I, 2=II, ...

    avm_uv = avm_nv / 1000.0
    int_signals = (signals / avm_uv).astype(np.int16)

    # Build sections
    # Section 1: Patient data
    sec1_body = b""
    if patient_id:
        pid_bytes = patient_id.encode("latin-1") + b"\x00"
        sec1_body += struct.pack("<B", 2) + struct.pack("<H", len(pid_bytes)) + pid_bytes

    sec1_header = _make_section_header(1, 16 + len(sec1_body))
    sec1 = sec1_header + sec1_body

    # Section 3: Lead definitions
    sec3_body = struct.pack("<B", num_leads)  # num leads
    sec3_body += struct.pack("<B", 0)  # flags
    for i, lead_id in enumerate(lead_ids):
        start_sample = 0
        end_sample = num_samples - 1
        sec3_body += struct.pack("<I", start_sample)
        sec3_body += struct.pack("<I", end_sample)
        sec3_body += struct.pack("<B", lead_id)

    sec3_header = _make_section_header(3, 16 + len(sec3_body))
    sec3 = sec3_header + sec3_body

    # Section 6: Waveform data
    sample_interval_us = int(1_000_000 / sample_rate)
    sec6_body = struct.pack("<H", avm_nv)  # AVM in nanovolts
    sec6_body += struct.pack("<H", sample_interval_us)
    sec6_body += struct.pack("<B", 0)  # encoding: real data
    sec6_body += struct.pack("<B", 0)  # compression: none

    # Interleave: [sample0_ch0, sample0_ch1, ..., sample1_ch0, ...]
    interleaved = int_signals.T.flatten()
    sec6_body += interleaved.tobytes()

    sec6_header = _make_section_header(6, 16 + len(sec6_body))
    sec6 = sec6_header + sec6_body

    # Section 0: Pointers
    # Offsets are 1-based from start of record
    global_header_size = 6
    sec0_body_size = 12 * 10  # 12 possible section pointers
    sec0_total = 16 + sec0_body_size
    sec0_header = _make_section_header(0, sec0_total)

    # Calculate offsets: global_header + sec0 + sec1 + sec3 + sec6
    sec1_offset = global_header_size + sec0_total + 1  # 1-based
    sec3_offset = sec1_offset + len(sec1)
    sec6_offset = sec3_offset + len(sec3)

    sec0_body = b""
    for sec_id in range(12):
        if sec_id == 1:
            sec0_body += struct.pack("<H", 1)
            sec0_body += struct.pack("<I", len(sec1))
            sec0_body += struct.pack("<I", sec1_offset)
        elif sec_id == 3:
            sec0_body += struct.pack("<H", 3)
            sec0_body += struct.pack("<I", len(sec3))
            sec0_body += struct.pack("<I", sec3_offset)
        elif sec_id == 6:
            sec0_body += struct.pack("<H", 6)
            sec0_body += struct.pack("<I", len(sec6))
            sec0_body += struct.pack("<I", sec6_offset)
        else:
            sec0_body += struct.pack("<H", sec_id)
            sec0_body += struct.pack("<I", 0)
            sec0_body += struct.pack("<I", 0)

    sec0 = sec0_header + sec0_body

    # Assemble the full record
    record_data = sec0 + sec1 + sec3 + sec6
    record_size = global_header_size + len(record_data)

    # Global header: CRC (2 bytes) + record size (4 bytes)
    global_header = struct.pack("<H", 0)  # CRC placeholder
    global_header += struct.pack("<I", record_size)

    full_record = global_header + record_data
    path.write_bytes(full_record)
    return path


def _make_section_header(section_id: int, total_length: int) -> bytes:
    """Build a 16-byte SCP-ECG section header."""
    header = struct.pack("<H", 0)  # CRC
    header += struct.pack("<H", section_id)
    header += struct.pack("<I", total_length)
    header += struct.pack("<B", 0)  # version
    header += struct.pack("<B", 0)  # protocol version
    header += b"\x00" * 6  # reserved
    return header


# =====================================================================
# DICOM Reader Tests
# =====================================================================


class TestReadDICOMBasic:
    """Basic DICOM ECG loading."""

    def test_basic_3_lead(self, tmp_path: Path) -> None:
        """Load a synthetic 3-lead DICOM ECG file."""
        signals = np.array(
            [[100, 200, 300, 400], [150, 250, 350, 450], [50, 100, 150, 200]],
            dtype=np.float64,
        )
        dcm_file = _make_dicom_ecg(
            tmp_path / "ecg.dcm",
            signals,
            sample_rate=500.0,
            lead_names=["I", "II", "III"],
        )
        rec = read_dicom(dcm_file)

        assert isinstance(rec, ECGRecord)
        assert rec.num_leads == 3
        assert rec.num_samples == 4
        assert rec.sample_rate == 500.0
        assert rec.lead_names == ["I", "II", "III"]
        assert rec.source_format == "dicom"
        assert rec.units == "µV"
        # Int16 round-trip may lose precision for very large values,
        # but our test values fit in int16 range
        np.testing.assert_array_almost_equal(rec.signals, signals, decimal=0)

    def test_12_lead(self, tmp_path: Path) -> None:
        """Load a synthetic 12-lead DICOM ECG file."""
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        signals = np.random.randn(12, 2500).astype(np.float64) * 100
        dcm_file = _make_dicom_ecg(
            tmp_path / "ecg12.dcm",
            signals,
            sample_rate=500.0,
            lead_names=leads,
        )
        rec = read_dicom(dcm_file)

        assert rec.num_leads == 12
        assert rec.num_samples == 2500
        assert rec.lead_names == leads

    def test_sample_rate_preserved(self, tmp_path: Path) -> None:
        """Custom sample rate is correctly read."""
        signals = np.ones((2, 100), dtype=np.float64)
        dcm_file = _make_dicom_ecg(
            tmp_path / "ecg_250.dcm", signals, sample_rate=250.0
        )
        rec = read_dicom(dcm_file)
        assert rec.sample_rate == 250.0


class TestReadDICOMMetadata:
    """DICOM patient metadata extraction."""

    def test_patient_demographics(self, tmp_path: Path) -> None:
        """Patient name, ID, and sex are extracted."""
        signals = np.ones((1, 10), dtype=np.float64)
        dcm_file = _make_dicom_ecg(
            tmp_path / "ecg_demo.dcm",
            signals,
            patient_name="Doe^John",
            patient_id="P12345",
            patient_sex="M",
        )
        rec = read_dicom(dcm_file)

        assert rec.patient_metadata is not None
        assert rec.patient_metadata["patient_name"] == "Doe^John"
        assert rec.patient_metadata["patient_id"] == "P12345"
        assert rec.patient_metadata["patient_sex"] == "M"

    def test_no_metadata(self, tmp_path: Path) -> None:
        """No patient metadata means patient_metadata is None."""
        signals = np.ones((1, 10), dtype=np.float64)
        dcm_file = _make_dicom_ecg(tmp_path / "ecg_nometa.dcm", signals)
        rec = read_dicom(dcm_file)
        assert rec.patient_metadata is None


class TestReadDICOMErrors:
    """Error handling for the DICOM reader."""

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_dicom("/nonexistent/ecg.dcm")

    def test_no_waveform_sequence(self, tmp_path: Path) -> None:
        """DICOM file without WaveformSequence raises ValueError."""
        ds = Dataset()
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        dcm_file = tmp_path / "no_waveform.dcm"
        ds.save_as(str(dcm_file), write_like_original=False)

        with pytest.raises(ValueError, match="WaveformSequence"):
            read_dicom(dcm_file)


# =====================================================================
# SCP-ECG Reader Tests
# =====================================================================


class TestReadSCPBasic:
    """Basic SCP-ECG file loading."""

    def test_basic_2_lead(self, tmp_path: Path) -> None:
        """Load a synthetic 2-lead SCP-ECG file."""
        signals = np.array(
            [[100, 200, 300, 400, 500], [150, 250, 350, 450, 550]],
            dtype=np.float64,
        )
        scp_file = _make_scp_ecg(
            tmp_path / "ecg.scp",
            signals,
            sample_rate=500.0,
            lead_ids=[1, 2],  # Lead I, Lead II
        )
        rec = read_scp(scp_file)

        assert isinstance(rec, ECGRecord)
        assert rec.num_leads == 2
        assert rec.num_samples == 5
        assert rec.sample_rate == 500.0
        assert rec.source_format == "scp-ecg"
        assert rec.units == "µV"
        assert rec.lead_names == ["I", "II"]
        np.testing.assert_array_almost_equal(rec.signals, signals, decimal=0)

    def test_12_lead(self, tmp_path: Path) -> None:
        """Load a synthetic 12-lead SCP-ECG file."""
        lead_ids = [1, 2, 17, 18, 19, 20, 3, 4, 5, 6, 7, 8]
        expected_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                          "V1", "V2", "V3", "V4", "V5", "V6"]
        signals = np.random.randn(12, 1000).astype(np.float64) * 100
        # Round to integer µV to survive int16 conversion
        signals = np.round(signals).astype(np.float64)
        scp_file = _make_scp_ecg(
            tmp_path / "ecg12.scp",
            signals,
            sample_rate=500.0,
            lead_ids=lead_ids,
        )
        rec = read_scp(scp_file)

        assert rec.num_leads == 12
        assert rec.num_samples == 1000
        assert rec.lead_names == expected_names

    def test_custom_sample_rate(self, tmp_path: Path) -> None:
        """Sample rate is extracted from section 6 interval."""
        signals = np.ones((1, 100), dtype=np.float64) * 50
        scp_file = _make_scp_ecg(
            tmp_path / "ecg_1000hz.scp", signals, sample_rate=1000.0, lead_ids=[1]
        )
        rec = read_scp(scp_file)
        assert rec.sample_rate == 1000.0


class TestReadSCPMetadata:
    """SCP-ECG patient metadata extraction."""

    def test_patient_id(self, tmp_path: Path) -> None:
        """Patient ID is extracted from Section 1."""
        signals = np.ones((1, 20), dtype=np.float64) * 100
        scp_file = _make_scp_ecg(
            tmp_path / "ecg_pid.scp",
            signals,
            patient_id="PAT-42",
            lead_ids=[1],
        )
        rec = read_scp(scp_file)

        assert rec.patient_metadata is not None
        assert rec.patient_metadata["patient_id"] == "PAT-42"

    def test_no_metadata(self, tmp_path: Path) -> None:
        """File without patient data still loads signals correctly."""
        signals = np.ones((2, 50), dtype=np.float64) * 200
        scp_file = _make_scp_ecg(
            tmp_path / "ecg_nopatient.scp",
            signals,
            lead_ids=[1, 2],
        )
        rec = read_scp(scp_file)
        # With no patient_id in Section 1, metadata might be empty
        assert rec.num_leads == 2
        assert rec.num_samples == 50


class TestReadSCPErrors:
    """Error handling for the SCP-ECG reader."""

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_scp("/nonexistent/ecg.scp")

    def test_too_small_file(self, tmp_path: Path) -> None:
        """Tiny file is rejected."""
        tiny = tmp_path / "tiny.scp"
        tiny.write_bytes(b"\x00\x00\x00")
        with pytest.raises(ValueError, match="too small"):
            read_scp(tiny)


# =====================================================================
# Integration: round-trip through ECGRecord
# =====================================================================


class TestDICOMSCPProduceValidECGRecord:
    """Both readers produce records with working ECGRecord methods."""

    def test_dicom_record_resample(self, tmp_path: Path) -> None:
        signals = np.random.randn(2, 500).astype(np.float64) * 100
        dcm_file = _make_dicom_ecg(
            tmp_path / "rt.dcm", signals, sample_rate=500.0
        )
        rec = read_dicom(dcm_file)
        resampled = rec.resample(250.0)
        assert resampled.sample_rate == 250.0
        assert resampled.num_samples == 250

    def test_dicom_record_select_leads(self, tmp_path: Path) -> None:
        signals = np.ones((3, 100), dtype=np.float64)
        dcm_file = _make_dicom_ecg(
            tmp_path / "sl.dcm",
            signals,
            lead_names=["A", "B", "C"],
        )
        rec = read_dicom(dcm_file)
        sub = rec.select_leads(["C", "A"])
        assert sub.lead_names == ["C", "A"]

    def test_scp_record_to_millivolts(self, tmp_path: Path) -> None:
        signals = np.ones((2, 100), dtype=np.float64) * 1000  # 1000 µV
        scp_file = _make_scp_ecg(
            tmp_path / "mv.scp", signals, sample_rate=500.0, lead_ids=[1, 2]
        )
        rec = read_scp(scp_file)
        mv = rec.to_millivolts()
        assert mv.units == "mV"
        np.testing.assert_array_almost_equal(mv.signals, np.ones((2, 100)), decimal=0)

    def test_dicom_record_to_millivolts(self, tmp_path: Path) -> None:
        signals = np.ones((1, 50), dtype=np.float64) * 500
        dcm_file = _make_dicom_ecg(tmp_path / "mv.dcm", signals)
        rec = read_dicom(dcm_file)
        mv = rec.to_millivolts()
        assert mv.units == "mV"
