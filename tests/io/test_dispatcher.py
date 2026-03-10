"""Tests for the Universal Format Dispatcher (US-008).

Covers auto-detection by extension, magic bytes, explicit format parameter,
UnsupportedFormatError, 12-lead normalisation, and resampling.
"""

from __future__ import annotations

import struct
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.io import savemat

from aortica.io import (
    STANDARD_12_LEAD_ORDER,
    UnsupportedFormatError,
    read_ecg,
)
from aortica.io.csv_reader import CSVConfig
from aortica.io.dispatcher import _normalise_leads, _sniff_format
from aortica.io.ecg_record import ECGRecord
from aortica.io.mat_reader import MATConfig


# ── Helpers ───────────────────────────────────────────────────────────

def _make_record(
    lead_names: list[str] | None = None,
    sample_rate: float = 500.0,
    n_samples: int = 5000,
    source_format: str = "test",
) -> ECGRecord:
    """Create a minimal ECGRecord for testing."""
    if lead_names is None:
        lead_names = ["I", "II"]
    signals = np.random.default_rng(42).standard_normal(
        (len(lead_names), n_samples)
    )
    return ECGRecord(
        signals=signals,
        sample_rate=sample_rate,
        lead_names=lead_names,
        source_format=source_format,
    )


def _write_csv_file(path: Path, leads: int = 2, samples: int = 100) -> None:
    """Write a minimal CSV file (column-per-lead)."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((samples, leads))
    header = ",".join(f"lead_{i}" for i in range(leads))
    lines = [header]
    for row in data:
        lines.append(",".join(f"{v:.6f}" for v in row))
    path.write_text("\n".join(lines))


def _write_mat_file(path: Path, leads: int = 2, samples: int = 100) -> None:
    """Write a minimal MAT file."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((leads, samples))
    savemat(str(path), {"signals": data})


def _write_dicom_file(path: Path) -> None:
    """Write a minimal DICOM file with correct magic bytes."""
    content = b"\x00" * 128 + b"DICM" + b"\x00" * 100
    path.write_bytes(content)


def _write_scp_file(path: Path) -> None:
    """Write a minimal SCP-ECG-like file header."""
    # CRC (2 bytes) + record length (4 bytes) + section id 0 (1 byte) + padding
    crc = struct.pack("<H", 0xFFFF)
    rec_len = struct.pack("<I", 5000)
    section_id = b"\x00"
    padding = b"\x00" * 100
    path.write_bytes(crc + rec_len + section_id + padding)


def _write_hl7_xml(path: Path) -> None:
    """Write a minimal HL7 aECG XML file."""
    xml = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <AnnotatedECG xmlns="urn:hl7-org:v3">
      <id root="test-record"/>
    </AnnotatedECG>
    """)
    path.write_text(xml)


def _write_generic_xml(path: Path) -> None:
    """Write a non-HL7 XML file."""
    xml = '<?xml version="1.0"?>\n<data><item>hello</item></data>'
    path.write_text(xml)


# ══════════════════════════════════════════════════════════════════════
# UnsupportedFormatError
# ══════════════════════════════════════════════════════════════════════


class TestUnsupportedFormatError:
    def test_raised_for_unknown_extension(self, tmp_path: Path) -> None:
        unknown = tmp_path / "data.xyz"
        unknown.write_text("not an ecg")
        with pytest.raises(UnsupportedFormatError, match="Cannot detect format"):
            read_ecg(unknown)

    def test_raised_for_invalid_explicit_format(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2")
        with pytest.raises(UnsupportedFormatError, match="Unsupported format"):
            read_ecg(f, format="mp3")

    def test_is_exception(self) -> None:
        assert issubclass(UnsupportedFormatError, Exception)


# ══════════════════════════════════════════════════════════════════════
# Magic-byte sniffing
# ══════════════════════════════════════════════════════════════════════


class TestMagicByteSniffing:
    def test_dicom_magic(self, tmp_path: Path) -> None:
        p = tmp_path / "ecg.bin"
        _write_dicom_file(p)
        assert _sniff_format(p) == "dicom"

    def test_scp_magic(self, tmp_path: Path) -> None:
        p = tmp_path / "ecg.bin"
        _write_scp_file(p)
        assert _sniff_format(p) == "scp"

    def test_hl7_xml_magic(self, tmp_path: Path) -> None:
        p = tmp_path / "ecg.bin"
        _write_hl7_xml(p)
        assert _sniff_format(p) == "hl7_aecg"

    def test_generic_xml_magic(self, tmp_path: Path) -> None:
        p = tmp_path / "ecg.bin"
        _write_generic_xml(p)
        assert _sniff_format(p) == "xml"

    def test_unknown_bytes(self, tmp_path: Path) -> None:
        p = tmp_path / "ecg.bin"
        p.write_bytes(b"random garbage data here")
        assert _sniff_format(p) is None

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "ecg.bin"
        p.write_bytes(b"")
        assert _sniff_format(p) is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        p = tmp_path / "nope.bin"
        assert _sniff_format(p) is None


# ══════════════════════════════════════════════════════════════════════
# Extension-based dispatch
# ══════════════════════════════════════════════════════════════════════


class TestExtensionDispatch:
    """Verify that the correct reader is called based on file extension.

    We mock the individual readers and assert they are called.
    """

    @patch("aortica.io.dispatcher.read_wfdb")
    def test_hea_extension(self, mock_reader: MagicMock, tmp_path: Path) -> None:
        mock_reader.return_value = _make_record(source_format="wfdb")
        p = tmp_path / "record.hea"
        p.write_text("dummy header")
        result = read_ecg(p, resample=False)
        mock_reader.assert_called_once()
        assert result.source_format == "wfdb"

    @patch("aortica.io.dispatcher.read_csv")
    def test_csv_extension(self, mock_reader: MagicMock, tmp_path: Path) -> None:
        mock_reader.return_value = _make_record(source_format="csv")
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2")
        result = read_ecg(p, csv_config=CSVConfig(sample_rate=500.0), resample=False)
        mock_reader.assert_called_once()
        assert result.source_format == "csv"

    @patch("aortica.io.dispatcher.read_mat")
    def test_mat_extension(self, mock_reader: MagicMock, tmp_path: Path) -> None:
        mock_reader.return_value = _make_record(source_format="mat")
        p = tmp_path / "data.mat"
        _write_mat_file(p)
        result = read_ecg(p, mat_config=MATConfig(sample_rate=500.0), resample=False)
        mock_reader.assert_called_once()
        assert result.source_format == "mat"

    @patch("aortica.io.dispatcher.read_dicom")
    def test_dcm_extension(self, mock_reader: MagicMock, tmp_path: Path) -> None:
        mock_reader.return_value = _make_record(source_format="dicom")
        p = tmp_path / "ecg.dcm"
        _write_dicom_file(p)
        result = read_ecg(p, resample=False)
        mock_reader.assert_called_once()
        assert result.source_format == "dicom"

    @patch("aortica.io.dispatcher.read_scp")
    def test_scp_extension(self, mock_reader: MagicMock, tmp_path: Path) -> None:
        mock_reader.return_value = _make_record(source_format="scp")
        p = tmp_path / "ecg.scp"
        _write_scp_file(p)
        result = read_ecg(p, resample=False)
        mock_reader.assert_called_once()
        assert result.source_format == "scp"

    @patch("aortica.io.dispatcher.read_hl7_aecg")
    def test_xml_extension_hl7(self, mock_reader: MagicMock, tmp_path: Path) -> None:
        mock_reader.return_value = _make_record(source_format="hl7_aecg")
        p = tmp_path / "ecg.xml"
        _write_hl7_xml(p)
        result = read_ecg(p, resample=False)
        # Magic bytes detect hl7_aecg for XML with the namespace
        mock_reader.assert_called_once()
        assert result.source_format == "hl7_aecg"


# ══════════════════════════════════════════════════════════════════════
# Explicit format parameter
# ══════════════════════════════════════════════════════════════════════


class TestExplicitFormat:
    @patch("aortica.io.dispatcher.read_csv")
    def test_explicit_csv_overrides_extension(
        self, mock_reader: MagicMock, tmp_path: Path
    ) -> None:
        mock_reader.return_value = _make_record(source_format="csv")
        p = tmp_path / "data.txt"  # wrong extension
        p.write_text("a,b\n1,2")
        result = read_ecg(p, format="csv", csv_config=CSVConfig(sample_rate=500.0), resample=False)
        mock_reader.assert_called_once()
        assert result.source_format == "csv"

    @patch("aortica.io.dispatcher.read_dicom")
    def test_explicit_dicom(self, mock_reader: MagicMock, tmp_path: Path) -> None:
        mock_reader.return_value = _make_record(source_format="dicom")
        p = tmp_path / "ecg.bin"
        p.write_bytes(b"\x00" * 200)
        result = read_ecg(p, format="dicom", resample=False)
        mock_reader.assert_called_once()
        assert result.source_format == "dicom"


# ══════════════════════════════════════════════════════════════════════
# WFDB base-path (extensionless) detection
# ══════════════════════════════════════════════════════════════════════


class TestWFDBBasePath:
    @patch("aortica.io.dispatcher.read_wfdb")
    def test_wfdb_without_extension(
        self, mock_reader: MagicMock, tmp_path: Path
    ) -> None:
        mock_reader.return_value = _make_record(source_format="wfdb")
        # Create a .hea sibling so that the dispatcher finds it
        (tmp_path / "record.hea").write_text("dummy header")
        result = read_ecg(tmp_path / "record", resample=False)
        mock_reader.assert_called_once()
        assert result.source_format == "wfdb"


# ══════════════════════════════════════════════════════════════════════
# 12-lead normalisation
# ══════════════════════════════════════════════════════════════════════


class TestLeadNormalisation:
    def test_reorders_shuffled_12_leads(self) -> None:
        shuffled = list(reversed(STANDARD_12_LEAD_ORDER))
        record = _make_record(lead_names=shuffled, n_samples=100)
        normalised = _normalise_leads(record)
        assert normalised.lead_names == STANDARD_12_LEAD_ORDER

    def test_preserves_signal_alignment(self) -> None:
        """Signals must follow their lead names after reordering."""
        shuffled = list(reversed(STANDARD_12_LEAD_ORDER))
        rng = np.random.default_rng(99)
        signals = rng.standard_normal((12, 100))
        record = ECGRecord(
            signals=signals,
            sample_rate=500.0,
            lead_names=shuffled,
            source_format="test",
        )
        normalised = _normalise_leads(record)
        # After normalisation, lead "I" should map to the same data
        orig_idx = shuffled.index("I")
        new_idx = STANDARD_12_LEAD_ORDER.index("I")
        np.testing.assert_array_equal(
            normalised.signals[new_idx], signals[orig_idx]
        )

    def test_already_standard_order_unchanged(self) -> None:
        record = _make_record(lead_names=list(STANDARD_12_LEAD_ORDER), n_samples=100)
        normalised = _normalise_leads(record)
        assert normalised.lead_names == STANDARD_12_LEAD_ORDER
        np.testing.assert_array_equal(normalised.signals, record.signals)

    def test_non_12_lead_untouched(self) -> None:
        record = _make_record(lead_names=["I", "II", "V1"], n_samples=100)
        normalised = _normalise_leads(record)
        assert normalised.lead_names == ["I", "II", "V1"]

    def test_nonstandard_lead_names_untouched(self) -> None:
        names = [f"ch{i}" for i in range(12)]
        record = _make_record(lead_names=names, n_samples=100)
        normalised = _normalise_leads(record)
        assert normalised.lead_names == names

    def test_case_insensitive_matching(self) -> None:
        """Lead names like 'avr' or 'AVR' should still be normalised."""
        mixed_case = ["i", "ii", "iii", "avr", "avl", "avf",
                       "v1", "v2", "v3", "v4", "v5", "v6"]
        record = _make_record(lead_names=mixed_case, n_samples=100)
        normalised = _normalise_leads(record)
        # Leads are reordered to standard order and given standard casing
        assert normalised.lead_names == STANDARD_12_LEAD_ORDER
        assert normalised.num_leads == 12


# ══════════════════════════════════════════════════════════════════════
# Resampling
# ══════════════════════════════════════════════════════════════════════


class TestResampling:
    @patch("aortica.io.dispatcher.read_csv")
    def test_resamples_to_target_rate(
        self, mock_reader: MagicMock, tmp_path: Path
    ) -> None:
        record = _make_record(sample_rate=250.0, n_samples=2500)
        mock_reader.return_value = record
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2")
        result = read_ecg(p, csv_config=CSVConfig(sample_rate=250.0), target_rate=500.0)
        assert result.sample_rate == 500.0
        assert result.num_samples == 5000  # doubled

    @patch("aortica.io.dispatcher.read_csv")
    def test_no_resample_when_disabled(
        self, mock_reader: MagicMock, tmp_path: Path
    ) -> None:
        record = _make_record(sample_rate=250.0, n_samples=2500)
        mock_reader.return_value = record
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2")
        result = read_ecg(p, csv_config=CSVConfig(sample_rate=250.0), resample=False)
        assert result.sample_rate == 250.0

    @patch("aortica.io.dispatcher.read_csv")
    def test_no_resample_when_already_at_target(
        self, mock_reader: MagicMock, tmp_path: Path
    ) -> None:
        record = _make_record(sample_rate=500.0, n_samples=5000)
        mock_reader.return_value = record
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2")
        result = read_ecg(p, csv_config=CSVConfig(sample_rate=500.0), target_rate=500.0)
        assert result.sample_rate == 500.0
        assert result.num_samples == 5000

    @patch("aortica.io.dispatcher.read_csv")
    def test_custom_target_rate(
        self, mock_reader: MagicMock, tmp_path: Path
    ) -> None:
        record = _make_record(sample_rate=500.0, n_samples=5000)
        mock_reader.return_value = record
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2")
        result = read_ecg(p, csv_config=CSVConfig(sample_rate=500.0), target_rate=250.0)
        assert result.sample_rate == 250.0
        assert result.num_samples == 2500


# ══════════════════════════════════════════════════════════════════════
# Integration: full pipeline (mock readers)
# ══════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    @patch("aortica.io.dispatcher.read_wfdb")
    def test_wfdb_normalised_and_resampled(
        self, mock_reader: MagicMock, tmp_path: Path
    ) -> None:
        """12-lead WFDB record should be normalised and resampled."""
        shuffled = list(reversed(STANDARD_12_LEAD_ORDER))
        record = _make_record(
            lead_names=shuffled, sample_rate=250.0, n_samples=2500,
            source_format="wfdb",
        )
        mock_reader.return_value = record
        p = tmp_path / "record.hea"
        p.write_text("dummy")
        result = read_ecg(p, target_rate=500.0)
        assert result.lead_names == STANDARD_12_LEAD_ORDER
        assert result.sample_rate == 500.0
        assert result.num_samples == 5000

    @patch("aortica.io.dispatcher.read_csv")
    def test_csv_config_forwarded(
        self, mock_reader: MagicMock, tmp_path: Path
    ) -> None:
        """CSVConfig should be forwarded to the CSV reader."""
        mock_reader.return_value = _make_record(source_format="csv")
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2")
        cfg = CSVConfig(sample_rate=1000.0)
        read_ecg(p, csv_config=cfg, resample=False)
        # read_csv is called with positional path and keyword config
        call_args = mock_reader.call_args
        assert call_args[1]["config"].sample_rate == 1000.0

    @patch("aortica.io.dispatcher.read_mat")
    def test_mat_config_forwarded(
        self, mock_reader: MagicMock, tmp_path: Path
    ) -> None:
        """MATConfig should be forwarded to the MAT reader."""
        mock_reader.return_value = _make_record(source_format="mat")
        p = tmp_path / "data.mat"
        _write_mat_file(p)
        cfg = MATConfig(sample_rate=500.0, signal_variable="ecg_data")
        read_ecg(p, mat_config=cfg, resample=False)
        call_args = mock_reader.call_args
        assert call_args[1]["config"].signal_variable == "ecg_data"


# ══════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_nonexistent_file_unknown_extension(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            read_ecg("/nonexistent/path/data.xyz")

    @patch("aortica.io.dispatcher.read_csv")
    def test_string_path_accepted(
        self, mock_reader: MagicMock, tmp_path: Path
    ) -> None:
        mock_reader.return_value = _make_record(source_format="csv")
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2")
        result = read_ecg(str(p), csv_config=CSVConfig(sample_rate=500.0), resample=False)
        assert result.source_format == "csv"

    def test_standard_12_lead_order_contents(self) -> None:
        assert len(STANDARD_12_LEAD_ORDER) == 12
        assert STANDARD_12_LEAD_ORDER[0] == "I"
        assert STANDARD_12_LEAD_ORDER[-1] == "V6"
