"""Tests for SCP-ECG serial port capture (US-084)."""

from __future__ import annotations

import struct
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aortica.io.ecg_record import ECGRecord
from aortica.integration.serial_capture import (
    CaptureResult,
    CaptureTimeoutError,
    CRCError,
    FramingError,
    SCPSerialCapture,
    _CRC_INIT,
    _crc16_ccitt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_minimal_scp_frame(
    num_leads: int = 3,
    samples_per_lead: int = 100,
    sample_rate_hz: int = 500,
    set_crc: bool = True,
    corrupt_crc: bool = False,
    zero_crc: bool = False,
) -> bytes:
    """Build a minimal synthetic SCP-ECG binary frame.

    Matches the format expected by ``aortica.io.scp_reader``:
    - Global header: 2-byte CRC + 4-byte record size
    - Section 0: 16-byte section header + 10-byte pointer entries
    - Section 3: 16-byte header + lead definitions
    - Section 6: 16-byte header + waveform data
    """
    sec_header_size = 16

    # --- Section 3 body: 1 byte num_leads + 1 byte flags + 9 bytes per lead
    sec3_body = struct.pack("<B", num_leads) + b"\x00"
    for i in range(num_leads):
        start_sample = 0
        end_sample = samples_per_lead - 1
        lead_id = i + 1  # 1=I, 2=II, 3=V1 ...
        sec3_body += struct.pack("<IIB", start_sample, end_sample, lead_id)
    sec3_total = sec_header_size + len(sec3_body)

    # --- Section 6 body: 2 AVM + 2 interval + 1 encoding + 1 compression + data
    interval_us = 1_000_000 // sample_rate_hz
    avm_nv = 1000  # 1 µV per LSB
    sec6_sub_header = struct.pack("<HHBB", avm_nv, interval_us, 0, 0)
    # interleaved int16 waveform
    rng = np.random.RandomState(42)
    raw_samples = rng.randint(-500, 500, size=(samples_per_lead, num_leads), dtype=np.int16)
    waveform_bytes = raw_samples.tobytes()
    sec6_body = sec6_sub_header + waveform_bytes
    sec6_total = sec_header_size + len(sec6_body)

    # --- Section 0: pointer entries
    # Each entry: uint16 sec_id + uint32 length + uint32 offset = 10 bytes
    # scp_reader expects: struct.unpack_from("<H", data, offset)[0] for sec_id
    #                     struct.unpack_from("<I", data, offset+2)[0] for length
    #                     struct.unpack_from("<I", data, offset+6)[0] for 1-based offset
    sec0_num_entries = 2  # sections 3 and 6
    sec0_body_size = sec0_num_entries * 10
    sec0_total = sec_header_size + sec0_body_size

    offset_sec0 = 6  # right after global header
    offset_sec3 = offset_sec0 + sec0_total
    offset_sec6 = offset_sec3 + sec3_total
    total_record_size = 6 + sec0_total + sec3_total + sec6_total

    # Build section 0 body with correct format: H (sec_id) + I (length) + I (1-based offset)
    sec0_body = b""
    sec0_body += struct.pack("<HII", 3, sec3_total, offset_sec3 + 1)
    sec0_body += struct.pack("<HII", 6, sec6_total, offset_sec6 + 1)

    # Build section headers: CRC(H) + ID(H) + Length(I) + Version(B) + Protocol(B) + Reserved(6)
    def _sec_header(sec_id: int, sec_length: int) -> bytes:
        return struct.pack("<HHIBB", 0, sec_id, sec_length, 0, 0) + b"\x00" * 6

    sec0_bytes = _sec_header(0, sec0_total) + sec0_body
    sec3_bytes = _sec_header(3, sec3_total) + sec3_body
    sec6_bytes = _sec_header(6, sec6_total) + sec6_body

    # Global header: CRC (2) + record size (4), then sections
    body_after_crc = struct.pack("<I", total_record_size) + sec0_bytes + sec3_bytes + sec6_bytes

    if zero_crc:
        frame = struct.pack("<H", 0) + body_after_crc
    elif set_crc:
        crc = _crc16_ccitt(body_after_crc)
        if corrupt_crc:
            crc = (crc + 1) & 0xFFFF
        frame = struct.pack("<H", crc) + body_after_crc
    else:
        frame = struct.pack("<H", 0) + body_after_crc

    return frame


def _make_mock_serial(
    data_chunks: list[bytes],
    fail_open: bool = False,
) -> tuple[MagicMock, MagicMock]:
    """Create mock serial module and Serial instance."""
    mock_module = MagicMock()
    mock_serial_instance = MagicMock()

    # Define a real exception class for SerialException
    class MockSerialException(Exception):
        pass

    mock_module.SerialException = MockSerialException

    if fail_open:
        mock_module.Serial.side_effect = MockSerialException("Port not found")
        return mock_module, mock_serial_instance

    mock_module.Serial.return_value = mock_serial_instance

    chunk_iter = iter(data_chunks)

    def _read(size: int) -> bytes:
        try:
            return next(chunk_iter)
        except StopIteration:
            return b""

    mock_serial_instance.read = _read
    return mock_module, mock_serial_instance


# ---------------------------------------------------------------------------
# Tests: CRC-16/CCITT
# ---------------------------------------------------------------------------


class TestCRC16:
    def test_known_value(self) -> None:
        result = _crc16_ccitt(b"123456789")
        assert result == 0x29B1

    def test_empty(self) -> None:
        result = _crc16_ccitt(b"")
        assert result == _CRC_INIT

    def test_single_byte(self) -> None:
        result = _crc16_ccitt(b"\x00")
        assert isinstance(result, int)
        assert 0 <= result <= 0xFFFF


# ---------------------------------------------------------------------------
# Tests: SCPSerialCapture construction
# ---------------------------------------------------------------------------


class TestSCPSerialCaptureInit:
    def test_defaults(self) -> None:
        cap = SCPSerialCapture()
        assert cap.port == "/dev/ttyUSB0"
        assert cap.baud_rate == 115200
        assert cap.timeout == 30
        assert cap.max_retries == 3

    def test_custom_params(self) -> None:
        cap = SCPSerialCapture(
            port="COM3", baud_rate=9600, timeout=60, max_retries=5, read_chunk_size=1024
        )
        assert cap.port == "COM3"
        assert cap.baud_rate == 9600
        assert cap.timeout == 60
        assert cap.max_retries == 5
        assert cap.read_chunk_size == 1024


# ---------------------------------------------------------------------------
# Tests: CRC validation
# ---------------------------------------------------------------------------


class TestCRCValidation:
    def test_valid_crc(self) -> None:
        frame = _build_minimal_scp_frame(set_crc=True)
        assert SCPSerialCapture._validate_crc(frame) is True

    def test_corrupt_crc(self) -> None:
        frame = _build_minimal_scp_frame(corrupt_crc=True)
        assert SCPSerialCapture._validate_crc(frame) is False

    def test_zero_crc_accepted(self) -> None:
        frame = _build_minimal_scp_frame(zero_crc=True)
        assert SCPSerialCapture._validate_crc(frame) is True

    def test_too_short(self) -> None:
        assert SCPSerialCapture._validate_crc(b"\x00\x01") is False


# ---------------------------------------------------------------------------
# Tests: Successful capture
# ---------------------------------------------------------------------------


class TestCaptureSuccess:
    def test_single_chunk_capture(self) -> None:
        frame = _build_minimal_scp_frame(num_leads=3, samples_per_lead=100)
        mock_mod, _ = _make_mock_serial([frame])

        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=5)
        result = cap._capture_once(mock_mod, effective_timeout=5)

        assert result.success is True
        assert result.ecg_record is not None
        assert result.crc_valid is True
        assert result.bytes_received == len(frame)
        assert result.ecg_record.source_format == "scp-ecg-serial"
        assert result.ecg_record.num_leads == 3

    def test_multi_chunk_capture(self) -> None:
        frame = _build_minimal_scp_frame(num_leads=3, samples_per_lead=50)
        mid = len(frame) // 2
        chunks = [frame[:mid], frame[mid:]]
        mock_mod, _ = _make_mock_serial(chunks)

        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=5)
        result = cap._capture_once(mock_mod, effective_timeout=5)

        assert result.success is True
        assert result.ecg_record is not None

    def test_capture_with_retries_success(self) -> None:
        frame = _build_minimal_scp_frame()
        mock_mod, _ = _make_mock_serial([frame])

        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=5)
        result = cap._capture_with_retries(mock_mod, effective_timeout=5)

        assert result.success is True

    def test_capture_records_duration(self) -> None:
        frame = _build_minimal_scp_frame()
        mock_mod, _ = _make_mock_serial([frame])

        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=5)
        result = cap._capture_once(mock_mod, effective_timeout=5)

        assert result.capture_duration_seconds >= 0


# ---------------------------------------------------------------------------
# Tests: Failure scenarios
# ---------------------------------------------------------------------------


class TestCaptureFailures:
    def test_port_open_failure(self) -> None:
        mock_mod, _ = _make_mock_serial([], fail_open=True)

        cap = SCPSerialCapture(port="/dev/nonexistent", timeout=2)
        result = cap._capture_once(mock_mod, effective_timeout=2)

        assert result.success is False
        assert "port" in result.error_message.lower()

    def test_timeout_no_data(self) -> None:
        mock_mod, mock_ser = _make_mock_serial([])
        mock_ser.read = lambda size: b""
        mock_mod.Serial.return_value = mock_ser

        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=1)
        result = cap._capture_once(mock_mod, effective_timeout=1)

        assert result.success is False
        assert "timeout" in result.error_message.lower()

    def test_crc_failure_triggers_retry(self) -> None:
        good_frame = _build_minimal_scp_frame(set_crc=True)

        # Pre-parse the good frame to get the expected ECGRecord
        good_record = SCPSerialCapture._parse_scp_bytes(good_frame)

        call_count = 0

        def mock_capture_once(
            self_ref: Any, serial_mod: Any, timeout: int
        ) -> CaptureResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return CaptureResult(
                    success=False,
                    error_message="CRC validation failed",
                    crc_valid=False,
                )
            # Second attempt succeeds
            return CaptureResult(
                success=True,
                ecg_record=good_record,
                raw_bytes=good_frame,
                bytes_received=len(good_frame),
                crc_valid=True,
            )

        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=5, max_retries=3)

        with patch.object(SCPSerialCapture, "_capture_once", mock_capture_once):
            result = cap._capture_with_retries(MagicMock(), effective_timeout=5)

        assert result.success is True
        assert result.retries == 1

    def test_invalid_record_size_too_small(self) -> None:
        bad_header = struct.pack("<HI", 0, 5) + b"\x00" * 20
        mock_mod, _ = _make_mock_serial([bad_header])

        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=5)
        result = cap._capture_once(mock_mod, effective_timeout=5)

        assert result.success is False
        assert "invalid" in result.error_message.lower() or "minimum" in result.error_message.lower()

    def test_invalid_record_size_too_large(self) -> None:
        huge_size = 100 * 1024 * 1024
        bad_header = struct.pack("<HI", 0, huge_size) + b"\x00" * 20
        mock_mod, _ = _make_mock_serial([bad_header])

        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=5)
        result = cap._capture_once(mock_mod, effective_timeout=5)

        assert result.success is False
        assert "exceeds" in result.error_message.lower() or "maximum" in result.error_message.lower()

    def test_all_retries_exhausted(self) -> None:
        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=5, max_retries=2)

        def always_fail(
            self_ref: Any, serial_mod: Any, timeout: int
        ) -> CaptureResult:
            return CaptureResult(success=False, error_message="CRC validation failed")

        with patch.object(SCPSerialCapture, "_capture_once", always_fail):
            result = cap._capture_with_retries(MagicMock(), effective_timeout=5)

        assert result.success is False
        assert "retries exhausted" in result.error_message.lower()

    def test_timeout_no_retry(self) -> None:
        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=5, max_retries=3)
        call_count = 0

        def timeout_fail(
            self_ref: Any, serial_mod: Any, timeout: int
        ) -> CaptureResult:
            nonlocal call_count
            call_count += 1
            return CaptureResult(success=False, error_message="Capture timeout after 5s")

        with patch.object(SCPSerialCapture, "_capture_once", timeout_fail):
            result = cap._capture_with_retries(MagicMock(), effective_timeout=5)

        assert result.success is False
        assert call_count == 1


# ---------------------------------------------------------------------------
# Tests: listen() exception raising
# ---------------------------------------------------------------------------


class TestListenExceptions:
    def test_listen_raises_timeout(self) -> None:
        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=1)

        def timeout_result(
            self_ref: Any, serial_module: Any, effective_timeout: int
        ) -> CaptureResult:
            return CaptureResult(success=False, error_message="Capture timeout after 1s")

        with patch.object(SCPSerialCapture, "_get_serial_module", return_value=MagicMock()), \
             patch.object(SCPSerialCapture, "_capture_with_retries", timeout_result):
            with pytest.raises(CaptureTimeoutError, match="timeout"):
                cap.listen(timeout=1)

    def test_listen_raises_crc_error(self) -> None:
        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=1)

        def crc_result(
            self_ref: Any, serial_module: Any, effective_timeout: int
        ) -> CaptureResult:
            return CaptureResult(success=False, error_message="CRC validation failed")

        with patch.object(SCPSerialCapture, "_get_serial_module", return_value=MagicMock()), \
             patch.object(SCPSerialCapture, "_capture_with_retries", crc_result):
            with pytest.raises(CRCError):
                cap.listen(timeout=1)

    def test_listen_raises_framing_error(self) -> None:
        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=1)

        def framing_result(
            self_ref: Any, serial_module: Any, effective_timeout: int
        ) -> CaptureResult:
            return CaptureResult(
                success=False, error_message="Failed to parse SCP-ECG data"
            )

        with patch.object(SCPSerialCapture, "_get_serial_module", return_value=MagicMock()), \
             patch.object(SCPSerialCapture, "_capture_with_retries", framing_result):
            with pytest.raises(FramingError):
                cap.listen(timeout=1)

    def test_listen_returns_record_on_success(self) -> None:
        frame = _build_minimal_scp_frame()
        mock_mod, _ = _make_mock_serial([frame])

        cap = SCPSerialCapture(port="/dev/ttyUSB0", timeout=5)

        def success_result(
            self_ref: Any, serial_module: Any, effective_timeout: int
        ) -> CaptureResult:
            return cap._capture_once(mock_mod, effective_timeout=effective_timeout)

        with patch.object(SCPSerialCapture, "_get_serial_module", return_value=mock_mod), \
             patch.object(SCPSerialCapture, "_capture_with_retries", success_result):
            record = cap.listen(timeout=5)

        assert record is not None
        assert isinstance(record, ECGRecord)
        assert record.source_format == "scp-ecg-serial"


# ---------------------------------------------------------------------------
# Tests: CaptureResult dataclass
# ---------------------------------------------------------------------------


class TestCaptureResult:
    def test_defaults(self) -> None:
        r = CaptureResult()
        assert r.success is False
        assert r.ecg_record is None
        assert r.raw_bytes == b""
        assert r.bytes_received == 0
        assert r.crc_valid is False
        assert r.retries == 0

    def test_with_values(self) -> None:
        r = CaptureResult(
            success=True,
            bytes_received=1024,
            capture_duration_seconds=1.5,
            crc_valid=True,
            retries=2,
        )
        assert r.success is True
        assert r.bytes_received == 1024
        assert r.capture_duration_seconds == 1.5
        assert r.retries == 2


# ---------------------------------------------------------------------------
# Tests: SCP-ECG frame parsing
# ---------------------------------------------------------------------------


class TestSCPParsing:
    def test_parse_valid_frame(self) -> None:
        frame = _build_minimal_scp_frame(num_leads=3, samples_per_lead=100)
        record = SCPSerialCapture._parse_scp_bytes(frame)

        assert isinstance(record, ECGRecord)
        assert record.source_format == "scp-ecg-serial"
        assert record.num_leads == 3
        assert record.num_samples == 100

    def test_parse_single_lead(self) -> None:
        frame = _build_minimal_scp_frame(num_leads=1, samples_per_lead=50)
        record = SCPSerialCapture._parse_scp_bytes(frame)
        assert record.num_leads == 1
        assert record.num_samples == 50

    def test_parse_12_lead(self) -> None:
        frame = _build_minimal_scp_frame(num_leads=12, samples_per_lead=50)
        record = SCPSerialCapture._parse_scp_bytes(frame)
        assert record.num_leads == 12

    def test_parse_preserves_signal_shape(self) -> None:
        frame = _build_minimal_scp_frame(num_leads=3, samples_per_lead=200)
        record = SCPSerialCapture._parse_scp_bytes(frame)
        assert record.signals.shape == (3, 200)
