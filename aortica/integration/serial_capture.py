"""SCP-ECG serial port capture for legacy ECG cart integration.

Provides a high-level ``SCPSerialCapture`` class that listens on a serial
(or USB-serial) port for incoming SCP-ECG frames transmitted by legacy
ECG carts.  Received bytes are parsed into
:class:`~aortica.io.ecg_record.ECGRecord` objects via the existing
``read_scp()`` reader.

Legacy ECG carts (e.g. Nihon Kohden Cardiofax, Schiller AT-series) can
be configured to output SCP-ECG over RS-232 or USB-serial after each
acquisition.  This module wraps ``pyserial`` to handle the byte-level
framing, CRC-16/CCITT validation, and incomplete-transmission retries.

Requires ``pyserial``::

    pip install pyserial

Example usage::

    from aortica.integration.serial_capture import SCPSerialCapture

    capture = SCPSerialCapture(port="/dev/ttyUSB0", baud_rate=115200)
    ecg_record = capture.listen(timeout=60)
    print(ecg_record.lead_names, ecg_record.sample_rate)
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aortica.io.ecg_record import ECGRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class CaptureTimeoutError(Exception):
    """Raised when no complete SCP-ECG frame is received within the timeout."""


class CRCError(Exception):
    """Raised when CRC validation fails on a received SCP-ECG frame."""


class FramingError(Exception):
    """Raised when the received data cannot be parsed as a valid SCP-ECG frame."""


# ---------------------------------------------------------------------------
# SCP-ECG frame constants
# ---------------------------------------------------------------------------

# SCP-ECG global header: 2-byte CRC + 4-byte record size = 6 bytes minimum
_SCP_HEADER_SIZE = 6

# CRC-16/CCITT polynomial used by SCP-ECG
_CRC_POLY = 0x1021
_CRC_INIT = 0xFFFF

# Minimum valid SCP-ECG record size (header + at least one section)
_MIN_RECORD_SIZE = 22  # 6 header + 16 section header

# Maximum reasonable SCP-ECG record size (64 MiB)
_MAX_RECORD_SIZE = 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# CRC-16/CCITT
# ---------------------------------------------------------------------------


def _crc16_ccitt(data: bytes, init: int = _CRC_INIT) -> int:
    """Compute CRC-16/CCITT over *data*.

    Uses the polynomial ``0x1021`` with an initial value of ``0xFFFF``,
    as specified in EN 1064 / ISO 11073-91064 for SCP-ECG records.

    Args:
        data: Bytes to checksum.
        init: Initial CRC register value.

    Returns:
        16-bit CRC value.
    """
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ _CRC_POLY
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CaptureResult:
    """Result of a serial port capture attempt.

    Attributes:
        success: Whether a valid SCP-ECG frame was captured.
        ecg_record: The parsed :class:`ECGRecord` if successful.
        raw_bytes: The raw captured bytes (for debugging).
        bytes_received: Total number of bytes received.
        capture_duration_seconds: Time elapsed during capture.
        error_message: Error description if the capture failed.
        crc_valid: Whether the CRC check passed.
        retries: Number of retries performed before success or failure.
    """

    success: bool = False
    ecg_record: Optional[ECGRecord] = None
    raw_bytes: bytes = b""
    bytes_received: int = 0
    capture_duration_seconds: float = 0.0
    error_message: str = ""
    crc_valid: bool = False
    retries: int = 0


# ---------------------------------------------------------------------------
# SCPSerialCapture
# ---------------------------------------------------------------------------


class SCPSerialCapture:
    """SCP-ECG serial port capture for legacy ECG carts.

    Listens on a serial (or USB-serial) port for incoming SCP-ECG frames,
    validates the CRC, and parses the waveform data into an
    :class:`~aortica.io.ecg_record.ECGRecord`.

    Args:
        port: Serial port path (e.g. ``/dev/ttyUSB0``, ``COM3``).
        baud_rate: Serial baud rate.  Default ``115200``.
        timeout: Per-read timeout in seconds.  Default ``30``.
        max_retries: Number of retries on incomplete/corrupt frames.
            Default ``3``.
        read_chunk_size: Bytes to read per serial read call.
            Default ``4096``.

    Example::

        cap = SCPSerialCapture(port="/dev/ttyUSB0")
        record = cap.listen(timeout=60)
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,
        timeout: int = 30,
        max_retries: int = 3,
        read_chunk_size: int = 4096,
    ) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.max_retries = max_retries
        self.read_chunk_size = read_chunk_size

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    @staticmethod
    def _get_serial_module() -> object:
        """Lazily import and return the ``serial`` module."""
        import serial  # type: ignore[import-untyped]
        return serial

    def listen(
        self,
        timeout: Optional[int] = None,
    ) -> ECGRecord:
        """Block until a complete SCP-ECG frame is received.

        Opens the configured serial port, reads bytes until a complete
        SCP-ECG record is assembled (determined by the 4-byte record
        length field in the SCP-ECG global header), validates the CRC,
        and parses the frame into an :class:`ECGRecord`.

        Args:
            timeout: Overall capture timeout in seconds.  Overrides
                the instance-level ``self.timeout`` if provided.

        Returns:
            An :class:`ECGRecord` parsed from the received SCP-ECG data.

        Raises:
            CaptureTimeoutError: If no complete frame is received
                within the timeout.
            CRCError: If CRC validation fails after all retries.
            FramingError: If the received data cannot be parsed.
        """
        serial_mod = self._get_serial_module()
        effective_timeout = timeout if timeout is not None else self.timeout

        result = self._capture_with_retries(
            serial_module=serial_mod,
            effective_timeout=effective_timeout,
        )

        if result.success and result.ecg_record is not None:
            return result.ecg_record

        # Raise the appropriate exception based on failure mode
        if "timeout" in result.error_message.lower():
            raise CaptureTimeoutError(result.error_message)
        elif "crc" in result.error_message.lower():
            raise CRCError(result.error_message)
        else:
            raise FramingError(result.error_message)

    def listen_with_result(
        self,
        timeout: Optional[int] = None,
    ) -> CaptureResult:
        """Listen for SCP-ECG frame and return a detailed result.

        Like :meth:`listen` but returns a :class:`CaptureResult` instead
        of raising exceptions, making it suitable for non-critical
        capture pipelines.

        Args:
            timeout: Overall capture timeout in seconds.

        Returns:
            A :class:`CaptureResult` with capture details.
        """
        serial_mod = self._get_serial_module()
        effective_timeout = timeout if timeout is not None else self.timeout

        return self._capture_with_retries(
            serial_module=serial_mod,
            effective_timeout=effective_timeout,
        )

    # ---------------------------------------------------------------
    # Internal capture logic
    # ---------------------------------------------------------------

    def _capture_with_retries(
        self,
        serial_module: object,
        effective_timeout: int,
    ) -> CaptureResult:
        """Attempt capture with retries on CRC / framing failures."""
        last_error = ""
        total_retries = 0

        for attempt in range(1 + self.max_retries):
            if attempt > 0:
                total_retries = attempt
                logger.warning(
                    "Retry %d/%d after failed capture: %s",
                    attempt,
                    self.max_retries,
                    last_error,
                )

            result = self._capture_once(serial_module, effective_timeout)
            result.retries = total_retries

            if result.success:
                return result

            last_error = result.error_message

            # Don't retry on timeout — it means no data at all
            if "timeout" in last_error.lower():
                return result

        # All retries exhausted
        return CaptureResult(
            success=False,
            error_message=f"All {self.max_retries} retries exhausted. Last error: {last_error}",
            retries=total_retries,
        )

    def _capture_once(
        self,
        serial_module: object,
        effective_timeout: int,
    ) -> CaptureResult:
        """Perform a single capture attempt."""
        Serial = getattr(serial_module, "Serial")  # noqa: N806
        SerialException = getattr(  # noqa: N806
            serial_module, "SerialException", Exception
        )

        start_time = time.monotonic()
        buffer = bytearray()

        try:
            ser = Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1,  # per-read timeout (1 second)
            )
        except SerialException as exc:
            return CaptureResult(
                success=False,
                error_message=f"Failed to open serial port {self.port}: {exc}",
                capture_duration_seconds=time.monotonic() - start_time,
            )

        try:
            expected_size: Optional[int] = None

            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= effective_timeout:
                    return CaptureResult(
                        success=False,
                        raw_bytes=bytes(buffer),
                        bytes_received=len(buffer),
                        capture_duration_seconds=elapsed,
                        error_message=(
                            f"Capture timeout after {effective_timeout}s "
                            f"({len(buffer)} bytes received, "
                            f"{'expected ' + str(expected_size) if expected_size else 'header not yet received'})"
                        ),
                    )

                # Read available data
                chunk = ser.read(self.read_chunk_size)
                if chunk:
                    buffer.extend(chunk)
                    logger.debug("Read %d bytes (total: %d)", len(chunk), len(buffer))

                # Once we have at least 6 bytes, parse the record size
                if expected_size is None and len(buffer) >= _SCP_HEADER_SIZE:
                    expected_size = struct.unpack_from("<I", buffer, 2)[0]

                    if expected_size < _MIN_RECORD_SIZE:
                        return CaptureResult(
                            success=False,
                            raw_bytes=bytes(buffer),
                            bytes_received=len(buffer),
                            capture_duration_seconds=time.monotonic() - start_time,
                            error_message=(
                                f"Invalid SCP-ECG record size: {expected_size} "
                                f"(minimum {_MIN_RECORD_SIZE})"
                            ),
                        )

                    if expected_size > _MAX_RECORD_SIZE:
                        return CaptureResult(
                            success=False,
                            raw_bytes=bytes(buffer),
                            bytes_received=len(buffer),
                            capture_duration_seconds=time.monotonic() - start_time,
                            error_message=(
                                f"SCP-ECG record size {expected_size} exceeds "
                                f"maximum ({_MAX_RECORD_SIZE})"
                            ),
                        )

                    logger.info(
                        "SCP-ECG header received: expecting %d bytes total",
                        expected_size,
                    )

                # Check if we have the complete frame
                if expected_size is not None and len(buffer) >= expected_size:
                    frame = bytes(buffer[:expected_size])
                    capture_duration = time.monotonic() - start_time

                    # Validate CRC
                    crc_valid = self._validate_crc(frame)

                    if not crc_valid:
                        return CaptureResult(
                            success=False,
                            raw_bytes=frame,
                            bytes_received=len(frame),
                            capture_duration_seconds=capture_duration,
                            error_message="CRC validation failed on received SCP-ECG frame",
                            crc_valid=False,
                        )

                    # Parse into ECGRecord
                    try:
                        ecg_record = self._parse_scp_bytes(frame)
                    except Exception as exc:
                        return CaptureResult(
                            success=False,
                            raw_bytes=frame,
                            bytes_received=len(frame),
                            capture_duration_seconds=capture_duration,
                            crc_valid=True,
                            error_message=f"Failed to parse SCP-ECG data: {exc}",
                        )

                    return CaptureResult(
                        success=True,
                        ecg_record=ecg_record,
                        raw_bytes=frame,
                        bytes_received=len(frame),
                        capture_duration_seconds=capture_duration,
                        crc_valid=True,
                    )

        except Exception as exc:
            return CaptureResult(
                success=False,
                raw_bytes=bytes(buffer),
                bytes_received=len(buffer),
                capture_duration_seconds=time.monotonic() - start_time,
                error_message=f"Serial capture error: {exc}",
            )
        finally:
            ser.close()

    # ---------------------------------------------------------------
    # CRC validation
    # ---------------------------------------------------------------

    @staticmethod
    def _validate_crc(frame: bytes) -> bool:
        """Validate CRC-16/CCITT of an SCP-ECG frame.

        The first 2 bytes are the stored CRC; the CRC is computed
        over bytes 2..end (i.e. everything after the CRC field).

        Args:
            frame: Complete SCP-ECG frame bytes.

        Returns:
            ``True`` if CRC matches or if the stored CRC is zero
            (some devices omit CRC).
        """
        if len(frame) < _SCP_HEADER_SIZE:
            return False

        stored_crc = struct.unpack_from("<H", frame, 0)[0]

        # Some legacy carts set CRC to 0x0000 — treat as valid
        if stored_crc == 0x0000:
            logger.debug("SCP-ECG frame has zero CRC — accepting without validation")
            return True

        computed_crc = _crc16_ccitt(frame[2:])
        if stored_crc != computed_crc:
            logger.warning(
                "CRC mismatch: stored=0x%04X, computed=0x%04X",
                stored_crc,
                computed_crc,
            )
            return False

        return True

    # ---------------------------------------------------------------
    # SCP-ECG parsing
    # ---------------------------------------------------------------

    @staticmethod
    def _parse_scp_bytes(frame: bytes) -> ECGRecord:
        """Parse raw SCP-ECG bytes into an ECGRecord.

        Writes the frame to a temporary in-memory buffer and delegates
        to the existing ``read_scp()`` reader which expects a file path.
        We use a temporary file to bridge the interface.

        Args:
            frame: Complete SCP-ECG frame bytes.

        Returns:
            Parsed :class:`ECGRecord`.

        Raises:
            ValueError: If the frame cannot be parsed.
        """
        import tempfile

        from aortica.io.scp_reader import read_scp

        # Write to a temporary file so read_scp() can process it
        with tempfile.NamedTemporaryFile(suffix=".scp", delete=False) as tmp:
            tmp.write(frame)
            tmp_path = tmp.name

        try:
            record = read_scp(tmp_path)
            # Tag the source format to indicate serial capture origin
            record.source_format = "scp-ecg-serial"
            return record
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
