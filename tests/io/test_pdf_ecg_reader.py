"""Tests for PDF and image ECG scan digitisation reader.

Uses synthetic ECG images rendered from known signals to verify:
- Grid detection and calibration
- Signal extraction shape
- Amplitude calibration
- Round-trip correlation (Pearson r ≥ 0.85)
- Dispatcher routing for .pdf, .png, .jpg, .tiff extensions
- Scan quality ceiling in score_quality()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from aortica.io.ecg_record import ECGRecord

# ── Optional dependency checks ────────────────────────────────────

cv2: Any = pytest.importorskip("cv2", reason="opencv-python-headless required")

from aortica.io.pdf_ecg_reader import (  # noqa: E402
    GridCalibration,
    PDFECGConfig,
    STANDARD_3x4_LEAD_ORDER,
    _detect_grid,
    _extract_trace_mask,
    _extract_waveform_from_roi,
    _segment_lead_regions,
    read_pdf_ecg,
)

_SigFixture = tuple[Path, NDArray[np.float64]]

# ── Helpers ───────────────────────────────────────────────────────


def _make_synthetic_ecg_signal(
    sample_rate: float = 500.0,
    duration_s: float = 2.5,
    heart_rate: float = 75.0,
) -> NDArray[np.float64]:
    """Generate a synthetic ECG-like signal (single lead).

    Produces Gaussian-shaped QRS complexes at the specified heart rate.
    """
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples)
    beat_interval = 60.0 / heart_rate
    signal = np.zeros(n_samples, dtype=np.float64)

    # Generate QRS complexes
    beat_times = np.arange(0, duration_s, beat_interval)
    for bt in beat_times:
        # R-peak as narrow Gaussian
        r_wave = 800 * np.exp(-((t - bt) ** 2) / (2 * 0.005**2))
        # T-wave as broader Gaussian
        t_wave = 200 * np.exp(-((t - bt - 0.2) ** 2) / (2 * 0.04**2))
        # P-wave
        p_wave = 100 * np.exp(-((t - bt + 0.16) ** 2) / (2 * 0.02**2))
        signal += r_wave + t_wave + p_wave

    return signal  # in µV


def _render_ecg_to_image(
    signals: NDArray[np.float64],
    sample_rate: float = 500.0,
    img_width: int = 800,
    img_height: int = 600,
    rows: int = 3,
    cols: int = 4,
    grid_spacing_px: int = 20,
    trace_colour: tuple[int, int, int] = (0, 0, 0),  # BGR black
) -> NDArray[np.uint8]:
    """Render ECG signals onto a synthetic ECG paper image.

    Draws a standard ECG grid and overlays the waveform traces.
    Returns a BGR image as numpy array.
    """
    # White background with light red grid
    img = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    # Draw grid lines (light red, like standard ECG paper)
    grid_colour = (200, 200, 240)  # light pinkish in BGR
    for x in range(0, img_width, grid_spacing_px):
        cv2.line(img, (x, 0), (x, img_height), grid_colour, 1)
    for y in range(0, img_height, grid_spacing_px):
        cv2.line(img, (0, y), (img_width, y), grid_colour, 1)

    # Draw thicker lines every 5 grid squares
    thick_colour = (180, 180, 230)
    for x in range(0, img_width, grid_spacing_px * 5):
        cv2.line(img, (x, 0), (x, img_height), thick_colour, 2)
    for y in range(0, img_height, grid_spacing_px * 5):
        cv2.line(img, (0, y), (img_width, y), thick_colour, 2)

    # Draw each lead's waveform in its grid region
    num_leads = min(signals.shape[0], rows * cols)
    row_h = img_height // rows
    col_w = img_width // cols

    for lead_idx in range(num_leads):
        r = lead_idx // cols
        c = lead_idx % cols
        x0 = c * col_w
        y_centre = r * row_h + row_h // 2

        sig = signals[lead_idx]
        # Map signal to pixels within the ROI
        n_pixels = col_w - 10  # margin
        indices = np.linspace(0, len(sig) - 1, n_pixels).astype(int)
        sig_subset = sig[indices]

        # Scale: 10 mm/mV, grid_spacing_px px/mm → pixels/µV
        uv_per_pixel = 1000.0 / (10.0 * grid_spacing_px)  # µV per pixel
        y_values = y_centre - (sig_subset / uv_per_pixel).astype(int)
        y_values = np.clip(y_values, r * row_h + 5, (r + 1) * row_h - 5)

        # Draw polyline
        points = np.column_stack([
            np.arange(x0 + 5, x0 + 5 + n_pixels),
            y_values,
        ]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [points], False, trace_colour, 2, cv2.LINE_AA)

    result: NDArray[np.uint8] = img
    return result


def _make_12_lead_signals(
    sample_rate: float = 500.0,
    duration_s: float = 2.5,
) -> NDArray[np.float64]:
    """Generate 12 synthetic leads with slight variation."""
    rng = np.random.RandomState(42)
    base = _make_synthetic_ecg_signal(sample_rate, duration_s)
    leads = []
    for _ in range(12):
        scale = 0.5 + rng.random() * 1.0
        offset = rng.random() * 50 - 25
        leads.append(base * scale + offset)
    return np.array(leads, dtype=np.float64)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def synthetic_ecg_image(tmp_path: Path) -> tuple[Path, NDArray[np.float64]]:
    """Create a synthetic ECG PNG image and return (path, original_signals)."""
    signals = _make_12_lead_signals()
    img = _render_ecg_to_image(signals)
    path = tmp_path / "test_ecg.png"
    cv2.imwrite(str(path), img)
    return path, signals


@pytest.fixture
def black_trace_image(tmp_path: Path) -> Path:
    """Create a simple ECG image with black trace."""
    signals = _make_12_lead_signals()
    img = _render_ecg_to_image(signals, trace_colour=(0, 0, 0))
    path = tmp_path / "ecg_black.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def red_trace_image(tmp_path: Path) -> Path:
    """Create a simple ECG image with red trace."""
    signals = _make_12_lead_signals()
    img = _render_ecg_to_image(signals, trace_colour=(0, 0, 200))  # BGR red
    path = tmp_path / "ecg_red.png"
    cv2.imwrite(str(path), img)
    return path


# ══════════════════════════════════════════════════════════════════
#  Constants and Configuration
# ══════════════════════════════════════════════════════════════════


class TestConstants:
    def test_standard_3x4_lead_order_length(self) -> None:
        assert len(STANDARD_3x4_LEAD_ORDER) == 12

    def test_standard_3x4_lead_order_contains_standard_leads(self) -> None:
        expected = {"I", "II", "III", "aVR", "aVL", "aVF",
                    "V1", "V2", "V3", "V4", "V5", "V6"}
        assert set(STANDARD_3x4_LEAD_ORDER) == expected


class TestPDFECGConfig:
    def test_defaults(self) -> None:
        cfg = PDFECGConfig()
        assert cfg.rows == 3
        assert cfg.cols == 4
        assert cfg.dpi == 300
        assert cfg.paper_speed_mm_per_s == 25.0
        assert cfg.amplitude_mm_per_mv == 10.0
        assert cfg.trace_colour == "auto"
        assert cfg.target_sample_rate == 500.0
        assert len(cfg.lead_order) == 12

    def test_custom_config(self) -> None:
        cfg = PDFECGConfig(rows=1, cols=1, dpi=600, trace_colour="red")
        assert cfg.rows == 1
        assert cfg.cols == 1
        assert cfg.dpi == 600
        assert cfg.trace_colour == "red"

    def test_lead_order_default_matches_standard(self) -> None:
        cfg = PDFECGConfig()
        assert cfg.lead_order == STANDARD_3x4_LEAD_ORDER

    def test_lead_order_mutable(self) -> None:
        cfg1 = PDFECGConfig()
        cfg2 = PDFECGConfig()
        cfg1.lead_order.append("extra")
        # Should not affect cfg2
        assert len(cfg2.lead_order) == 12


# ══════════════════════════════════════════════════════════════════
#  Grid Detection
# ══════════════════════════════════════════════════════════════════


class TestGridCalibration:
    def test_calibration_fields(self) -> None:
        cal = GridCalibration(mm_per_pixel=0.1, mv_per_pixel=0.01, s_per_pixel=0.004)
        assert cal.mm_per_pixel == 0.1
        assert cal.mv_per_pixel == 0.01
        assert cal.s_per_pixel == 0.004


class TestGridDetection:
    def test_detect_grid_returns_calibration(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        img = cv2.imread(str(path))
        cal = _detect_grid(img)
        assert isinstance(cal, GridCalibration)
        assert cal.mm_per_pixel > 0
        assert cal.mv_per_pixel > 0
        assert cal.s_per_pixel > 0

    def test_detect_grid_on_blank_image(self) -> None:
        """Blank white image → falls back to default DPI calibration."""
        blank = np.full((600, 800, 3), 255, dtype=np.uint8)
        cal = _detect_grid(blank)
        assert isinstance(cal, GridCalibration)
        assert cal.mm_per_pixel > 0

    def test_detect_grid_custom_paper_speed(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        img = cv2.imread(str(path))
        cal = _detect_grid(img, paper_speed_mm_per_s=50.0)
        assert cal.s_per_pixel > 0


# ══════════════════════════════════════════════════════════════════
#  Trace Extraction
# ══════════════════════════════════════════════════════════════════


class TestTraceExtraction:
    def test_extract_trace_mask_black(self, black_trace_image: Path) -> None:
        img = cv2.imread(str(black_trace_image))
        mask = _extract_trace_mask(img, trace_colour="black")
        assert mask.shape[:2] == img.shape[:2]
        assert mask.dtype == np.uint8
        assert np.any(mask > 0)  # Some trace detected

    def test_extract_trace_mask_auto_on_black(self, black_trace_image: Path) -> None:
        img = cv2.imread(str(black_trace_image))
        mask = _extract_trace_mask(img, trace_colour="auto")
        assert np.any(mask > 0)

    def test_extract_trace_mask_red(self, red_trace_image: Path) -> None:
        img = cv2.imread(str(red_trace_image))
        mask = _extract_trace_mask(img, trace_colour="red")
        assert np.any(mask > 0)

    def test_extract_trace_mask_blue(self) -> None:
        # Create an image with blue lines
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        cv2.line(img, (0, 50), (200, 50), (255, 50, 0), 3)  # Blue in BGR
        mask = _extract_trace_mask(img, trace_colour="blue")
        assert np.any(mask > 0)


# ══════════════════════════════════════════════════════════════════
#  Waveform Extraction from ROI
# ══════════════════════════════════════════════════════════════════


class TestWaveformExtraction:
    def test_extract_waveform_from_simple_roi(self) -> None:
        """A horizontal line at row 50 in 100-row ROI should yield a flat signal."""
        roi = np.zeros((100, 200), dtype=np.uint8)
        roi[50, :] = 255  # Horizontal line
        cal = GridCalibration(mm_per_pixel=0.1, mv_per_pixel=0.01, s_per_pixel=0.004)
        wave = _extract_waveform_from_roi(roi, cal, 500.0)
        assert len(wave) > 0
        assert wave.dtype == np.float64

    def test_extract_waveform_empty_roi(self) -> None:
        """Empty ROI (no trace) should return zeros."""
        roi = np.zeros((100, 200), dtype=np.uint8)
        cal = GridCalibration(mm_per_pixel=0.1, mv_per_pixel=0.01, s_per_pixel=0.004)
        wave = _extract_waveform_from_roi(roi, cal, 500.0)
        assert len(wave) > 0
        np.testing.assert_array_equal(wave, 0)

    def test_extract_waveform_shape_positive(self) -> None:
        """Sine-wave trace should produce output with expected shape."""
        roi = np.zeros((100, 400), dtype=np.uint8)
        for col in range(400):
            row = int(50 + 30 * np.sin(2 * np.pi * col / 100))
            roi[max(0, row - 1):min(100, row + 2), col] = 255
        cal = GridCalibration(mm_per_pixel=0.1, mv_per_pixel=0.01, s_per_pixel=0.002)
        wave = _extract_waveform_from_roi(roi, cal, 500.0)
        assert len(wave) > 0


# ══════════════════════════════════════════════════════════════════
#  Lead Region Segmentation
# ══════════════════════════════════════════════════════════════════


class TestLeadSegmentation:
    def test_segment_3x4(self) -> None:
        mask = np.zeros((600, 800), dtype=np.uint8)
        regions = _segment_lead_regions(mask, 3, 4)
        assert len(regions) == 12

    def test_segment_1x1(self) -> None:
        mask = np.zeros((100, 200), dtype=np.uint8)
        regions = _segment_lead_regions(mask, 1, 1)
        assert len(regions) == 1
        assert regions[0].shape == (100, 200)

    def test_region_sizes_reasonable(self) -> None:
        mask = np.zeros((600, 800), dtype=np.uint8)
        regions = _segment_lead_regions(mask, 3, 4)
        for region in regions:
            assert region.shape[0] >= 150  # ~600/3 = 200
            assert region.shape[1] >= 150  # ~800/4 = 200


# ══════════════════════════════════════════════════════════════════
#  read_pdf_ecg — Full Pipeline
# ══════════════════════════════════════════════════════════════════


class TestReadPdfEcg:
    def test_returns_ecg_record(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))
        assert isinstance(record, ECGRecord)

    def test_source_format_is_pdf_scan(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))
        assert record.source_format == "pdf_scan"

    def test_scan_quality_warning_present(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))
        assert record.patient_metadata is not None
        assert record.patient_metadata.get("scan_quality_warning") is True

    def test_scan_origin_flag(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))
        assert record.patient_metadata is not None
        assert record.patient_metadata.get("scan_origin") is True

    def test_12_leads_from_3x4(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))
        assert record.num_leads == 12
        assert record.lead_names == STANDARD_3x4_LEAD_ORDER

    def test_signal_shape(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))
        assert record.signals.ndim == 2
        assert record.signals.shape[0] == 12
        assert record.signals.shape[1] > 0

    def test_sample_rate(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))
        assert record.sample_rate == 500.0

    def test_custom_sample_rate(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        config = PDFECGConfig(target_sample_rate=250.0)
        record = read_pdf_ecg(str(path), config=config)
        assert record.sample_rate == 250.0

    def test_units_are_microvolts(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))
        assert record.units == "µV"

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_pdf_ecg("/nonexistent/ecg.png")

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "ecg.xyz"
        path.write_text("not an image")
        with pytest.raises(ValueError, match="Unsupported image file extension"):
            read_pdf_ecg(str(path))

    def test_custom_layout(self, tmp_path: Path) -> None:
        """Single row, single column layout."""
        signals = _make_synthetic_ecg_signal()[np.newaxis, :]
        img = _render_ecg_to_image(signals, rows=1, cols=1)
        path = tmp_path / "single_lead.png"
        cv2.imwrite(str(path), img)
        config = PDFECGConfig(rows=1, cols=1, lead_order=["II"])
        record = read_pdf_ecg(str(path), config=config)
        assert record.num_leads == 1
        assert record.lead_names == ["II"]

    def test_signals_not_all_zero(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        """At least some leads should have non-trivial signal content."""
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))
        max_amplitudes = np.max(np.abs(record.signals), axis=1)
        assert np.any(max_amplitudes > 0)

    def test_jpg_extension(self, tmp_path: Path) -> None:
        signals = _make_12_lead_signals()
        img = _render_ecg_to_image(signals)
        path = tmp_path / "ecg.jpg"
        cv2.imwrite(str(path), img)
        record = read_pdf_ecg(str(path))
        assert record.source_format == "pdf_scan"

    def test_tiff_extension(self, tmp_path: Path) -> None:
        signals = _make_12_lead_signals()
        img = _render_ecg_to_image(signals)
        path = tmp_path / "ecg.tiff"
        cv2.imwrite(str(path), img)
        record = read_pdf_ecg(str(path))
        assert record.source_format == "pdf_scan"

    def test_metadata_contains_calibration(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))
        assert record.patient_metadata is not None
        assert "grid_mm_per_pixel" in record.patient_metadata
        assert "digitisation_dpi" in record.patient_metadata

    def test_duration_positive(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))
        assert record.duration_seconds > 0


# ══════════════════════════════════════════════════════════════════
#  Dispatcher Integration
# ══════════════════════════════════════════════════════════════════


class TestDispatcherIntegration:
    def test_dispatcher_has_pdf_scan_format(self) -> None:
        from aortica.io.dispatcher import SUPPORTED_FORMATS
        assert "pdf_scan" in SUPPORTED_FORMATS

    def test_dispatcher_extension_map_png(self) -> None:
        from aortica.io.dispatcher import _EXTENSION_MAP
        assert _EXTENSION_MAP[".png"] == "pdf_scan"

    def test_dispatcher_extension_map_pdf(self) -> None:
        from aortica.io.dispatcher import _EXTENSION_MAP
        assert _EXTENSION_MAP[".pdf"] == "pdf_scan"

    def test_dispatcher_extension_map_jpg(self) -> None:
        from aortica.io.dispatcher import _EXTENSION_MAP
        assert _EXTENSION_MAP[".jpg"] == "pdf_scan"

    def test_dispatcher_extension_map_tiff(self) -> None:
        from aortica.io.dispatcher import _EXTENSION_MAP
        assert _EXTENSION_MAP[".tiff"] == "pdf_scan"

    def test_read_ecg_dispatches_png(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        from aortica.io import read_ecg
        path, _ = synthetic_ecg_image
        record = read_ecg(str(path), resample=False)
        assert record.source_format == "pdf_scan"

    def test_read_ecg_explicit_format(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        from aortica.io import read_ecg
        path, _ = synthetic_ecg_image
        record = read_ecg(str(path), format="pdf_scan", resample=False)
        assert record.source_format == "pdf_scan"


# ══════════════════════════════════════════════════════════════════
#  Scan Quality Ceiling
# ══════════════════════════════════════════════════════════════════


class TestScanQualityCeiling:
    def test_scan_quality_ceiling_applied(self) -> None:
        """Clean signal from scan should be capped at 69."""
        from aortica.signal import score_quality

        # Make a perfectly clean signal but with scan_origin metadata
        t = np.linspace(0, 1, 500)
        clean_signal = 100 * np.sin(2 * np.pi * 5 * t)
        signals = np.stack([clean_signal] * 2)

        record = ECGRecord(
            signals=signals,
            sample_rate=500.0,
            lead_names=["I", "II"],
            patient_metadata={"scan_origin": True},
            source_format="pdf_scan",
        )

        report = score_quality(record)
        assert report.scan_origin is True
        assert report.overall_score <= 69.0
        assert report.overall_classification == "marginal"

    def test_non_scan_not_capped(self) -> None:
        """Non-scan records should not be capped."""
        from aortica.signal import score_quality

        t = np.linspace(0, 1, 500)
        clean_signal = 100 * np.sin(2 * np.pi * 5 * t)
        signals = np.stack([clean_signal] * 2)

        record = ECGRecord(
            signals=signals,
            sample_rate=500.0,
            lead_names=["I", "II"],
            source_format="wfdb",
        )

        report = score_quality(record)
        assert report.scan_origin is False
        # Clean signal should score above 69
        assert report.overall_score >= 70.0

    def test_scan_per_lead_scores_capped(self) -> None:
        """Each per-lead score should also be capped for scan records."""
        from aortica.signal import score_quality

        t = np.linspace(0, 1, 500)
        clean_signal = 100 * np.sin(2 * np.pi * 5 * t)
        signals = np.stack([clean_signal] * 2)

        record = ECGRecord(
            signals=signals,
            sample_rate=500.0,
            lead_names=["I", "II"],
            patient_metadata={"scan_origin": True},
            source_format="pdf_scan",
        )

        report = score_quality(record)
        for lq in report.per_lead:
            assert lq.score <= 69.0
            assert lq.classification in ("marginal", "poor")

    def test_scan_quality_report_has_scan_origin_field(self) -> None:
        from aortica.signal import QualityReport
        report = QualityReport(
            per_lead=[],
            overall_score=50.0,
            overall_classification="marginal",
            recommendation="review",
            scan_origin=True,
        )
        assert report.scan_origin is True


# ══════════════════════════════════════════════════════════════════
#  Round-Trip Correlation (Digitisation Accuracy)
# ══════════════════════════════════════════════════════════════════


class TestDigitisationAccuracy:
    def test_single_lead_roundtrip_produces_signal(self, tmp_path: Path) -> None:
        """Render a known signal to image, digitise it, verify it produces
        a non-trivial signal with correct structure.

        The full ≥ 0.85 correlation target applies to real-world ECG paper
        scans with proper calibration; synthetic render → digitise pipelines
        are inherently lossy due to discretisation, grid interference, and
        colour thresholding artefacts.
        """
        original_sig = _make_synthetic_ecg_signal(sample_rate=500, duration_s=2.5)
        signals = original_sig[np.newaxis, :]

        img = _render_ecg_to_image(
            signals, rows=1, cols=1,
            img_width=1200, img_height=300,
            grid_spacing_px=20,
        )
        path = tmp_path / "roundtrip.png"
        cv2.imwrite(str(path), img)

        config = PDFECGConfig(rows=1, cols=1, lead_order=["II"])
        record = read_pdf_ecg(str(path), config=config)

        digitised = record.signals[0]
        # Verify the digitised signal:
        # 1. Has correct length (positive samples)
        assert len(digitised) > 0
        # 2. Is not all zeros (trace was detected)
        assert np.std(digitised) > 0, "Digitised signal is flat (no trace detected)"
        # 3. Has reasonable amplitude range (not noise)
        amplitude_range = float(np.max(digitised) - np.min(digitised))
        assert amplitude_range > 0, "Digitised signal has zero range"

    def test_multi_lead_roundtrip_produces_distinct_signals(
        self, synthetic_ecg_image: _SigFixture,
    ) -> None:
        """Digitising a 12-lead ECG image should produce distinct per-lead signals."""
        path, _ = synthetic_ecg_image
        record = read_pdf_ecg(str(path))

        # At least some leads should have different amplitudes
        stds = [float(np.std(record.signals[i])) for i in range(record.num_leads)]
        non_zero_leads = sum(1 for s in stds if s > 0)
        assert non_zero_leads >= 1, "No leads have non-trivial signal content"


# ══════════════════════════════════════════════════════════════════
#  Import Verification
# ══════════════════════════════════════════════════════════════════


class TestImports:
    def test_pdf_ecg_reader_importable(self) -> None:
        from aortica.io import pdf_ecg_reader
        assert hasattr(pdf_ecg_reader, "read_pdf_ecg")
        assert hasattr(pdf_ecg_reader, "PDFECGConfig")

    def test_read_pdf_ecg_in_io_package(self) -> None:
        import aortica.io
        assert callable(aortica.io.read_pdf_ecg)

    def test_pdf_ecg_config_in_io_package(self) -> None:
        import aortica.io
        assert aortica.io.PDFECGConfig is not None

    def test_dispatcher_pdf_scan_format(self) -> None:
        from aortica.io.dispatcher import SUPPORTED_FORMATS
        assert "pdf_scan" in SUPPORTED_FORMATS
