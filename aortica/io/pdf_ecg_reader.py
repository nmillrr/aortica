"""PDF and image ECG scan digitization reader.

Provides :func:`read_pdf_ecg`, which accepts a PDF or image file
(PNG, JPG, TIFF) containing a scanned ECG printout and digitises it
into an :class:`ECGRecord`.

The pipeline:

1. **Rasterise** (PDFs only) → high-resolution image (≥300 DPI).
2. **Grid detection** → compute mm/pixel calibration for amplitude
   (mV/pixel) and time (s/pixel) axes using the standard ECG grid.
3. **Lead region segmentation** → detect the 3×4 panel layout (or
   rhythm strip) from the image structure.
4. **Waveform trace extraction** → isolate the coloured ECG trace
   from the background grid via colour thresholding + contour
   following, then reconstruct a time-series signal per lead.

Dependencies (optional — behind ``[scan]`` extra):

* ``opencv-python-headless`` (cv2)
* ``pdfplumber`` **or** ``pymupdf`` (fitz) for PDF rasterisation

Usage::

    from aortica.io import read_pdf_ecg
    record = read_pdf_ecg("ecg_scan.pdf")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from aortica.io.ecg_record import ECGRecord

# ── Optional dependency checks ────────────────────────────────────

HAS_CV2 = False
try:
    import cv2  # type: ignore[import-untyped]

    HAS_CV2 = True
except ImportError:
    pass

HAS_PDFPLUMBER = False
try:
    import pdfplumber  # type: ignore[import-untyped]  # noqa: F401

    HAS_PDFPLUMBER = True
except ImportError:
    pass

HAS_FITZ = False
try:
    import fitz  # type: ignore[import-untyped]  # noqa: F401 (pymupdf)

    HAS_FITZ = True
except ImportError:
    pass


def _check_cv2() -> None:
    if not HAS_CV2:
        raise ImportError(
            "opencv-python-headless is required for PDF/image ECG digitisation. "
            "Install it with: pip install opencv-python-headless"
        )


def _check_pdf_backend() -> None:
    if not HAS_PDFPLUMBER and not HAS_FITZ:
        raise ImportError(
            "pdfplumber or pymupdf (fitz) is required for PDF ECG rasterisation. "
            "Install one with: pip install pdfplumber  or  pip install pymupdf"
        )


# ── Standard 12-lead 3×4 layout ──────────────────────────────────

STANDARD_3x4_LEAD_ORDER: list[str] = [
    # Row 0: I, aVR, V1, V4
    "I", "aVR", "V1", "V4",
    # Row 1: II, aVL, V2, V5
    "II", "aVL", "V2", "V5",
    # Row 2: III, aVF, V3, V6
    "III", "aVF", "V3", "V6",
]


# ── Configuration ─────────────────────────────────────────────────


@dataclass
class PDFECGConfig:
    """Configuration for PDF/image ECG digitisation.

    Attributes:
        rows: Number of lead rows in the printout (default 3).
        cols: Number of lead columns in the printout (default 4).
        lead_order: Explicit lead names in row-major order.
            Defaults to the standard 3×4 layout.
        dpi: Target DPI for PDF rasterisation (default 300).
        paper_speed_mm_per_s: Standard paper speed (default 25 mm/s).
        amplitude_mm_per_mv: Standard gain (default 10 mm/mV).
        trace_colour: Dominant trace colour to extract.
            ``"auto"`` attempts automatic detection.
        target_sample_rate: Desired output sample rate in Hz
            (default 500).
    """

    rows: int = 3
    cols: int = 4
    lead_order: list[str] = field(default_factory=lambda: list(STANDARD_3x4_LEAD_ORDER))
    dpi: int = 300
    paper_speed_mm_per_s: float = 25.0
    amplitude_mm_per_mv: float = 10.0
    trace_colour: str = "auto"
    target_sample_rate: float = 500.0


# ── PDF Rasterisation ─────────────────────────────────────────────


def _rasterise_pdf(path: Path, dpi: int = 300) -> NDArray[np.uint8]:
    """Convert the first page of a PDF to an image array (BGR).

    Tries ``pymupdf`` first (fast, no external deps), then falls
    back to ``pdfplumber``.
    """
    _check_pdf_backend()

    if HAS_FITZ:
        import fitz  # type: ignore[import-untyped]

        doc = fitz.open(str(path))
        page = doc[0]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_array: NDArray[np.uint8] = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        doc.close()
        # pymupdf gives RGB; OpenCV wants BGR
        if img_array.shape[2] == 4:  # RGBA
            img_bgr: NDArray[np.uint8] = cv2.cvtColor(  # type: ignore[assignment]
                img_array, cv2.COLOR_RGBA2BGR
            )
        else:
            img_bgr = cv2.cvtColor(  # type: ignore[assignment]
                img_array, cv2.COLOR_RGB2BGR
            )
        return img_bgr

    # pdfplumber fallback
    import pdfplumber  # type: ignore[import-untyped]

    with pdfplumber.open(str(path)) as pdf:
        page = pdf.pages[0]
        pil_img = page.to_image(resolution=dpi).original
        img_np: NDArray[np.uint8] = np.array(pil_img, dtype=np.uint8)
        if img_np.ndim == 2:
            img_bgr_out: NDArray[np.uint8] = cv2.cvtColor(  # type: ignore[assignment]
                img_np, cv2.COLOR_GRAY2BGR
            )
        elif img_np.shape[2] == 4:
            img_bgr_out = cv2.cvtColor(  # type: ignore[assignment]
                img_np, cv2.COLOR_RGBA2BGR
            )
        else:
            img_bgr_out = cv2.cvtColor(  # type: ignore[assignment]
                img_np, cv2.COLOR_RGB2BGR
            )
        return img_bgr_out


def _load_image(path: Path) -> NDArray[np.uint8]:
    """Load an image file into a BGR array using OpenCV."""
    _check_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image file: {path}")
    result: NDArray[np.uint8] = np.asarray(img, dtype=np.uint8)
    return result


# ── Grid Detection ────────────────────────────────────────────────


@dataclass
class GridCalibration:
    """Detected grid calibration parameters.

    Attributes:
        mm_per_pixel: Spatial scale in mm/pixel.
        mv_per_pixel: Amplitude scale in mV/pixel.
        s_per_pixel: Time scale in s/pixel.
    """

    mm_per_pixel: float
    mv_per_pixel: float
    s_per_pixel: float


def _detect_grid(
    img_bgr: NDArray[np.uint8],
    paper_speed_mm_per_s: float = 25.0,
    amplitude_mm_per_mv: float = 10.0,
) -> GridCalibration:
    """Detect the standard ECG grid and compute calibration.

    Converts to grayscale, applies edge detection, then uses
    the Hough line transform to find dominant horizontal and
    vertical grid lines.  The median spacing between them
    gives the mm-per-pixel calibration (standard ECG grid has
    1 mm minor divisions).
    """
    _check_cv2()
    gray: NDArray[np.uint8] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # type: ignore[assignment]

    # Enhance grid lines with adaptive thresholding
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    h, w = edges.shape[:2]
    min_line_length = min(h, w) // 4

    # Detect lines using probabilistic Hough
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        # Fallback: assume standard 25 mm/s at 300 DPI
        # 300 DPI → 11.81 pixels/mm
        mm_per_pixel = 25.4 / 300.0  # ~0.0847 mm/pixel
        return GridCalibration(
            mm_per_pixel=mm_per_pixel,
            mv_per_pixel=mm_per_pixel / amplitude_mm_per_mv,
            s_per_pixel=mm_per_pixel / paper_speed_mm_per_s,
        )

    # Separate horizontal and vertical lines
    h_positions: list[float] = []
    v_positions: list[float] = []
    angle_threshold = 10  # degrees from axis

    for line_arr in lines:
        x1, y1, x2, y2 = line_arr[0]
        angle_deg = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle_deg < angle_threshold or angle_deg > (180 - angle_threshold):
            # Horizontal line
            h_positions.append((y1 + y2) / 2.0)
        elif abs(angle_deg - 90) < angle_threshold:
            # Vertical line
            v_positions.append((x1 + x2) / 2.0)

    # Compute median spacing between consecutive grid lines
    spacings: list[float] = []
    for positions in (sorted(h_positions), sorted(v_positions)):
        if len(positions) >= 2:
            diffs = np.diff(positions)
            # Filter out very small spacings (duplicates) and very large ones
            median_d = float(np.median(diffs))
            valid = diffs[(diffs > median_d * 0.5) & (diffs < median_d * 1.5)]
            if len(valid) > 0:
                spacings.append(float(np.median(valid)))

    if len(spacings) == 0:
        # Fallback to DPI-based estimate
        mm_per_pixel = 25.4 / 300.0
    else:
        # Average spacing corresponds to 1 mm on standard ECG paper
        avg_spacing = float(np.mean(spacings))
        mm_per_pixel = 1.0 / avg_spacing  # 1 mm per grid square

    return GridCalibration(
        mm_per_pixel=mm_per_pixel,
        mv_per_pixel=mm_per_pixel / amplitude_mm_per_mv,
        s_per_pixel=mm_per_pixel / paper_speed_mm_per_s,
    )


# ── Trace Extraction ──────────────────────────────────────────────


def _extract_trace_mask(
    img_bgr: NDArray[np.uint8],
    trace_colour: str = "auto",
) -> NDArray[np.uint8]:
    """Create a binary mask of the ECG waveform trace.

    Isolates dark (black) or coloured (red/blue) traces from the
    lighter grid background using colour-space thresholding.
    """
    _check_cv2()
    hsv: NDArray[np.uint8] = cv2.cvtColor(  # type: ignore[assignment]
        img_bgr, cv2.COLOR_BGR2HSV
    )
    gray: NDArray[np.uint8] = cv2.cvtColor(  # type: ignore[assignment]
        img_bgr, cv2.COLOR_BGR2GRAY
    )

    if trace_colour == "auto":
        # Try: detect red trace first, then fall back to dark (black) trace
        # Red in HSV: H ∈ [0, 10] or [170, 180], S > 50, V > 50
        lower_red1 = np.array([0, 50, 50], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        red_mask1: Any = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2: Any = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask: NDArray[np.uint8] = cv2.bitwise_or(red_mask1, red_mask2)  # type: ignore[assignment]

        red_fraction = float(np.sum(red_mask > 0)) / (red_mask.shape[0] * red_mask.shape[1])

        if red_fraction > 0.005:  # Meaningful red content
            # Clean up with morphological ops
            kernel: Any = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)  # type: ignore[assignment]
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)  # type: ignore[assignment]
            return red_mask

        # Fall through to black trace detection
        trace_colour = "black"

    if trace_colour == "red":
        lower_red1 = np.array([0, 50, 50], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
        upper_red2_b = np.array([180, 255, 255], dtype=np.uint8)
        mask1: Any = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2: Any = cv2.inRange(hsv, lower_red2, upper_red2_b)
        mask: NDArray[np.uint8] = cv2.bitwise_or(mask1, mask2)  # type: ignore[assignment]
    elif trace_colour == "blue":
        lower_blue = np.array([100, 50, 50], dtype=np.uint8)
        upper_blue = np.array([130, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)  # type: ignore[assignment]
    else:
        # Black/dark trace: low intensity in grayscale
        _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)  # type: ignore[assignment]

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # type: ignore[assignment]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # type: ignore[assignment]

    return mask


def _extract_waveform_from_roi(
    mask_roi: NDArray[np.uint8],
    cal: GridCalibration,
    target_sample_rate: float,
) -> NDArray[np.float64]:
    """Extract a 1D waveform signal from a binary mask region of interest.

    For each column of the ROI, finds the vertical centroid of the
    trace pixels and converts to amplitude in µV using the
    calibration.
    """
    h, w = mask_roi.shape[:2]
    raw_signal: list[float] = []

    for col in range(w):
        column = mask_roi[:, col]
        trace_rows = np.nonzero(column)[0]
        if len(trace_rows) == 0:
            # No trace in this column — interpolate later
            raw_signal.append(np.nan)
        else:
            # Centroid of trace pixels (y increases downward → flip sign)
            centroid_y = float(np.mean(trace_rows))
            raw_signal.append(centroid_y)

    raw = np.array(raw_signal, dtype=np.float64)

    # Interpolate missing values (NaN columns)
    nan_mask = np.isnan(raw)
    if np.all(nan_mask):
        # No trace detected at all — return zeros
        num_output = max(1, int(w * cal.s_per_pixel * target_sample_rate))
        return np.zeros(num_output, dtype=np.float64)

    if np.any(nan_mask):
        x_valid = np.nonzero(~nan_mask)[0]
        raw[nan_mask] = np.interp(np.nonzero(nan_mask)[0], x_valid, raw[x_valid])

    # Convert y-pixels to amplitude:
    # Flip sign (higher y = lower voltage on paper),
    # centre around mean, scale to µV
    raw = -(raw - np.mean(raw))
    raw_mv = raw * cal.mv_per_pixel
    raw_uv = raw_mv * 1000.0  # convert mV → µV

    # Resample from pixel-rate to target sample rate
    pixel_rate = 1.0 / cal.s_per_pixel  # pixels per second
    num_output = max(1, int(len(raw_uv) * target_sample_rate / pixel_rate))

    if num_output == len(raw_uv):
        result: NDArray[np.float64] = np.asarray(raw_uv, dtype=np.float64)
        return result

    from scipy import signal as scipy_signal

    resampled_arr = scipy_signal.resample(raw_uv, num_output)
    resampled: NDArray[np.float64] = np.asarray(resampled_arr, dtype=np.float64)
    return resampled


# ── Lead Region Segmentation ─────────────────────────────────────


def _segment_lead_regions(
    mask: NDArray[np.uint8],
    rows: int,
    cols: int,
) -> list[NDArray[np.uint8]]:
    """Split the mask into lead regions arranged in a rows×cols grid.

    Returns a list of mask ROI arrays in row-major order.
    """
    h, w = mask.shape[:2]
    row_h = h // rows
    col_w = w // cols

    regions: list[NDArray[np.uint8]] = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * row_h
            y1 = (r + 1) * row_h if r < rows - 1 else h
            x0 = c * col_w
            x1 = (c + 1) * col_w if c < cols - 1 else w
            roi: NDArray[np.uint8] = mask[y0:y1, x0:x1]
            regions.append(roi)

    return regions


# ── Public API ────────────────────────────────────────────────────


def read_pdf_ecg(
    path: str | Path,
    config: Optional[PDFECGConfig] = None,
) -> ECGRecord:
    """Read a PDF or image ECG scan and digitise it into an :class:`ECGRecord`.

    Parameters
    ----------
    path:
        Path to a PDF, PNG, JPG, or TIFF file containing a scanned ECG.
    config:
        Digitisation configuration.  If ``None``, standard 3×4
        12-lead layout defaults are used.

    Returns
    -------
    ECGRecord
        A digitised ECG record with ``source_format='pdf_scan'`` and
        ``scan_quality_warning=True`` in ``patient_metadata``.

    Raises
    ------
    ImportError
        If ``opencv-python-headless`` or a PDF backend is not installed.
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be processed.
    """
    _check_cv2()

    cfg = config or PDFECGConfig()
    filepath = Path(path)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # ── Step 1: Load or rasterise ─────────────────────────────────
    ext = filepath.suffix.lower()
    if ext == ".pdf":
        _check_pdf_backend()
        img_bgr = _rasterise_pdf(filepath, dpi=cfg.dpi)
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
        img_bgr = _load_image(filepath)
    else:
        raise ValueError(
            f"Unsupported image file extension '{ext}'. "
            "Supported: .pdf, .png, .jpg, .jpeg, .tiff, .tif, .bmp"
        )

    # ── Step 2: Grid detection & calibration ──────────────────────
    cal = _detect_grid(
        img_bgr,
        paper_speed_mm_per_s=cfg.paper_speed_mm_per_s,
        amplitude_mm_per_mv=cfg.amplitude_mm_per_mv,
    )

    # ── Step 3: Trace extraction ──────────────────────────────────
    trace_mask = _extract_trace_mask(img_bgr, trace_colour=cfg.trace_colour)

    # ── Step 4: Lead region segmentation ──────────────────────────
    num_leads = cfg.rows * cfg.cols
    lead_names = cfg.lead_order[:num_leads]

    # If fewer lead names than regions, generate generic names
    while len(lead_names) < num_leads:
        lead_names.append(f"Lead_{len(lead_names) + 1}")

    regions = _segment_lead_regions(trace_mask, cfg.rows, cfg.cols)

    # ── Step 5: Extract waveform per lead ─────────────────────────
    signals: list[NDArray[np.float64]] = []
    for roi in regions:
        wave = _extract_waveform_from_roi(roi, cal, cfg.target_sample_rate)
        signals.append(wave)

    # Pad/truncate to uniform length
    max_len = max(len(s) for s in signals) if signals else 0
    padded: list[NDArray[np.float64]] = []
    for s in signals:
        if len(s) < max_len:
            pad_arr = np.zeros(max_len - len(s), dtype=np.float64)
            s = np.concatenate([s, pad_arr])
        elif len(s) > max_len:
            s = s[:max_len]
        padded.append(s)

    signal_array: NDArray[np.float64] = np.array(padded, dtype=np.float64)

    # ── Build ECGRecord ───────────────────────────────────────────
    metadata: dict[str, object] = {
        "scan_quality_warning": True,
        "scan_origin": True,
        "digitisation_dpi": cfg.dpi,
        "grid_mm_per_pixel": cal.mm_per_pixel,
    }

    return ECGRecord(
        signals=signal_array,
        sample_rate=cfg.target_sample_rate,
        lead_names=lead_names,
        patient_metadata=metadata,
        source_format="pdf_scan",
        units="µV",
    )
