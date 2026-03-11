"""Signal processing: QRS detection, denoising, quality assessment."""

from aortica.signal.denoising import denoise
from aortica.signal.qrs_detection import detect_qrs
from aortica.signal.quality_scoring import LeadQuality, QualityReport, score_quality

__all__ = ["denoise", "detect_qrs", "score_quality", "QualityReport", "LeadQuality"]
