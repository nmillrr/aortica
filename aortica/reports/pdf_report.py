"""PDF Clinical Report Generator.

Generates a professional PDF clinical report containing:

* Patient demographics (if available)
* 12-lead ECG waveform plot (SVG via matplotlib)
* Signal quality summary
* Per-task findings with confidence bars
* XAI feature attributions (top-3 per finding)
* Risk scores with clinical labels
* Uncertainty / OOD flags

Uses **WeasyPrint** (HTML-to-PDF) with a clinical report template.  The
ECG waveform is rendered as inline SVG via :mod:`matplotlib`.

Usage::

    from aortica.reports import generate_pdf
    generate_pdf(multi_task_output, ecg_record, xai_report, "report.pdf")
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from aortica.io.ecg_record import ECGRecord
from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
from aortica.models.rhythm_head import RHYTHM_CLASSES
from aortica.models.risk_head import RISK_OUTPUTS
from aortica.models.structural_head import STRUCTURAL_CLASSES


# ---------------------------------------------------------------------------
# Lazy imports for optional dependencies
# ---------------------------------------------------------------------------


def _get_weasyprint() -> Any:
    """Lazily import weasyprint."""
    try:
        import weasyprint

        return weasyprint
    except ImportError:
        raise ImportError(
            "WeasyPrint is required for PDF report generation. "
            "Install with: pip install aortica[reports]"
        ) from None


def _get_matplotlib() -> tuple[Any, Any]:
    """Lazily import matplotlib and return (matplotlib, pyplot)."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")  # non-interactive backend
        return matplotlib, plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for PDF report generation. "
            "Install with: pip install aortica[reports]"
        ) from None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single AI finding for display in the report.

    Attributes:
        name: Condition/class name.
        confidence: Confidence percentage (0–100).
        task: Task head that produced this finding.
    """

    name: str
    confidence: float
    task: str


@dataclass
class XAIFeature:
    """A single XAI feature attribution.

    Attributes:
        feature_name: Named ECG segment (e.g. ``"QRS complex"``).
        lead: Lead name.
        delta_score: Attribution strength.
    """

    feature_name: str
    lead: str
    delta_score: float


@dataclass
class XAIReport:
    """Simplified XAI report for PDF rendering.

    Attributes:
        top_features: Top-N contributing features across all findings.
        task: Task that was explained.
    """

    top_features: list[XAIFeature] = field(default_factory=list)
    task: str = "rhythm"


# ---------------------------------------------------------------------------
# Risk score labels
# ---------------------------------------------------------------------------

RISK_LABELS: dict[str, str] = {
    "mortality_1y": "1-Year Mortality",
    "hf_hosp_12m": "12-Month HF Hospitalisation",
    "af_onset_12m": "12-Month AF Onset",
    "ecg_predicted_ef": "ECG-Predicted EF",
    "conduction_disease_trajectory": "Conduction Disease Trajectory",
    "sudden_cardiac_death_risk": "Sudden Cardiac Death Risk",
}


# ---------------------------------------------------------------------------
# ECG waveform rendering
# ---------------------------------------------------------------------------


def _render_ecg_svg(
    ecg_record: ECGRecord,
    width_inches: float = 10.0,
    height_inches: float = 7.5,
) -> str:
    """Render a 12-lead ECG waveform plot as an inline base64-encoded PNG.

    Uses matplotlib to draw the ECG in a standard clinical 3×4 + rhythm
    strip layout with a background grid.

    Returns a base64-encoded PNG data URI string.
    """
    _, plt = _get_matplotlib()

    # Standard 12-lead order
    standard_leads = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]

    # Map available leads
    lead_map: dict[str, NDArray[np.float64]] = {}
    for i, name in enumerate(ecg_record.lead_names):
        lead_map[name] = ecg_record.signals[i].astype(np.float64)

    # Use available leads in standard order, fall back to whatever is available
    leads_to_plot: list[str] = []
    for lead in standard_leads:
        if lead in lead_map:
            leads_to_plot.append(lead)
    # Add any remaining leads not in standard order
    for lead in ecg_record.lead_names:
        if lead not in leads_to_plot:
            leads_to_plot.append(lead)

    n_leads = len(leads_to_plot)
    if n_leads == 0:
        return ""

    fig, axes = plt.subplots(
        n_leads, 1,
        figsize=(width_inches, height_inches),
        sharex=True,
    )
    if n_leads == 1:
        axes = [axes]

    # Time axis
    n_samples = ecg_record.num_samples
    time = np.arange(n_samples) / ecg_record.sample_rate

    for idx, lead_name in enumerate(leads_to_plot):
        ax = axes[idx]
        sig = lead_map[lead_name]

        # Convert to mV for display if in µV
        if ecg_record.units in ("µV", "uV"):
            sig = (sig / 1000.0).astype(np.float64)
        elif ecg_record.units == "V":
            sig = (sig * 1000.0).astype(np.float64)

        ax.plot(time, sig, color="#1a1a2e", linewidth=0.5)

        # Grid
        ax.grid(True, which="major", color="#e8c4c4", linewidth=0.3, alpha=0.6)
        ax.grid(True, which="minor", color="#f0dada", linewidth=0.15, alpha=0.4)
        ax.minorticks_on()

        # Lead label
        ax.set_ylabel(lead_name, fontsize=7, fontweight="bold", rotation=0, labelpad=25)
        ax.tick_params(axis="both", labelsize=5)

        # Consistent y-axis
        y_range = max(abs(sig.max() - sig.min()), 0.5)
        y_center = (sig.max() + sig.min()) / 2
        ax.set_ylim(y_center - y_range * 0.7, y_center + y_range * 0.7)

    axes[-1].set_xlabel("Time (s)", fontsize=7)
    fig.suptitle("12-Lead ECG", fontsize=9, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Save to buffer as PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


# ---------------------------------------------------------------------------
# Prediction extraction helpers
# ---------------------------------------------------------------------------


def _extract_predictions(
    multi_task_output: Any,
) -> dict[str, list[float]]:
    """Extract per-task prediction lists from various output formats.

    Handles MultiTaskOutput dataclass, dict, and tensor/array types.
    """
    tasks = {
        "rhythm": RHYTHM_CLASSES,
        "structural": STRUCTURAL_CLASSES,
        "ischaemia": ISCHAEMIA_CLASSES,
        "risk": RISK_OUTPUTS,
    }

    result: dict[str, list[float]] = {}

    for task_name, class_names in tasks.items():
        # Get raw values
        if hasattr(multi_task_output, task_name):
            values = getattr(multi_task_output, task_name)
        elif isinstance(multi_task_output, dict):
            values = multi_task_output.get(task_name)
        else:
            values = None

        if values is None:
            continue

        # Convert tensor/array to list
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        if hasattr(values, "tolist"):
            values = values.tolist()  # type: ignore[union-attr]
        if isinstance(values, np.ndarray):
            values = values.tolist()

        # Handle batch dimension
        if isinstance(values, list) and len(values) > 0:
            if isinstance(values[0], list):
                values = values[0]  # Take first sample from batch

        # Ensure length matches class count
        n = min(len(values), len(class_names))
        result[task_name] = [float(v) for v in values[:n]]

    return result


def _get_findings(
    predictions: dict[str, list[float]],
    threshold: float = 0.5,
) -> list[Finding]:
    """Extract positive findings from predictions."""
    tasks_classes = {
        "rhythm": RHYTHM_CLASSES,
        "structural": STRUCTURAL_CLASSES,
        "ischaemia": ISCHAEMIA_CLASSES,
    }

    findings: list[Finding] = []
    for task_name, class_names in tasks_classes.items():
        preds = predictions.get(task_name, [])
        for i, conf in enumerate(preds):
            if i < len(class_names) and conf >= threshold:
                findings.append(
                    Finding(
                        name=class_names[i],
                        confidence=conf * 100.0,
                        task=task_name,
                    )
                )

    # Sort by confidence descending
    findings.sort(key=lambda f: f.confidence, reverse=True)
    return findings


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------


def _severity_color(confidence: float) -> str:
    """Return CSS color class based on confidence level."""
    if confidence >= 80.0:
        return "#dc2626"  # red
    elif confidence >= 50.0:
        return "#d97706"  # amber
    return "#16a34a"  # green


def _quality_badge_color(classification: str) -> str:
    """Return CSS color for quality classification."""
    if classification == "good":
        return "#16a34a"
    elif classification == "marginal":
        return "#d97706"
    return "#dc2626"


def _risk_gauge_color(value: float) -> str:
    """Return CSS color for a risk score (0–1)."""
    if value >= 0.7:
        return "#dc2626"
    elif value >= 0.4:
        return "#d97706"
    return "#16a34a"


def _build_html(
    ecg_record: ECGRecord,
    predictions: dict[str, list[float]],
    findings: list[Finding],
    ecg_image_data_uri: str,
    xai_report: Optional[XAIReport] = None,
    quality_report: Any = None,
    uncertainty_report: Any = None,
    model_version: str = "unknown",
) -> str:
    """Build the HTML string for the clinical PDF report."""

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Patient demographics
    patient_html = ""
    if ecg_record.patient_metadata:
        meta = ecg_record.patient_metadata
        demo_items = []
        for key in ("patient_id", "name", "age", "sex", "dob", "mrn"):
            if key in meta:
                label = key.replace("_", " ").title()
                demo_items.append(f"<li><strong>{label}:</strong> {meta[key]}</li>")
        if demo_items:
            patient_html = (
                '<div class="section">'
                "<h2>Patient Demographics</h2>"
                f'<ul>{"".join(demo_items)}</ul>'
                "</div>"
            )

    # Quality summary
    quality_html = ""
    if quality_report is not None:
        overall = getattr(quality_report, "overall_score", None)
        classification = getattr(quality_report, "overall_classification", "unknown")
        recommendation = getattr(quality_report, "recommendation", "unknown")
        badge_color = _quality_badge_color(classification)

        lead_rows = ""
        per_lead = getattr(quality_report, "per_lead", [])
        for lq in per_lead:
            lead_name = getattr(lq, "lead_name", "?")
            score = getattr(lq, "score", 0)
            lclass = getattr(lq, "classification", "unknown")
            flags: set[str] = getattr(lq, "flags", set())
            flag_str = ", ".join(sorted(flags)) if flags else "—"
            lcolor = _quality_badge_color(lclass)
            lead_rows += (
                f"<tr>"
                f"<td>{lead_name}</td>"
                f"<td>{score:.0f}</td>"
                f'<td style="color:{lcolor};font-weight:bold">{lclass}</td>'
                f"<td>{flag_str}</td>"
                f"</tr>"
            )

        quality_html = (
            '<div class="section">'
            "<h2>Signal Quality</h2>"
            f'<p>Overall: <span class="badge" style="background:{badge_color}">'
            f"{classification}</span> ({overall:.0f}/100) — {recommendation}</p>"
            '<table class="quality-table">'
            "<thead><tr><th>Lead</th><th>Score</th><th>Class</th><th>Flags</th></tr></thead>"
            f"<tbody>{lead_rows}</tbody>"
            "</table>"
            "</div>"
        )

    # ECG waveform
    ecg_html = ""
    if ecg_image_data_uri:
        ecg_html = (
            '<div class="section">'
            "<h2>ECG Waveform</h2>"
            f'<img src="{ecg_image_data_uri}" class="ecg-image" alt="12-Lead ECG" />'
            "</div>"
        )

    # Findings by task
    findings_html = ""
    if findings:
        rows = ""
        for f in findings:
            color = _severity_color(f.confidence)
            bar_width = min(f.confidence, 100.0)
            rows += (
                f"<tr>"
                f"<td>{f.name}</td>"
                f"<td>{f.task}</td>"
                f"<td>"
                f'<div class="conf-bar-bg">'
                f'<div class="conf-bar" style="width:{bar_width}%;background:{color}"></div>'
                f"</div>"
                f"</td>"
                f'<td style="color:{color};font-weight:bold">{f.confidence:.1f}%</td>'
                f"</tr>"
            )
        findings_html = (
            '<div class="section">'
            "<h2>AI Findings</h2>"
            '<table class="findings-table">'
            "<thead><tr><th>Condition</th><th>Task</th>"
            "<th>Confidence</th><th>Score</th></tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
            "</div>"
        )
    else:
        findings_html = (
            '<div class="section">'
            "<h2>AI Findings</h2>"
            '<p class="empty-state">No significant findings detected.</p>'
            "</div>"
        )

    # Risk scores
    risk_html = ""
    risk_preds = predictions.get("risk", [])
    if risk_preds:
        risk_items = ""
        for i, val in enumerate(risk_preds):
            if i < len(RISK_OUTPUTS):
                label = RISK_LABELS.get(RISK_OUTPUTS[i], RISK_OUTPUTS[i])
                color = _risk_gauge_color(val)
                bar_width = val * 100.0
                risk_items += (
                    f'<div class="risk-item">'
                    f'<span class="risk-label">{label}</span>'
                    f'<div class="risk-bar-bg">'
                    f'<div class="risk-bar" style="width:{bar_width:.0f}%;background:{color}">'
                    f"</div></div>"
                    f'<span class="risk-value" style="color:{color}">{val:.3f}</span>'
                    f"</div>"
                )
        risk_html = (
            '<div class="section">'
            "<h2>Risk Scores</h2>"
            f"{risk_items}"
            "</div>"
        )

    # XAI feature attributions
    xai_html = ""
    if xai_report and xai_report.top_features:
        rows = ""
        for feat in xai_report.top_features:
            rows += (
                f"<tr>"
                f"<td>{feat.feature_name}</td>"
                f"<td>{feat.lead}</td>"
                f"<td>{feat.delta_score:.4f}</td>"
                f"</tr>"
            )
        xai_html = (
            '<div class="section">'
            f"<h2>XAI Feature Attributions ({xai_report.task})</h2>"
            '<table class="xai-table">'
            "<thead><tr><th>Feature</th><th>Lead</th><th>Attribution</th></tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
            "</div>"
        )

    # Uncertainty / OOD flags
    uncertainty_html = ""
    if uncertainty_report is not None:
        ood = getattr(uncertainty_report, "ood_flag", None)
        entropy = getattr(uncertainty_report, "entropy_score", None)
        items = []
        if ood is not None:
            ood_color = "#dc2626" if ood else "#16a34a"
            ood_text = "Yes — interpret with caution" if ood else "No"
            items.append(
                f'<li><strong>Out-of-Distribution:</strong> '
                f'<span style="color:{ood_color}">{ood_text}</span></li>'
            )
        if entropy is not None:
            items.append(
                f"<li><strong>Entropy Score:</strong> {entropy:.4f}</li>"
            )
        if items:
            uncertainty_html = (
                '<div class="section">'
                "<h2>Uncertainty Indicators</h2>"
                f'<ul>{"".join(items)}</ul>'
                "</div>"
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Aortica Clinical ECG Report</title>
<style>
@page {{
    size: A4;
    margin: 15mm;
}}
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}
body {{
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 9pt;
    color: #1a1a2e;
    line-height: 1.4;
}}
.header {{
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 8px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}}
.header h1 {{
    font-size: 16pt;
    color: #1e3a5f;
    margin: 0;
}}
.header .meta {{
    font-size: 7pt;
    color: #6b7280;
    text-align: right;
}}
.watermark {{
    background: #fef3c7;
    border: 1px solid #d97706;
    color: #92400e;
    padding: 4px 8px;
    font-size: 7pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 10px;
    border-radius: 3px;
}}
.section {{
    margin-bottom: 12px;
    page-break-inside: avoid;
}}
.section h2 {{
    font-size: 11pt;
    color: #1e3a5f;
    border-bottom: 1px solid #d1d5db;
    padding-bottom: 3px;
    margin-bottom: 6px;
}}
.section ul {{
    list-style: none;
    padding-left: 0;
}}
.section ul li {{
    padding: 2px 0;
}}
.badge {{
    display: inline-block;
    color: white;
    padding: 1px 8px;
    border-radius: 3px;
    font-size: 8pt;
    font-weight: bold;
}}
.ecg-image {{
    width: 100%;
    max-height: 350px;
    object-fit: contain;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
}}
th, td {{
    border: 1px solid #e5e7eb;
    padding: 3px 6px;
    text-align: left;
}}
th {{
    background: #f3f4f6;
    font-weight: bold;
    color: #374151;
}}
.conf-bar-bg {{
    width: 100%;
    height: 10px;
    background: #e5e7eb;
    border-radius: 2px;
    overflow: hidden;
}}
.conf-bar {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s;
}}
.risk-item {{
    display: flex;
    align-items: center;
    margin-bottom: 4px;
}}
.risk-label {{
    width: 220px;
    font-weight: bold;
    font-size: 8pt;
}}
.risk-bar-bg {{
    flex: 1;
    height: 12px;
    background: #e5e7eb;
    border-radius: 3px;
    overflow: hidden;
    margin: 0 8px;
}}
.risk-bar {{
    height: 100%;
    border-radius: 3px;
}}
.risk-value {{
    width: 50px;
    text-align: right;
    font-weight: bold;
    font-size: 8pt;
}}
.empty-state {{
    color: #6b7280;
    font-style: italic;
}}
.footer {{
    margin-top: 16px;
    border-top: 1px solid #d1d5db;
    padding-top: 6px;
    font-size: 6pt;
    color: #9ca3af;
    text-align: center;
}}
</style>
</head>
<body>

<div class="header">
    <h1>Aortica &mdash; AI ECG Analysis Report</h1>
    <div class="meta">
        Model: {model_version}<br>
        Generated: {timestamp}<br>
        Format: {ecg_record.source_format} &bull;
        {ecg_record.sample_rate:.0f} Hz &bull;
        {ecg_record.duration_seconds:.1f}s &bull;
        {ecg_record.num_leads} leads
    </div>
</div>

<div class="watermark">
    &#x26A0; AI Decision Support &mdash; Requires Clinical Review
</div>

{patient_html}
{quality_html}
{ecg_html}
{findings_html}
{risk_html}
{xai_html}
{uncertainty_html}

<div class="footer">
    Aortica v{model_version} &bull; {timestamp} &bull;
    This report is AI-generated decision support and must be reviewed by a qualified clinician.
</div>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_pdf(
    multi_task_output: Any,
    ecg_record: ECGRecord,
    xai_report: Optional[Any] = None,
    output_path: Union[str, Path] = "report.pdf",
    *,
    quality_report: Optional[Any] = None,
    uncertainty_report: Optional[Any] = None,
    model_version: str = "unknown",
    finding_threshold: float = 0.5,
) -> Path:
    """Generate a PDF clinical report.

    Parameters
    ----------
    multi_task_output:
        Multi-task model predictions (``MultiTaskOutput``, dict, or
        similar).
    ecg_record:
        The source ECG recording.
    xai_report:
        Optional XAI feature attribution report (``FeatureAttribution``
        or :class:`XAIReport`).
    output_path:
        File path for the output PDF.
    quality_report:
        Optional signal quality report (``QualityReport``).
    uncertainty_report:
        Optional uncertainty/OOD report (``UncertaintyReport``).
    model_version:
        Model version string shown in report header.
    finding_threshold:
        Confidence threshold (0–1) for including a finding.

    Returns
    -------
    Path
        Absolute path to the generated PDF file.
    """
    weasyprint = _get_weasyprint()
    output_path = Path(output_path)

    # Extract predictions
    predictions = _extract_predictions(multi_task_output)

    # Extract findings
    findings = _get_findings(predictions, threshold=finding_threshold)

    # Adapt XAI report if it's a FeatureAttribution
    adapted_xai: Optional[XAIReport] = None
    if xai_report is not None:
        if isinstance(xai_report, XAIReport):
            adapted_xai = xai_report
        elif hasattr(xai_report, "top_features"):
            # Convert FeatureAttribution to our simplified XAIReport
            top_feats = []
            for fc in xai_report.top_features:
                feat_name = getattr(fc, "feature_name", str(fc))
                lead = getattr(fc, "lead", "")
                delta = getattr(fc, "delta_score", 0.0)
                top_feats.append(XAIFeature(
                    feature_name=feat_name,
                    lead=lead,
                    delta_score=float(delta),
                ))
            adapted_xai = XAIReport(
                top_features=top_feats,
                task=getattr(xai_report, "task", "rhythm"),
            )

    # Render ECG waveform
    ecg_image = _render_ecg_svg(ecg_record)

    # Build HTML
    html = _build_html(
        ecg_record=ecg_record,
        predictions=predictions,
        findings=findings,
        ecg_image_data_uri=ecg_image,
        xai_report=adapted_xai,
        quality_report=quality_report,
        uncertainty_report=uncertainty_report,
        model_version=model_version,
    )

    # Generate PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = weasyprint.HTML(string=html)
    doc.write_pdf(str(output_path))

    return output_path.resolve()
