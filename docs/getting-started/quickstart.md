# Quick Start

This guide walks through the core Aortica workflows: loading ECGs, running inference, and viewing results.

## 1. CLI Inference

The fastest way to get started:

```bash
# Predict on a single ECG file (WFDB: pass the record's .hea)
aortica predict patient_ecg.hea --model releases/v0.4.0/aortica_v0.4.0.pt

# JSON output
aortica predict patient_ecg.hea --model ... --format json

# Select specific task heads (the risk head is untrained and always suppressed)
aortica predict patient_ecg.hea --model ... --tasks rhythm,ischaemia
```

## 2. Python API

### Load and Inspect an ECG

```python
from aortica.io import read_ecg

# Auto-detects format (WFDB, DICOM, SCP-ECG, CSV, MAT, HL7, PDF)
record = read_ecg("patient_ecg.hea")

print(f"Leads: {record.lead_names}")
print(f"Sample rate: {record.sample_rate} Hz")
print(f"Duration: {record.duration_seconds}s")
print(f"Shape: {record.signals.shape}")
```

### Signal Processing

```python
from aortica.signal import denoise, score_quality, detect_qrs

# Denoise (baseline wander + powerline + high-freq)
clean = denoise(record)

# Quality assessment
quality = score_quality(clean)
print(f"Overall quality: {quality.overall_score}/100 ({quality.overall_classification})")

# QRS detection
r_peaks = detect_qrs(clean)
print(f"Detected {len(r_peaks)} beats")
```

### Model Inference

Model input must match training: 12 leads, **100 Hz, 10 s, µV, not denoised**.
`model_input` builds exactly that from the record as read — pass `record`, not
the denoised `clean`, which is for quality scoring only.

```python
import torch

from aortica.models.checkpoint import load_checkpoint
from aortica.models.output_gating import filter_task_outputs, is_positive
from aortica.models.task_dims import TASK_CLASS_NAMES
from aortica.signal.preprocess import model_input, model_record

# Pretrained weights are not hosted yet; train them with
# scripts/train_reference.py or use a checkpoint you have.
model = load_checkpoint("releases/v0.4.0/aortica_v0.4.0.pt")

x = torch.from_numpy(model_input(record)).unsqueeze(0)   # [1, 12, 1000]
with torch.no_grad():
    output = model(x).as_dict()

# 28 of 72 outputs are trained; drop the rest before using anything.
for task in ("rhythm", "structural", "ischaemia"):
    names, probs, _ = filter_task_outputs(TASK_CLASS_NAMES[task], output[task][0].tolist())
    found = [n for n, p in zip(names, probs) if is_positive(n, p, model.class_thresholds)]
    print(task, found)
```

### Explainability (XAI)

```python
from aortica.xai import explain

# Attribute on the same 100 Hz record the model scored.
attribution = explain(model, model_record(record), task="rhythm")

for feature in attribution.top_features:
    print(f"  {feature.lead} {feature.feature_name}: {feature.delta_score:.3f}")
```

## 3. REST API Server

```bash
# Start the API server
aortica-server

# Or with uvicorn directly
uvicorn aortica.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Then send requests:

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@patient_ecg.dat"

# With XAI annotations
curl -X POST "http://localhost:8000/api/v1/predict?include_xai=true" \
  -F "file=@patient_ecg.dat"
```

## 4. Web UI

The React frontend provides an interactive ECG viewer with AI copilot:

```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173` to access:

- **Dashboard** — overview of recent analyses
- **Upload** — drag-and-drop ECG file upload
- **Results** — interactive 12-lead waveform, AI findings, XAI overlays
- **Batch** — process multiple ECGs at once

## 5. Benchmarking

```bash
# Run the full evaluation harness on PTB-XL
aortica benchmark /path/to/ptbxl/ --format table

# Export results as CSV
aortica benchmark /path/to/ptbxl/ --csv-export results.csv
```

## Next Steps

- [API Reference](../api/overview.md) — detailed module documentation
- [CLI Reference](../cli/commands.md) — all CLI commands and options
- [Deployment Guide](../deployment/docker.md) — Docker and edge deployment
- [Clinical Background](../clinical/ecg-primer.md) — ECG and AI context
