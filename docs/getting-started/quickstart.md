# Quick Start

This guide walks through the core Aortica workflows: loading ECGs, running inference, and viewing results.

## 1. CLI Inference

The fastest way to get started:

```bash
# Predict on a single ECG file
aortica predict patient_ecg.dat

# JSON output
aortica predict patient_ecg.dat --format json

# Select specific task heads
aortica predict patient_ecg.dat --tasks rhythm,risk
```

## 2. Python API

### Load and Inspect an ECG

```python
from aortica.io import read_ecg

# Auto-detects format (WFDB, DICOM, SCP-ECG, CSV, MAT, HL7, PDF)
record = read_ecg("patient_ecg.dat")

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

```python
from aortica.models import AorticaModel, load_pretrained

# Load pre-trained model from HuggingFace Hub
model = load_pretrained("latest")

# Or load a local checkpoint
# model = AorticaModel()
# model.load_state_dict(torch.load("checkpoint.pt")["model_state_dict"])

import torch
import numpy as np

# Prepare input tensor
signals = torch.tensor(clean.signals, dtype=torch.float32).unsqueeze(0)

# Run inference
output = model(signals)

print(f"Rhythm predictions: {output.rhythm.shape}")       # [1, 22]
print(f"Structural predictions: {output.structural.shape}") # [1, 15]
print(f"Ischaemia predictions: {output.ischaemia.shape}")  # [1, 10]
print(f"Risk scores: {output.risk.shape}")                 # [1, 3]
```

### Explainability (XAI)

```python
from aortica.xai import explain

# Integrated gradient attribution mapped to ECG features
attribution = explain(model, record, task="rhythm")

for feature in attribution.top_features:
    print(f"  {feature.segment_name}: {feature.delta_score:.3f}")
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
