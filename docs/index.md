# Aortica

**Open-source AI ECG analysis platform** — closing the most critical gaps in clinical ECG: poor device generalization, narrow single-task models, black-box outputs, inaccessible tooling, and exclusion of rural/low-resource settings.

---

## What is Aortica?

Aortica is a self-hosted, open-source toolkit for AI-powered 12-lead ECG analysis. Clinicians and institutions download and run it locally — **no data ever leaves the deployment site**.

### Multi-Task AI Engine

A single forward pass through Aortica's deep learning engine produces:

| Task Head | Classes | Output |
|-----------|---------|--------|
| **Rhythm & Conduction** | 22 classes | AF, AFL, SVT, VT, VF, AV blocks, BBB, WPW, and more |
| **Structural & Functional** | 15 classes | LVH, RVH, LVSD, HCM, DCM, amyloidosis, and more |
| **Ischaemia & Metabolic** | 10 classes | STEMI, NSTEMI, hyperkalaemia, QTc prolongation, and more |
| **Risk Prediction** | 3 continuous | 1-year mortality, HF hospitalization, AF onset |

### Key Features

- :material-file-multiple: **Universal format support** — WFDB, DICOM, SCP-ECG, CSV, MAT, HL7 aECG, PDF/image scans
- :material-brain: **Calibrated uncertainty** — conformal prediction sets and OOD detection
- :material-chart-bar: **ECG-native XAI** — integrated gradients mapped to named ECG features (P wave, QRS, ST segment)
- :material-cellphone-link: **Edge deployment** — INT8 ONNX models for ARM devices (Raspberry Pi)
- :material-wifi-off: **Offline-first** — PWA with in-browser WASM inference, encrypted local storage
- :material-docker: **Docker ready** — multi-arch images for server (amd64) and edge (arm64)
- :material-api: **REST API + gRPC + CLI** — integrate with any workflow

---

## Quick Install

```bash
pip install aortica
```

For the full platform with API server and CLI:

```bash
pip install "aortica[api,cli]"
```

See the [Installation Guide](getting-started/installation.md) for detailed instructions.

---

## Quick Start

```bash
# Run inference on an ECG file
aortica predict patient_ecg.dat

# Start the API server
aortica-server

# Run a benchmark
aortica benchmark /path/to/ptbxl/
```

See the [Quick Start Guide](getting-started/quickstart.md) for a walkthrough.

---

## Architecture

```
aortica/
├── io/          # Universal ECG format readers
├── signal/      # QRS detection, denoising, quality scoring
├── models/      # Multi-task deep learning engine
├── xai/         # Explainability (integrated gradients, VAE)
├── evaluation/  # Benchmark harness, metrics
├── edge/        # ONNX export, quantization, edge profiles
├── api/         # FastAPI REST API, gRPC service
├── cli/         # Click-based CLI tool
├── sync/        # Offline storage, sync engine
├── data/        # Dataset loaders, augmentations
└── utils/       # Shared utilities
```

---

## License

Apache 2.0 — see [LICENSE](https://github.com/nmillrr/aortica/blob/main/LICENSE).
