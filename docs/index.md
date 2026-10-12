# Aortica

**Open-source AI ECG analysis platform** — closing the most critical gaps in clinical ECG: poor device generalization, narrow single-task models, black-box outputs, inaccessible tooling, and exclusion of rural/low-resource settings.

---

## What is Aortica?

Aortica is a self-hosted, open-source toolkit for AI-powered 12-lead ECG analysis. Clinicians and institutions download and run it locally — **no data ever leaves the deployment site**.

### Multi-Task AI Engine

The model has 72 outputs across four heads; **28 are trained** (PTB-XL +
Chapman-Shaoxing). The rest — including every risk prediction — have no label
source in the public training data and are suppressed on every inference path.

| Task Head | Trained outputs |
|-----------|-----------------|
| **Rhythm & Conduction** | 18 — AF, AFL, SVT, AVRT, sinus brady/tachycardia, PAC, PVC, 1st/2nd/3rd-degree AV block, LBBB, RBBB, LAFB, LPFB, WPW, paced, normal sinus |
| **Structural** | 4 — LVH, RVH, left/right atrial enlargement |
| **Ischaemia & Repolarisation** | 6 — STEMI, posterior MI, old MI, QTc prolongation, digitalis effect, early repolarisation |
| **Risk Prediction** | none trained |

!!! warning "Research use only"
    Not FDA-cleared or CE-marked; not for diagnosis, treatment or triage.

### Key Features

- :material-file-multiple: **Universal format support** — WFDB, DICOM, SCP-ECG, CSV, MAT, HL7 aECG, PDF/image scans
- :material-brain: **Per-class operating points** — thresholds tuned on validation data ship with each checkpoint
- :material-chart-bar: **ECG-native XAI** — integrated gradients mapped to named ECG features (P wave, QRS, ST segment)
- :material-cellphone-link: **Edge deployment** — INT8 ONNX models for ARM devices (Raspberry Pi)
- :material-wifi-off: **Offline-first** — PWA with in-browser WASM inference, encrypted local storage
- :material-docker: **Docker** — Dockerfiles for server (amd64) and edge (arm64); build locally
- :material-api: **REST API + gRPC + CLI** — integrate with any workflow

---

## Quick Install

Aortica is not yet published to PyPI, Docker Hub or the HuggingFace Hub —
install from a clone of the repository.

```bash
git clone https://github.com/nmillrr/aortica.git
cd aortica
pip install -e ".[api,cli,torch]"
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
