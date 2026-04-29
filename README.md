# Aortica

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](https://github.com/nmillrr/aortica)

**Open-source AI ECG analysis platform** designed to close the most critical gaps in clinical ECG: poor device generalization, narrow single-task models, black-box outputs, inaccessible tooling, and exclusion of rural/low-resource settings.

> **Self-hosted & privacy-first.** No data ever leaves the deployment site. Clinicians and institutions download and run Aortica locally — via Docker or `pip install` — preserving patient privacy, eliminating recurring infrastructure costs, and enabling deployment in data-sovereignty-constrained settings.

---

## ✨ Key Features

### 🔌 Universal ECG Ingestion
- **7+ format readers** — WFDB (.hea/.dat), DICOM (Supp 30/130), SCP-ECG, CSV, MAT, HL7 aECG XML
- **PDF/image scan digitization** — Scanned paper ECGs (PDF, PNG, JPG, TIFF) digitized via OpenCV grid detection and waveform trace extraction
- **Auto-detection** — `aortica.io.read_ecg(path)` dispatches to the correct reader by extension and file magic bytes

### 🧠 Multi-Task AI Engine
Four task heads sharing a single ResNet backbone with cross-lead temporal attention:

| Task Head | Classes | Examples |
|-----------|---------|----------|
| **Rhythm & Conduction** | 22 → 28* | AF, VT, VF, LBBB, RBBB, WPW, Brugada, CPVT |
| **Structural & Functional** | 15 → 19* | LVH, LVSD, HCM, ARVC, Takotsubo, RV strain in PE |
| **Ischaemia & Metabolic** | 10 → 19* | STEMI, de Winter, Wellens, hyperkalaemia grading, TCA toxicity |
| **Risk Prediction** | 3 → 6* | 1-year mortality, HF hospitalization, AF onset, ECG-predicted EF |

<sub>* Expanded dimensions in Phase 3</sub>

### 🎯 Trustworthy AI
- **Temperature-scaled calibration** — well-calibrated probability outputs per task head
- **Conformal prediction** — per-prediction confidence intervals at configurable coverage levels
- **OOD detection** — Mahalanobis distance flagging of out-of-distribution inputs
- **Integrated gradient XAI** — attributions mapped to named ECG features (P wave, QRS complex, ST segment, T wave)
- **VAE latent factor model** — interpretable 24-dimensional latent space with synthetic waveform generation

### 📊 Signal Processing
- **QRS detection** — NeuroKit2 and Pan-Tompkins backends
- **Denoising** — baseline wander removal, powerline notch filter (50/60 Hz auto-detect), high-frequency noise reduction
- **Quality scoring** — per-lead 0–100 scores with good/marginal/poor classification and lead-off/clipping detection

### 🔬 Reproducible Benchmarking
- Per-task metrics: macro-F1, per-class AUC, sensitivity/specificity, ECE, C-index, Brier score
- Demographic subgroup stratification (age decile, sex)
- Equity gating: blocks releases with statistically significant demographic performance gaps
- Auto-generated public performance cards (markdown + CSV) per model version

---

## 🏗 Architecture

```
aortica/
├── io/            # ECG format readers (WFDB, DICOM, CSV, MAT, SCP, HL7, PDF scan)
├── signal/        # Signal processing (QRS detection, denoising, quality scoring)
├── models/        # Deep learning (backbone, attention, task heads, training, calibration)
├── xai/           # Explainability (integrated gradients, VAE latent factor model)
├── evaluation/    # Benchmark harness, equity gating, performance cards
├── data/          # Dataset loaders (PTB-XL), PyTorch Dataset, TF tf.data wrappers
├── edge/          # ONNX export, MobileNet-1D backbone, distillation, INT8 quantization
├── api/           # FastAPI REST API + gRPC service
├── cli/           # Click + Rich CLI (predict, benchmark, train, profile)
├── sync/          # Offline result storage (SQLite + AES-256), vector clock sync
├── federated/     # Flower FL server/client, FedAvg/FedProx/SCAFFOLD, DP, secure aggregation
├── integration/   # FHIR R4, HL7 v2.x, DICOM SR, DIMSE, SCP serial capture, SMART on FHIR
├── reports/       # PDF clinical reports, JSON-LD, CSV batch export
├── retrieval/     # Latent space index, case-based ECG retrieval (Annoy/FAISS)
├── validation/    # Prospective study tooling, performance monitoring, adverse event tracking
└── utils/         # Shared utilities

frontend/          # React + Vite + TypeScript PWA (offline-capable with ONNX WASM fallback)
docs/              # MkDocs Material documentation site
  ├── regulatory/  # IEC 80601-2-86, FDA SaMD, CE-MDR, TRIPOD-AI/STARD-AI/CONSORT-AI templates
  └── validation/  # Prospective study protocol and data collection templates
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nmillrr/aortica.git
cd aortica

# Install in editable mode (core only)
pip install -e .

# Install with optional dependency groups
pip install -e ".[cli]"          # CLI (Click + Rich)
pip install -e ".[api]"          # REST API (FastAPI + Uvicorn)
pip install -e ".[edge]"         # ONNX edge inference
pip install -e ".[torch]"        # PyTorch backend
pip install -e ".[tf]"           # TensorFlow/Keras backend
pip install -e ".[scan]"         # PDF/image ECG digitization (OpenCV)
pip install -e ".[dev]"          # Development tools (pytest, mypy, ruff)

# Install everything
pip install -e ".[cli,api,edge,torch,scan,dev]"
```

### CLI Usage

```bash
# Run inference on an ECG file
aortica predict recording.hea

# JSON output with specific task heads
aortica predict recording.hea --format json --tasks rhythm,ischaemia

# Benchmark a model on PTB-XL
aortica benchmark /path/to/ptbxl --tasks all --output results.csv

# Train a model from config
aortica train config.yaml
```

### REST API

```bash
# Start the API server
aortica-server
# or: uvicorn aortica.api.app:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# Single ECG inference
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@recording.hea"

# Batch inference
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -F "files=@ecg1.hea" -F "files=@ecg2.hea"
```

### Python API

```python
from aortica.io import read_ecg
from aortica.signal import denoise, score_quality
from aortica.models import AorticaModel

# Load and preprocess
ecg = read_ecg("recording.hea")          # auto-detects format
ecg = denoise(ecg)                        # baseline + powerline + HF removal
quality = score_quality(ecg)              # per-lead quality scores

# Multi-task inference
model = AorticaModel.load("checkpoint.pt")
output = model(ecg)                       # rhythm, structural, ischaemia, risk

# Explainability
from aortica.xai import explain
attribution = explain(model, ecg, task="rhythm")
print(attribution.top_features)           # e.g., [("QRS width", 0.42), ...]
```

---

## 🌍 Edge & Rural Deployment

Aortica is designed for **rural clinics with intermittent internet, a laptop, and a USB-attached ECG device**.

### Hybrid Offline Architecture
- **Local server mode** — FastAPI backend + full PyTorch model when a workstation is available
- **PWA offline mode** — React frontend caches itself + ONNX INT8 edge model; inference falls back to ONNX Runtime Web (WASM) in the browser when the server is unreachable
- **No internet required** after initial setup

### Edge Model
- **MobileNet-1D backbone** — depthwise-separable convolutions, ≤2.5M parameters
- **Knowledge distillation** — trained from the full ResNet model (KL divergence + hard labels)
- **INT8 quantization** — static quantization via ONNX Runtime for ARM deployment
- **Target**: AUC within 3% of full model, inference <350ms on Raspberry Pi 4

### Raspberry Pi Deployment
```bash
# Deploy on ARM64 hardware
docker run -p 8000:8000 aortica/edge:latest-arm64

# Or install directly
pip install aortica[cli,edge]
aortica predict ecg_file.hea  # uses INT8 ONNX model
```

### Docker Deployment

```bash
# Build the server image (amd64 — includes API server + built frontend)
docker build -f Dockerfile.server -t aortica/server:latest .

# Build the edge image (arm64 — CLI + ONNX edge inference only)
docker build -f Dockerfile.edge -t aortica/edge:latest .

# Run the server
docker run -p 8000:8000 aortica/server:latest

# Run edge inference on a file
docker run -v /path/to/ecg:/data aortica/edge:latest predict /data/recording.hea

# Local development with docker-compose (API + frontend dev server)
docker compose up
```

### CHW-Facing Simplified Output
For community health workers without cardiology training:
- **Three-tier system**: Low risk → Refer for assessment → Urgent referral
- Plain-language summaries with recommended actions
- Localizable via JSON locale files

---

## 🤝 Federated Learning

Train models across institutions without sharing raw patient data.

- **Flower framework** — FedAvg, FedProx, and SCAFFOLD aggregation strategies
- **Differential privacy** — OpenDP integration with configurable ε budget (default ε=1.0)
- **Secure aggregation** — CKKS homomorphic encryption via TenSEAL
- **CLI-driven** — `aortica federated server config.yaml` / `aortica federated client config.yaml`

---

## 🏥 EHR & Device Integration

| Standard | Description |
|----------|-------------|
| **FHIR R4** | DiagnosticReport + Observation resources with LOINC/SNOMED codes |
| **HL7 v2.x** | ORU^R01 messages for legacy EHR systems |
| **DICOM SR** | Structured Report write-back to PACS |
| **DICOM DIMSE** | C-STORE/C-FIND for GE MUSE-style ECG management |
| **SCP-ECG Serial** | Serial/USB capture from legacy ECG carts |
| **SMART on FHIR** | EHR-embedded launch with patient context |

### Report Generation
- **PDF reports** — ECG waveform, AI findings, XAI annotations (via WeasyPrint)
- **JSON-LD** — Machine-readable reports with Schema.org + SNOMED/LOINC ontology links
- **CSV batch export** — Streaming export for cohort-level research analytics
- **Worklist prioritization** — AI-sorted by clinical urgency

---

## 📋 Regulatory Readiness

Aortica ships document templates and validation tooling for regulatory pathways:

- **IEC 80601-2-86** — Algorithm Testing Documentation template
- **FDA SaMD** — Pre-submission template
- **CE-MDR** — Technical file template (Annex I, ISO 14971, IEC 62304)
- **TRIPOD-AI / STARD-AI / CONSORT-AI** — Publication reporting checklists
- **CI enforcement** — minimum per-class performance thresholds block releases
- **Prospective validation** — multi-site study protocol template, data collection pipeline, automated quarterly performance reports

---

## 🖥 Web UI

React + Vite + TypeScript frontend with:

- **Interactive ECG viewer** — 12-lead standard layout with grid, pan/zoom, caliper tool
- **AI Copilot panel** — ranked findings with confidence, severity badges, and clinical suggestion prompts
- **Second Reader mode** — enter your interpretation, compare against AI, see discrepancies
- **Edge-Case Spotlight** — rare but dangerous findings (Brugada, Wellens, de Winter) are never buried
- **Batch processing dashboard** — upload multiple ECGs, sortable results table, CSV export
- **XAI overlays** — gradient heatmaps and named segment callouts on the waveform

---

## 🧪 Development

```bash
# Run the full test suite
pytest

# Run type checking
mypy aortica

# Run linter
ruff check aortica

# Run tests with coverage
pytest --cov=aortica --cov-report=term-missing
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| ML (primary) | PyTorch |
| ML (parallel) | TensorFlow / Keras |
| Signal processing | NeuroKit2, SciPy |
| REST API | FastAPI + Uvicorn |
| High-throughput API | gRPC |
| CLI | Click + Rich |
| Web frontend | React + Vite + TypeScript (PWA) |
| Edge inference | ONNX Runtime, ONNX Runtime Web (WASM) |
| Documentation | MkDocs Material |
| Federated learning | Flower (flwr) |
| Differential privacy | OpenDP |
| Encryption | cryptography (Fernet / AES-256) |
| EHR integration | fhir.resources, hl7apy, pydicom, pynetdicom |
| Report generation | WeasyPrint, pyld |
| Case retrieval | Annoy / FAISS |
| PDF scan digitization | OpenCV, pdfplumber |
| Containerization | Docker (amd64 + arm64) |

---

## 🗺 Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 0** | Foundation — format readers, signal processing, baseline model | ✅ Complete |
| **Phase 1** | Core Engine — multi-task model, calibration, XAI, benchmarking | ✅ Complete |
| **Phase 2** | Edge & Rural Deployment — API, CLI, Web UI, ONNX, offline sync, Docker | 🔄 In Progress |
| **Phase 3** | Federated Learning & Equity — Flower SDK, DP, equity gating, expanded task heads | 📋 Planned |
| **Phase 4** | Regulatory & Scale — FHIR/HL7/DICOM integration, reports, case retrieval, regulatory templates | 📋 Planned |

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **PTB-XL** (Wagner et al. 2020, PhysioNet) — primary training and evaluation dataset (CC BY 4.0)
- **PhysioNet** — public ECG databases for development and testing
- No proprietary data is used. All training uses public or federated data.
