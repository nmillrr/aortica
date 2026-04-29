# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- pip 21.0+

## Basic Installation

Install the core library (format readers, signal processing, models):

```bash
pip install aortica
```

## Installation with Extras

Aortica uses optional dependency groups for different deployment scenarios:

### Full Platform (API + CLI)

```bash
pip install "aortica[api,cli]"
```

### Researcher Setup (Training + Evaluation)

```bash
pip install "aortica[torch,cli,dev]"
```

### Edge Deployment

```bash
pip install "aortica[cli,edge]"
```

### All Optional Dependencies

| Extra | Packages | Use Case |
|-------|----------|----------|
| `torch` | PyTorch ≥2.0 | Training, inference |
| `tf` | TensorFlow ≥2.13 | TF/Keras backend |
| `signal` | NeuroKit2 | Advanced signal processing |
| `api` | FastAPI, Uvicorn | REST API server |
| `grpc` | gRPC | High-throughput inference |
| `cli` | Click, Rich | Command-line interface |
| `edge` | ONNX, ONNX Runtime | Edge model deployment |
| `scan` | OpenCV, pdfplumber | PDF/image ECG digitization |
| `hub` | HuggingFace Hub | Pre-trained model download |
| `sync` | cryptography | Encrypted offline storage |
| `auth` | authlib, PyJWT | OAuth/API key authentication |
| `dev` | pytest, mypy, ruff | Development and testing |

## Development Installation

Clone and install in editable mode:

```bash
git clone https://github.com/nmillrr/aortica.git
cd aortica
pip install -e ".[dev,torch,cli,api]"
```

## Docker Installation

Pull pre-built images:

```bash
# Server (amd64)
docker build -f Dockerfile.server -t aortica-server .
docker run -p 8000:8000 aortica-server

# Edge (arm64)
docker build -f Dockerfile.edge -t aortica-edge .
```

Or use Docker Compose:

```bash
docker compose up
```

See the [Docker Deployment Guide](../deployment/docker.md) for details.

## Verify Installation

```bash
python -c "import aortica; print(aortica.__version__)"
# 0.2.0

aortica --help
```
