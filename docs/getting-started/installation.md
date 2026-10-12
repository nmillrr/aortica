# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- pip 21.0+

## Basic Installation

Aortica is not yet published to PyPI, Docker Hub or the HuggingFace Hub —
install from a clone of the repository.

```bash
git clone https://github.com/nmillrr/aortica.git
cd aortica
pip install -e .
```

## Installation with Extras

Aortica uses optional dependency groups for different deployment scenarios.
Run these from the repository root.

### Full Platform (API + CLI + inference)

```bash
pip install -e ".[api,cli,torch]"
```

### Researcher Setup (Training + Evaluation)

```bash
pip install -e ".[torch,cli,dev]"
```

### Edge Deployment

```bash
pip install -e ".[cli,edge]"
```

The INT8 edge model is in the repository at
`releases/v0.3.0/aortica_edge_int8_v0.3.0.onnx`.

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
| `hub` | HuggingFace Hub | Pre-trained model download (repository not yet public) |
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

No images are published; build them locally:

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
# 0.4.0

aortica --help
```
