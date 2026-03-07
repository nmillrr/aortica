# Aortica

[![CI](https://github.com/nmillrr/aortica/actions/workflows/ci.yml/badge.svg)](https://github.com/nmillrr/aortica/actions/workflows/ci.yml)

Open-source AI ECG analysis platform designed to close the most critical gaps in clinical ECG interpretation.

## Overview

Aortica provides:

- **Universal ECG format reader** — WFDB, DICOM, SCP-ECG, CSV, MAT, HL7 aECG, XML
- **AI-based signal quality assessment** — QRS detection, denoising, per-segment quality scoring
- **Multi-task deep learning engine** — rhythm, structural, ischaemia, and risk prediction heads sharing a single backbone
- **Calibrated uncertainty estimation** — temperature scaling and conformal prediction
- **ECG-native explainability** — integrated gradient attribution mapped to named ECG features, plus a VAE latent factor model
- **Reproducible benchmarking** — evaluation harness with demographic subgroup reporting

## Installation

```bash
# Clone the repository
git clone https://github.com/nmillrr/aortica.git
cd aortica

# Install in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install with TensorFlow support
pip install -e ".[tf]"
```

## Project Structure

```
aortica/
├── io/          # ECG format readers (WFDB, DICOM, CSV, MAT, etc.)
├── signal/      # Signal processing (QRS detection, denoising, quality)
├── models/      # Deep learning models (backbone, task heads, training)
├── xai/         # Explainability (integrated gradients, VAE)
├── evaluation/  # Benchmarking and metrics
├── data/        # Dataset loaders and data utilities
└── utils/       # Shared utilities
```

## Development

```bash
# Run tests
pytest

# Run type checking
mypy aortica

# Run linter
ruff check aortica
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
