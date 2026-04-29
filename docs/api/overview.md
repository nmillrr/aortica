# API Reference — Overview

Aortica's Python API is organized into focused subpackages:

| Module | Description |
|--------|-------------|
| [`aortica.io`](io.md) | Universal ECG format readers (WFDB, DICOM, SCP-ECG, CSV, MAT, HL7, PDF) |
| [`aortica.signal`](signal.md) | QRS detection, denoising, signal quality scoring |
| [`aortica.models`](models.md) | Multi-task deep learning engine, training, calibration |
| [`aortica.xai`](xai.md) | Explainability — integrated gradients, VAE latent factors |
| [`aortica.evaluation`](evaluation.md) | Benchmark harness, metrics, demographic subgroup reporting |
| [`aortica.edge`](edge.md) | ONNX export, INT8 quantization, edge validation |
| [`aortica.api`](rest.md) | FastAPI REST API endpoints |
| [`aortica.sync`](sync.md) | Offline storage, sync engine, bandwidth management |

## Common Patterns

### ECGRecord

All format readers return an `ECGRecord` — the canonical in-memory ECG representation:

```python
from aortica.io import read_ecg

record = read_ecg("patient.dat")
# record.signals: numpy array [leads, samples]
# record.sample_rate: int (Hz)
# record.lead_names: list[str]
# record.duration_seconds: float
```

### Multi-Task Output

Model inference returns a `MultiTaskOutput`:

```python
output = model(input_tensor)
# output.rhythm:     Tensor [batch, 22]  — sigmoid probabilities
# output.structural: Tensor [batch, 15]
# output.ischaemia:  Tensor [batch, 10]
# output.risk:       Tensor [batch, 3]   — continuous 0–1 scores
```
