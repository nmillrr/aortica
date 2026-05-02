# Federated Learning Onboarding Guide

This guide walks new partner sites through joining the Aortica Federated Learning Network — from prerequisites and installation to running your first federated training round.

---

## Prerequisites

Before you begin, ensure you have:

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| **Python** | 3.10 | 3.12 |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 5 GB free | 20 GB free |
| **GPU** | Not required | NVIDIA with CUDA 11.8+ |
| **Network** | Outbound TCP to FL server | Stable broadband (≥1 Mbps) |
| **OS** | Linux, macOS, or Windows | Ubuntu 22.04+ / Debian 12+ |

**Regulatory:** Obtain local IRB / ethics board approval and execute the [Data Use Agreement](DUA_TEMPLATE.md) with the Coordinator before proceeding.

---

## Step 1: Install Aortica with Federated Dependencies

```bash
# Create and activate a virtual environment
python -m venv aortica-fl
source aortica-fl/bin/activate  # Linux/macOS
# aortica-fl\Scripts\activate   # Windows

# Install Aortica with federated learning extras
pip install aortica[cli,federated]

# Verify the installation
aortica --version
aortica federated --help
```

The `[federated]` extra installs Flower (`flwr`), OpenDP, and TenSEAL for differential privacy and secure aggregation.

---

## Step 2: Prepare Your Local Data

Aortica expects ECG data in one of the supported formats (WFDB, DICOM, SCP-ECG, CSV, MAT, HL7 aECG). Organise your data in a directory structure:

```
/path/to/local/data/
├── records/           # ECG files (any supported format)
│   ├── patient_001.hea
│   ├── patient_001.dat
│   ├── patient_002.csv
│   └── ...
├── labels.csv         # Label file with columns: filename, rhythm, structural, ischaemia, risk
└── metadata.csv       # Optional: patient demographics (age, sex) for subgroup reporting
```

**Label format (labels.csv):**

```csv
filename,rhythm,structural,ischaemia,risk
patient_001,AF;LBBB,LVH,STEMI_anterior,0.72;0.45;0.30
patient_002,NSR,,normal,0.10;0.05;0.02
```

> **Important:** Local data never leaves your site. Only encrypted model weight updates are transmitted during federated training.

---

## Step 3: Configure the FL Client

Create a YAML configuration file (`fl_client_config.yaml`):

```yaml
# Server connection
server_address: "fl.example.com:8080"

# Local data
data_path: "/path/to/local/data"

# Training parameters
local_epochs: 3
batch_size: 32
lr: 0.001
sampling_rate: 500
window_seconds: 10.0

# Model configuration
feature_dim: 256
enabled_tasks:
  - rhythm
  - structural
  - ischaemia
  - risk

# Privacy (optional — DP is enabled by default)
# dp_epsilon: 1.0
# dp_delta: 1.0e-5
# dp_max_grad_norm: 1.0

# Reproducibility
seed: 42
```

### Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `server_address` | string | `localhost:8080` | FL server `host:port` |
| `data_path` | string | (required) | Path to local ECG dataset |
| `local_epochs` | int | 1 | Training epochs per round |
| `batch_size` | int | 32 | Training batch size |
| `lr` | float | 0.001 | Learning rate |
| `sampling_rate` | int | 500 | ECG sampling rate (Hz) |
| `window_seconds` | float | 10.0 | Signal window length |
| `feature_dim` | int | 256 | Backbone feature dimension |
| `enabled_tasks` | list | all four | Task heads to train |
| `base_checkpoint` | string | null | Custom starting checkpoint |
| `seed` | int | 42 | Random seed |

---

## Step 4: Test the Connection

Before joining a live training round, verify that your client can reach the FL server:

```bash
aortica federated test-connection fl.example.com:8080
```

**Successful output:**

```
✓ Connection to fl.example.com:8080 succeeded (latency: 45ms)
  Server is reachable and accepting connections.
```

**If the connection fails:**

```
✗ Connection to fl.example.com:8080 failed
  Error: Connection refused
  
  Troubleshooting:
  - Verify the server address and port
  - Check your firewall allows outbound TCP on port 8080
  - Ensure the FL server is running
  - Contact the Coordinator for server status
```

> **Tip:** Use `--timeout` to adjust the connection timeout (default: 10 seconds):
> ```bash
> aortica federated test-connection fl.example.com:8080 --timeout 30
> ```

---

## Step 5: Run Your First Federated Round

Once the connection test passes and the Coordinator confirms the server is ready:

```bash
# Start the FL client
aortica federated client fl_client_config.yaml
```

The client will:

1. Load the pretrained Aortica model (or your custom checkpoint)
2. Connect to the FL server and wait for the round to begin
3. Receive global model weights from the server
4. Train locally on your data for `local_epochs` epochs
5. Apply differential privacy noise to the weight updates
6. Transmit the (privacy-protected) updates to the server
7. Receive the new global model after aggregation
8. Repeat for each configured round

### Monitoring Training

The client logs per-round metrics to stdout:

```
[Round 1/10] Local training: loss=0.4523, rhythm_f1=0.82, structural_f1=0.78
[Round 1/10] Uploaded updates (4,231 examples, ε_spent=0.10)
[Round 2/10] Received aggregated model from server
[Round 2/10] Local training: loss=0.3891, rhythm_f1=0.85, structural_f1=0.81
...
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ImportError: flwr not found` | Missing federated deps | `pip install aortica[federated]` |
| `Connection refused` | Server not running / wrong port | Verify `server_address`, check firewall |
| `Timeout waiting for server` | Slow network or server overloaded | Increase `--timeout`, check bandwidth |
| `CUDA out of memory` | Batch size too large for GPU | Reduce `batch_size` in config |
| `Privacy budget exhausted` | ε fully spent | Reduce `local_epochs` or increase ε |
| `Parameter count mismatch` | Model architecture differs | Ensure `feature_dim` and `enabled_tasks` match server config |

### Getting Help

- **Documentation:** [https://aortica.io/docs/federated](https://aortica.io/docs/federated)
- **Issues:** [https://github.com/nmillrr/aortica/issues](https://github.com/nmillrr/aortica/issues)
- **Coordinator contact:** Provided in your executed DUA

---

## Security Checklist

Before going live, confirm:

- [ ] DUA executed and approved by legal
- [ ] IRB / ethics board approval obtained
- [ ] Virtual environment isolated from other applications
- [ ] Firewall configured to allow only FL server traffic
- [ ] Differential privacy enabled (default ε = 1.0)
- [ ] Secure aggregation enabled (if available)
- [ ] Local data stored on encrypted volume
- [ ] `aortica federated test-connection` passes
- [ ] Coordinator notified of your readiness to join
