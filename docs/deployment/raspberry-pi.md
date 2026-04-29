# Raspberry Pi Deployment

Deploy Aortica on Raspberry Pi 4/5 for low-cost, offline ECG analysis in rural clinics.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Board | Raspberry Pi 4 (2GB) | Raspberry Pi 5 (4GB) |
| Storage | 16GB microSD | 32GB+ microSD (Class 10/A2) |
| Power | 5V 3A USB-C | Official PSU |
| Network | Optional (Wi-Fi for sync) | Ethernet for initial setup |

## Installation

### 1. OS Setup

Flash Raspberry Pi OS (64-bit Lite) to the microSD card using [Raspberry Pi Imager](https://www.raspberrypi.com/software/).

### 2. Install Aortica

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
python3 -m venv ~/aortica-env
source ~/aortica-env/bin/activate
pip install "aortica[cli,edge]"
```

### 3. Download Edge Model

```bash
python -c "from aortica.models.registry import load_pretrained; load_pretrained('latest', variant='edge')"
```

### 4. Verify

```bash
aortica predict test_ecg.dat
```

## Auto-Start on Boot

Install the systemd service:

```bash
sudo cp aortica-edge.service /etc/systemd/system/
sudo systemctl enable aortica-edge
sudo systemctl start aortica-edge
```

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference latency (p50) | ≤350ms | — |
| Peak memory | ≤512MB | — |
| Model size (INT8 ONNX) | ≤8MB | — |
| AUC gap vs full model | ≤3% | — |

## Offline Operation

The edge deployment works fully offline:

1. ECG files ingested via USB or local network
2. Inference runs locally using the INT8 ONNX model via ONNX Runtime ARM64
3. Results stored in encrypted SQLite database
4. When connectivity is available, results sync to the central server
