# Raspberry Pi Deployment

Deploy Aortica on Raspberry Pi 4/5 for low-cost, offline ECG analysis in rural clinics.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Board | Raspberry Pi 4 (2GB) | Raspberry Pi 5 (4GB) |
| Storage | 16GB microSD | 32GB+ microSD (Class 10/A2) |
| Power | 5V 3A USB-C | Official PSU |
| Network | Optional (Wi-Fi for sync) | Ethernet for initial setup |

## Quick Setup (Automated)

The fastest way to deploy is using the included setup script:

```bash
# Clone or download the repository
git clone https://github.com/nmillrr/aortica.git
cd aortica

# Run the setup script (requires sudo)
sudo ./create_pi_image_script.sh
```

This script will:

1. Install system dependencies (Python 3, pip, venv, libatlas)
2. Create a dedicated `aortica` service user
3. Set up a Python virtual environment at `/opt/aortica/venv`
4. Install `aortica[cli,edge]` into the virtualenv
5. Download the INT8 ONNX edge model (~5–8 MB)
6. Create data and log directories
7. Install and enable the systemd service for auto-start

## Manual Installation

### 1. SD Card Preparation

Flash **Raspberry Pi OS (64-bit Lite)** to the microSD card using [Raspberry Pi Imager](https://www.raspberrypi.com/software/).

### 2. Install Aortica

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv libatlas-base-dev
python3 -m venv ~/aortica-env
source ~/aortica-env/bin/activate
pip install "aortica[cli,edge]"
```

### 3. Download Edge Model

```bash
python -c "from aortica.models.registry import load_pretrained; load_pretrained('latest', variant='edge')"
```

### 4. First-Run Verification

```bash
aortica predict test_ecg.dat
```

You should see multi-task predictions (rhythm, structural, ischaemia, risk) printed to the terminal.

## Deployment Profile Configuration

Aortica provides a `RaspberryPiProfile` dataclass for programmatic configuration:

```python
from aortica.edge.deploy_profiles import RaspberryPiProfile

# Default profile (RPi4, INT8, 512 MB, 350 ms target)
profile = RaspberryPiProfile()
print(profile.summary())

# Custom profile for RPi5 with more memory
profile = RaspberryPiProfile(
    device_name="rpi5",
    max_memory_mb=1024,
    target_latency_ms=200,
    num_threads=4,
)

# Save/load configuration
profile.to_json("/etc/aortica/profile.json")
loaded = RaspberryPiProfile.from_json("/etc/aortica/profile.json")
```

### Profile Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantization` | `INT8` | Model quantization level |
| `max_memory_mb` | `512` | Maximum memory budget (MB) |
| `target_latency_ms` | `350` | Target inference latency (ms) |
| `num_threads` | `4` | ONNX Runtime threads |
| `enable_sync` | `True` | Enable automatic result sync |
| `sync_interval_minutes` | `30` | Sync interval (minutes) |
| `data_dir` | `/var/lib/aortica/data` | Encrypted result storage |
| `log_dir` | `/var/log/aortica` | Application logs |

## Auto-Start on Boot (systemd)

The setup script installs a systemd service automatically. To install manually:

```bash
sudo cp aortica-edge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable aortica-edge
sudo systemctl start aortica-edge
```

You can also generate a custom service file programmatically:

```python
from aortica.edge.deploy_profiles import write_systemd_service, RaspberryPiProfile

profile = RaspberryPiProfile(max_memory_mb=1024)
write_systemd_service("/etc/systemd/system/aortica-edge.service", profile)
```

### Service Management

```bash
# Start / stop / restart
sudo systemctl start aortica-edge
sudo systemctl stop aortica-edge
sudo systemctl restart aortica-edge

# Check status
sudo systemctl status aortica-edge

# View logs
sudo journalctl -u aortica-edge -f
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

## Troubleshooting

### `aortica predict` hangs or is slow

- Ensure you're using the INT8 quantized model, not the full FP32 model
- Check available memory: `free -m` (should have ≥256 MB free)
- Reduce threads if thermal throttling: `ORT_NUM_THREADS=2 aortica predict file.dat`

### Service fails to start

```bash
# Check logs for errors
sudo journalctl -u aortica-edge --no-pager -n 50

# Verify the edge model exists
ls -la ~/.cache/aortica/aortica_edge_int8.onnx

# Test manually
sudo -u aortica aortica predict --help
```

### Sync not working

- Verify network connectivity: `curl -s https://httpbin.org/ip`
- Check sync config: ensure `remote_url` is set correctly
- Review sync logs in `/var/log/aortica/`
