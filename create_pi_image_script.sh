#!/usr/bin/env bash
# ----------------------------------------------------------------
# Aortica Edge — Raspberry Pi Image Setup Script
#
# This script prepares a Raspberry Pi (4 or 5) for offline ECG
# analysis using the Aortica INT8 ONNX edge model.
#
# Usage:
#   chmod +x create_pi_image_script.sh
#   sudo ./create_pi_image_script.sh
#
# Prerequisites:
#   - Raspberry Pi OS (64-bit Lite) flashed to microSD
#   - Internet connection for initial package download
# ----------------------------------------------------------------
set -euo pipefail

echo "=== Aortica Edge Setup for Raspberry Pi ==="

# ---- 1. System dependencies ----
echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv libatlas-base-dev

# ---- 2. Create service user ----
echo "[2/7] Creating service user 'aortica'..."
if ! id -u aortica > /dev/null 2>&1; then
    sudo useradd --system --create-home --shell /usr/sbin/nologin aortica
fi

# ---- 3. Python virtual environment ----
echo "[3/7] Setting up Python virtual environment..."
VENV_DIR="/opt/aortica/venv"
sudo mkdir -p /opt/aortica
sudo python3 -m venv "$VENV_DIR"
sudo "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# ---- 4. Install Aortica ----
echo "[4/7] Installing aortica[cli,edge]..."
sudo "$VENV_DIR/bin/pip" install "aortica[cli,edge]"
sudo ln -sf "$VENV_DIR/bin/aortica" /usr/local/bin/aortica

# ---- 5. Download edge model ----
echo "[5/7] Downloading INT8 edge model..."
sudo -u aortica "$VENV_DIR/bin/python" -c \
    "from aortica.models.registry import load_pretrained; load_pretrained('latest', variant='edge')"

# ---- 6. Create data/log directories ----
echo "[6/7] Creating data and log directories..."
sudo mkdir -p /var/lib/aortica/data/inbox
sudo mkdir -p /var/log/aortica
sudo chown -R aortica:aortica /var/lib/aortica/data
sudo chown -R aortica:aortica /var/log/aortica

# ---- 7. Install systemd service ----
echo "[7/7] Installing systemd service..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/aortica-edge.service" ]; then
    sudo cp "$SCRIPT_DIR/aortica-edge.service" /etc/systemd/system/aortica-edge.service
else
    # Generate inline if service file not co-located
    cat > /tmp/aortica-edge.service << 'EOF'
[Unit]
Description=Aortica Edge ECG Analysis Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=aortica
Group=aortica
Environment=AORTICA_DATA_DIR=/var/lib/aortica/data
Environment=AORTICA_LOG_DIR=/var/log/aortica
Environment=ORT_NUM_THREADS=4
ExecStart=/usr/bin/env aortica predict --watch /var/lib/aortica/data/inbox --model ~/.cache/aortica/aortica_edge_int8.onnx
Restart=on-failure
RestartSec=5
WatchdogSec=60

# Resource limits
MemoryMax=512M
CPUQuota=90%

# Security hardening
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/var/lib/aortica/data /var/log/aortica
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF
    sudo mv /tmp/aortica-edge.service /etc/systemd/system/aortica-edge.service
fi

sudo systemctl daemon-reload
sudo systemctl enable aortica-edge.service

echo ""
echo "=== Setup complete ==="
echo "Start the service:  sudo systemctl start aortica-edge"
echo "Check status:       sudo systemctl status aortica-edge"
echo "View logs:          sudo journalctl -u aortica-edge -f"
echo ""
echo "Drop ECG files into /var/lib/aortica/data/inbox for analysis."
