#!/usr/bin/env bash
# VPS one-click setup: Chrome + Python env + systemd services
# Target: Debian 13 / 2 vCPU / 4 GiB RAM (root@47.79.32.65)
set -e

DEPLOY_DIR="/opt/hibt"
VENV_DIR="$DEPLOY_DIR/.venv"
SERVICE_USER="hibt"
DATA_ROOT="${DATA_ROOT:-aligned_data_oos}"

echo "=== [1/5] System dependencies ==="
apt-get update -qq
apt-get install -y --no-install-recommends \
  wget gnupg2 ca-certificates curl \
  python3 python3-venv python3-pip \
  xvfb fonts-noto-cjk dbus

echo "=== [2/5] Install Google Chrome Stable ==="
if ! command -v google-chrome-stable &>/dev/null; then
  wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  apt-get install -y /tmp/chrome.deb || (apt-get -f install -y && apt-get install -y /tmp/chrome.deb)
  rm -f /tmp/chrome.deb
fi
echo "Chrome: $(google-chrome-stable --version)"

echo "=== [3/5] Create service user & deploy code ==="
id "$SERVICE_USER" &>/dev/null || useradd -r -m -d "$DEPLOY_DIR" -s /bin/bash "$SERVICE_USER"
mkdir -p "$DEPLOY_DIR"

echo "=== [4/5] Python venv ==="
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install -q --upgrade pip
if [ -f "$DEPLOY_DIR/requirements.txt" ]; then
  "$VENV_DIR/bin/pip" install -q -r "$DEPLOY_DIR/requirements.txt"
fi
if [ -f "$DEPLOY_DIR/live/hibt/requirements_hibt.txt" ]; then
  "$VENV_DIR/bin/pip" install -q -r "$DEPLOY_DIR/live/hibt/requirements_hibt.txt"
else
  "$VENV_DIR/bin/pip" install -q playwright
fi
"$VENV_DIR/bin/python" -m playwright install-deps 2>/dev/null || true

echo "=== [5/5] Systemd services ==="

cat > /etc/systemd/system/xvfb.service <<'EOF'
[Unit]
Description=Xvfb virtual display
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/Xvfb :99 -screen 0 1920x1080x24 -ac -nolisten tcp
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/hibt-trader.service <<EOF
[Unit]
Description=HiBT auto trader
After=network-online.target xvfb.service
Wants=network-online.target xvfb.service

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$DEPLOY_DIR
Environment=DISPLAY=:99
Environment=HOME=$DEPLOY_DIR
ExecStart=$VENV_DIR/bin/python3 live/run_hibt_stack.py --data-root $DATA_ROOT --symbols BTCUSDT,ETHUSDT --signal-model-dir 15m=live/models_15m --hibt-config live/hibt/hibt_config.vps.json
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable xvfb.service
systemctl start xvfb.service

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Copy code to VPS:"
echo "     rsync -avz --exclude='.venv' --exclude='__pycache__' \\"
echo "       ~/Crypto_UP_or_DOWN/live/ root@47.79.32.65:$DEPLOY_DIR/live/"
echo ""
echo "  2. First-time login (over SSH with X forwarding or noVNC):"
echo "     export DISPLAY=:99"
echo "     cd $DEPLOY_DIR/live/hibt"
echo "     $VENV_DIR/bin/python3 run_hibt_live.py --config hibt_config.vps.json --login"
echo "     # Manually log in, then Ctrl+C"
echo ""
echo "  3. Start the trader service:"
echo "     systemctl start hibt-trader"
echo "     journalctl -u hibt-trader -f"
echo ""
echo "  4. To run live (real trading):"
echo "     Edit live/hibt/hibt_config.vps.json: set dry_run=false, click_confirm_order=true"
echo "     systemctl restart hibt-trader"
echo ""
