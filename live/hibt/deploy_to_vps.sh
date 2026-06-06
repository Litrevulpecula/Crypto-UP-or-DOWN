#!/usr/bin/env bash
# HiBT-only deploy helper. Do not use this for the current Polymarket 15m stack.
# Deploy live/hibt to VPS and optionally run HiBT setup.
# Usage: ./deploy_to_vps.sh [--setup]
set -e

VPS="root@47.79.32.65"
REMOTE_DIR="/opt/hibt"
HIBT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIVE_DIR="$(cd "$HIBT_DIR/.." && pwd)"

echo "=== Syncing code to VPS ==="
ssh "$VPS" "mkdir -p '$REMOTE_DIR/live/hibt'"
rsync -avz --delete \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='runtime/' \
  --exclude='models/' \
  --exclude='models_5m/' \
  --exclude='models_15m/' \
  --exclude='*.pyc' \
  "$HIBT_DIR/" "$VPS:$REMOTE_DIR/live/hibt/"

echo "=== Syncing shared live modules ==="
rsync -avz \
  "$LIVE_DIR/log_colors.py" \
  "$LIVE_DIR/run_hibt_stack.py" \
  "$LIVE_DIR/update_live_klines.py" \
  "$LIVE_DIR/write_lightgbm_signals.py" \
  "$LIVE_DIR/lightgbm_5m_direction_btc_eth.py" \
  "$VPS:$REMOTE_DIR/live/"
rsync -avz "$LIVE_DIR/../requirements.txt" "$VPS:$REMOTE_DIR/requirements.txt"

echo "=== Syncing 15m models ==="
rsync -avz --delete \
  "$LIVE_DIR/models_15m/" "$VPS:$REMOTE_DIR/live/models_15m/"

if [ "$1" = "--setup" ]; then
  echo "=== Running setup on VPS ==="
  ssh "$VPS" "bash $REMOTE_DIR/live/setup_chrome_vps.sh"
fi

echo ""
echo "Done. Code deployed to $VPS:$REMOTE_DIR/live/"
echo ""
echo "Commands:"
echo "  ssh $VPS 'systemctl restart hibt-trader'   # restart HiBT stack"
echo "  ssh $VPS 'journalctl -u hibt-trader -f'    # view logs"
