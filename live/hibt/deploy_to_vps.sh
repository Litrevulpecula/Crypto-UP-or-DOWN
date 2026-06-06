#!/usr/bin/env bash
# Deploy live/ to VPS and optionally run setup
# Usage: ./deploy_to_vps.sh [--setup]
set -e

VPS="root@47.79.32.65"
REMOTE_DIR="/opt/hibt"

echo "=== Syncing code to VPS ==="
rsync -avz --delete \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='runtime/' \
  --exclude='models/' \
  --exclude='models_5m/' \
  --exclude='*.pyc' \
  "$(dirname "$0")/" "$VPS:$REMOTE_DIR/live/"

echo "=== Syncing models ==="
rsync -avz --delete \
  "$(dirname "$0")/models_5m/" "$VPS:$REMOTE_DIR/live/models_5m/" 2>/dev/null || true

if [ "$1" = "--setup" ]; then
  echo "=== Running setup on VPS ==="
  ssh "$VPS" "bash $REMOTE_DIR/live/setup_chrome_vps.sh"
fi

echo ""
echo "Done. Code deployed to $VPS:$REMOTE_DIR/live/"
echo ""
echo "Commands:"
echo "  ssh $VPS 'systemctl restart hibt-trader'   # restart trader"
echo "  ssh $VPS 'journalctl -u hibt-trader -f'    # view logs"
