#!/usr/bin/env bash
set -euo pipefail

VPS="${SIGNAL_VPS:-${VPS:-poly-vps}}"
REMOTE_DIR="${SIGNAL_REMOTE_DIR:-/opt/crypto_up_or_down}"
SSH_CMD="${SIGNAL_SSH_CMD:-ssh}"
RSYNC_RSH="${SIGNAL_RSYNC_RSH:-}"
PIP_PACKAGES="${SIGNAL_PIP_PACKAGES:-lightgbm matplotlib numpy pandas scikit-learn websockets}"
SETUP=0

for arg in "$@"; do
  case "$arg" in
    --setup)
      SETUP=1
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

LIVE_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$LIVE_DIR/.." && pwd)"

echo "sync signal stack to $VPS:$REMOTE_DIR"
$SSH_CMD "$VPS" "mkdir -p '$REMOTE_DIR/live/turboflow' '$REMOTE_DIR/live/models_3m' '$REMOTE_DIR/live/models_5m' '$REMOTE_DIR/live/models_15m' '$REMOTE_DIR/aligned_data_oos'"

RSYNC_SSH_ARGS=()
if [ -n "$RSYNC_RSH" ]; then
  RSYNC_SSH_ARGS=(-e "$RSYNC_RSH")
fi

rsync -az "${RSYNC_SSH_ARGS[@]}" \
  "$LIVE_DIR/log_colors.py" \
  "$LIVE_DIR/update_live_klines.py" \
  "$LIVE_DIR/write_lightgbm_signals.py" \
  "$LIVE_DIR/lightgbm_5m_direction_btc_eth.py" \
  "$LIVE_DIR/control_panel.py" \
  "$VPS:$REMOTE_DIR/live/"

rsync -az "${RSYNC_SSH_ARGS[@]}" --delete \
  --exclude='__pycache__/' \
  --exclude='runtime/' \
  --exclude='signals.json' \
  --exclude='*.pyc' \
  "$LIVE_DIR/turboflow/" "$VPS:$REMOTE_DIR/live/turboflow/"

for timeframe in 3m 5m 15m; do
  rsync -az "${RSYNC_SSH_ARGS[@]}" --delete \
    "$LIVE_DIR/models_${timeframe}/" "$VPS:$REMOTE_DIR/live/models_${timeframe}/"
done

rsync -az --partial --info=progress2 "${RSYNC_SSH_ARGS[@]}" \
  --exclude='*/1m_live.csv' \
  "$ROOT_DIR/aligned_data_oos/" "$VPS:$REMOTE_DIR/aligned_data_oos/"

rsync -az "${RSYNC_SSH_ARGS[@]}" "$ROOT_DIR/requirements.txt" "$VPS:$REMOTE_DIR/requirements.txt"

if [ "$SETUP" -eq 1 ]; then
  echo "setup Python environment on VPS"
  $SSH_CMD "$VPS" "cd '$REMOTE_DIR' && python3 -m venv .venv && .venv/bin/python -m pip install --disable-pip-version-check -q $PIP_PACKAGES"
fi

cat <<EOF

deployed to $VPS:$REMOTE_DIR

signal process:
  cd $REMOTE_DIR && .venv/bin/python live/turboflow/run_turboflow_signal_stack.py

live trader:
  cd $REMOTE_DIR && .venv/bin/python live/turboflow/run_turboflow_api_trader.py --live --timeframes 3m,5m,15m
EOF
