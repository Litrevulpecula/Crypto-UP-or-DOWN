#!/usr/bin/env bash
set -euo pipefail

VPS="${HIBT_VPS:-poly-vps}"
REMOTE_DIR="${HIBT_REMOTE_DIR:-/opt/hibt}"
SSH_CMD="${HIBT_SSH_CMD:-ssh}"
RSYNC_RSH="${HIBT_RSYNC_RSH:-}"
PIP_PACKAGES="${HIBT_PIP_PACKAGES:-lightgbm matplotlib numpy pandas scikit-learn websockets}"
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

HIBT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIVE_DIR="$(cd "$HIBT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$LIVE_DIR/.." && pwd)"

echo "sync HiBT API runner to $VPS:$REMOTE_DIR"
$SSH_CMD "$VPS" "mkdir -p '$REMOTE_DIR/live/hibt' '$REMOTE_DIR/live/models_15m' '$REMOTE_DIR/aligned_data_oos'"

RSYNC_SSH_ARGS=()
if [ -n "$RSYNC_RSH" ]; then
  RSYNC_SSH_ARGS=(-e "$RSYNC_RSH")
fi

rsync -az "${RSYNC_SSH_ARGS[@]}" --delete \
  --exclude='__pycache__/' \
  --exclude='runtime/' \
  --exclude='signals.json' \
  --exclude='*.pyc' \
  "$HIBT_DIR/" "$VPS:$REMOTE_DIR/live/hibt/"

rsync -az "${RSYNC_SSH_ARGS[@]}" \
  "$LIVE_DIR/log_colors.py" \
  "$LIVE_DIR/update_live_klines.py" \
  "$LIVE_DIR/write_lightgbm_signals.py" \
  "$LIVE_DIR/lightgbm_5m_direction_btc_eth.py" \
  "$VPS:$REMOTE_DIR/live/"

rsync -az "${RSYNC_SSH_ARGS[@]}" --delete \
  "$LIVE_DIR/models_15m/" "$VPS:$REMOTE_DIR/live/models_15m/"

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
  cd $REMOTE_DIR && .venv/bin/python live/hibt/run_hibt_signal_stack.py

dry-run trader:
  cd $REMOTE_DIR && .venv/bin/python live/hibt/run_hibt_api_trader.py --once

live trader:
  cd $REMOTE_DIR && HIBT_API_V='...' HIBT_AUTHORIZATION='...' HIBT_X_AUTH_TOKEN='...' \\
    .venv/bin/python live/hibt/run_hibt_api_trader.py --live --timeframes 15m --amount 15m=3
EOF
