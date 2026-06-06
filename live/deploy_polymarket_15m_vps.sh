#!/usr/bin/env bash
set -euo pipefail

VPS="${VPS:-poly-vps}"
REMOTE_DIR="${REMOTE_DIR:-/root/Crypto_UP_or_DOWN}"
SESSION="${SESSION:-poly15}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

SSH_CMD=(ssh -o LogLevel=ERROR)
RSYNC_SSH=(ssh -o LogLevel=ERROR)
if [[ -n "${SSHPASS:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "SSHPASS is set but sshpass is not installed" >&2
    exit 1
  fi
  SSH_CMD=(sshpass -e ssh -o LogLevel=ERROR)
  RSYNC_SSH=(sshpass -e ssh -o LogLevel=ERROR)
fi

echo "== Sync Polymarket 15m live runtime =="
"${SSH_CMD[@]}" "$VPS" "mkdir -p '$REMOTE_DIR/live/runtime'"

rsync -az -e "${RSYNC_SSH[*]}" --delete \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='runtime/' \
  --exclude='signals.json' \
  --exclude='models/' \
  --exclude='models_5m/' \
  live/ "$VPS:$REMOTE_DIR/live/"

rsync -az -e "${RSYNC_SSH[*]}" requirements.txt "$VPS:$REMOTE_DIR/requirements.txt"

echo "== Remove stale non-15m model dirs from remote =="
"${SSH_CMD[@]}" "$VPS" "rm -rf '$REMOTE_DIR/live/models' '$REMOTE_DIR/live/models_5m'; find '$REMOTE_DIR/live' -type d -name __pycache__ -prune -exec rm -rf {} +"

echo "== Install Python dependencies =="
"${SSH_CMD[@]}" "$VPS" "cd '$REMOTE_DIR' && python3 -m venv .venv && .venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt"

echo "== Clear runtime signal file =="
"${SSH_CMD[@]}" "$VPS" "cd '$REMOTE_DIR' && .venv/bin/python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

path = Path('live/signals.json')
payload = {
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'source': 'deploy_clear',
    'signals': [],
    'diagnostics': [],
}
path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
PY"

if [[ "${1:-}" == "--start" ]]; then
  echo "== Restart tmux session $SESSION =="
  "${SSH_CMD[@]}" "$VPS" "tmux has-session -t '$SESSION' 2>/dev/null && tmux kill-session -t '$SESSION' || true"
  "${SSH_CMD[@]}" "$VPS" "pkill -f '$REMOTE_DIR/live/[u]pdate_live_klines.py' || true; pkill -f '$REMOTE_DIR/live/polymarket/[r]un_poly_live.py' || true; pkill -f 'live/[r]un_polymarket_stack.py' || true"
  "${SSH_CMD[@]}" "$VPS" "cd '$REMOTE_DIR' && tmux new-session -d -s '$SESSION' \"PYTHONUNBUFFERED=1 .venv/bin/python live/run_polymarket_stack.py --data-root aligned_data_oos --symbols BTCUSDT,ETHUSDT --signal-model-dir 15m=live/models_15m --rest-backfill-minutes 360 --rest-catchup-minutes 15 --rest-catchup-seconds 2.0\""
  "${SSH_CMD[@]}" "$VPS" "tmux pipe-pane -o -t '$SESSION' \"cat >> '$REMOTE_DIR/live/runtime/${SESSION}_tmux.log'\" && tmux ls"
else
  echo "Deploy finished. Start with:"
  echo "  VPS=$VPS REMOTE_DIR=$REMOTE_DIR SESSION=$SESSION live/deploy_polymarket_15m_vps.sh --start"
fi
