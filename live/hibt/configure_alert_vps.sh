#!/usr/bin/env bash
set -euo pipefail

VPS="${HIBT_VPS:-poly-vps}"
REMOTE_DIR="${HIBT_REMOTE_DIR:-/opt/hibt}"
ALERT_TO="${HIBT_ALERT_EMAIL_TO:-}"
ALERT_FROM="${HIBT_ALERT_EMAIL_FROM:-}"
SMTP_USER="${HIBT_ALERT_SMTP_USER:-$ALERT_FROM}"
SMTP_HOST="${HIBT_ALERT_SMTP_HOST:-smtp.gmail.com}"
SMTP_PORT="${HIBT_ALERT_SMTP_PORT:-587}"
SMTP_TLS="${HIBT_ALERT_SMTP_TLS:-1}"
SMTP_SSL="${HIBT_ALERT_SMTP_SSL:-0}"
SMTP_PASSWORD="${HIBT_ALERT_SMTP_PASSWORD:-}"
SKIP_TEST=0

for arg in "$@"; do
  case "$arg" in
    --skip-test)
      SKIP_TEST=1
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [ -z "${SSHPASS:-}" ]; then
  echo "SSHPASS is required" >&2
  exit 2
fi
if [ -z "$SMTP_PASSWORD" ]; then
  echo "HIBT_ALERT_SMTP_PASSWORD is required" >&2
  exit 2
fi
if [ -z "$ALERT_TO" ]; then
  echo "HIBT_ALERT_EMAIL_TO is required" >&2
  exit 2
fi
if [ -z "$ALERT_FROM" ]; then
  echo "HIBT_ALERT_EMAIL_FROM is required" >&2
  exit 2
fi
if [ -z "$SMTP_USER" ]; then
  echo "HIBT_ALERT_SMTP_USER is required" >&2
  exit 2
fi

SSH=(
  sshpass -e ssh
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
)

ENV_B64="$({
  printf 'export HIBT_ALERT_EMAIL_TO=%q\n' "$ALERT_TO"
  printf 'export HIBT_ALERT_EMAIL_FROM=%q\n' "$ALERT_FROM"
  printf 'export HIBT_ALERT_SMTP_HOST=%q\n' "$SMTP_HOST"
  printf 'export HIBT_ALERT_SMTP_PORT=%q\n' "$SMTP_PORT"
  printf 'export HIBT_ALERT_SMTP_TLS=%q\n' "$SMTP_TLS"
  printf 'export HIBT_ALERT_SMTP_SSL=%q\n' "$SMTP_SSL"
  printf 'export HIBT_ALERT_SMTP_USER=%q\n' "$SMTP_USER"
  printf 'export HIBT_ALERT_SMTP_PASSWORD=%q\n' "$SMTP_PASSWORD"
} | base64 -w0)"

"${SSH[@]}" "$VPS" "cd '$REMOTE_DIR' && tmp=live/hibt/runtime/hibt_env.tmp && grep -v '^export HIBT_ALERT_' live/hibt/runtime/hibt_env.sh > \"\$tmp\" && printf '%s' '$ENV_B64' | base64 -d >> \"\$tmp\" && mv \"\$tmp\" live/hibt/runtime/hibt_env.sh && chmod 600 live/hibt/runtime/hibt_env.sh"
echo "wrote alert env to $VPS:$REMOTE_DIR/live/hibt/runtime/hibt_env.sh"

if [ "$SKIP_TEST" -eq 0 ]; then
  "${SSH[@]}" "$VPS" "cd '$REMOTE_DIR' && . live/hibt/runtime/hibt_env.sh && PYTHONPATH=live/hibt .venv/bin/python -c 'from run_hibt_api_trader import AlertManager; AlertManager.from_env(0).send(\"hibt-alert-test\", \"HiBT alert test\", \"HiBT token-expiry email alert is configured.\")'"
fi

"${SSH[@]}" "$VPS" "cd '$REMOTE_DIR' && tmux kill-session -t hibt3_trader 2>/dev/null || true"
"${SSH[@]}" "$VPS" "cd '$REMOTE_DIR' && tmux new -ds hibt15_trader \". live/hibt/runtime/hibt_env.sh && .venv/bin/python live/hibt/run_hibt_api_trader.py --live --timeframes 15m --amount 15m=3\""
"${SSH[@]}" "$VPS" "cd '$REMOTE_DIR' && sleep 2 && tmux capture-pane -p -t hibt15_trader -S -20"
