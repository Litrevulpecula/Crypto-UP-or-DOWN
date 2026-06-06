# Polymarket 15m Live Deploy

Current production path is Polymarket 15m only.

Use only the minimal runtime on the VPS:

- `live/`
- `requirements.txt`

Do not deploy these runtime or legacy paths:

- `aligned_data_oos/`
- `live/signals.json`
- `live/runtime/`
- `live/__pycache__/`
- `live/models/`
- `live/models_5m/`

Deploy and restart the VPS tmux session:

```bash
live/deploy_polymarket_15m_vps.sh --start
```

Defaults:

- VPS SSH alias: `poly-vps`
- remote directory: `/root/Crypto_UP_or_DOWN`
- tmux session: `poly15`
- model directory: `15m=live/models_15m`

Override defaults when needed:

```bash
VPS=root@47.79.32.65 REMOTE_DIR=/root/Crypto_UP_or_DOWN SESSION=poly15 \
  live/deploy_polymarket_15m_vps.sh --start
```

Manual run command inside the deployed directory:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python live/run_polymarket_stack.py \
  --data-root aligned_data_oos \
  --symbols BTCUSDT,ETHUSDT \
  --signal-model-dir 15m=live/models_15m \
  --rest-backfill-minutes 360 \
  --rest-catchup-minutes 15 \
  --rest-catchup-seconds 2.0
```

Operational commands:

```bash
ssh poly-vps
tmux attach -t poly15
tail -f /root/Crypto_UP_or_DOWN/live/runtime/poly15_tmux.log
```

Before each start, the deploy script clears `live/signals.json` on the VPS so the trader cannot read an old signal from a previous run.
