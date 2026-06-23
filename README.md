# Crypto UP or DOWN

Python workspace for Binance kline data, LightGBM direction models, live signal generation, and the TurboFlow execution panel.

## Layout

- `scripts/` - offline data and experiment scripts.
- `live/` - live model training, Binance kline overlay, and signal generation.
- `live/turboflow/` - default TurboFlow execution client, dry-run/live runner, and control panel.
- `live/hibt/` - retained HiBT execution client.
- `config/` - checked-in model/training configuration.
- `results/` - generated experiment outputs.

## Common Commands

Align historical klines:

```bash
.venv/bin/python scripts/align_klines.py --raw-root raw_data --out-root aligned_data
```

Refresh a live model:

```bash
.venv/bin/python live/train_live_model.py
```

Run the live signal stack:

```bash
.venv/bin/python live/run_signal_stack.py --signal-file live/signals.json
```

Run TurboFlow components:

```bash
.venv/bin/python live/turboflow/run_turboflow_signal_stack.py
.venv/bin/python live/turboflow/run_turboflow_api_trader.py --timeframes 3m,5m,15m
.venv/bin/python live/turboflow/control_panel.py --host 127.0.0.1 --port 8765
```

See `live/turboflow/README_TURBOFLOW.md` for TurboFlow details. HiBT remains documented in `live/hibt/README_HIBT.md`.
