# Crypto UP or DOWN

Python workspace for Binance kline data, LightGBM direction models, live signal generation, and the HiBT execution panel.

## Layout

- `scripts/` - offline data and experiment scripts.
- `live/` - live model training, Binance kline overlay, and signal generation.
- `live/hibt/` - HiBT execution client, dry-run/live runner, and control panel.
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

Run HiBT components:

```bash
.venv/bin/python live/hibt/run_hibt_signal_stack.py
.venv/bin/python live/hibt/run_hibt_api_trader.py --timeframes 3m,5m,15m
.venv/bin/python live/hibt/control_panel.py --host 127.0.0.1 --port 8765
```

See `live/hibt/README_HIBT.md` for HiBT-specific details.

## Notes

- Do not commit secrets, runtime state, live signals, model artifacts, or large data.
- `aligned_data_oos/**/1m_live.csv` is runtime overlay data, not training input.
- `live/train_live_model.py` uses fixed checked-in parameters; do not run new production Optuna searches without a separate walk-forward study.
