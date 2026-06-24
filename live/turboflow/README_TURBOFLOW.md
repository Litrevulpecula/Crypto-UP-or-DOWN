# TurboFlow Runner

Default live path for prediction-market execution. HiBT remains in `live/hibt/`.

```bash
.venv/bin/python live/turboflow/run_turboflow_signal_stack.py
.venv/bin/python live/turboflow/run_turboflow_api_trader.py --timeframes 3m,5m,15m
.venv/bin/python live/control_panel.py --venue turboflow --host 127.0.0.1 --port 8765
```

Live order env:

```bash
export TURBOFLOW_TOKEN='...'
export TURBOFLOW_UID='...'
export TURBOFLOW_COIN_CODE='...'
export TURBOFLOW_POOL_ID='...'
.venv/bin/python live/turboflow/run_turboflow_api_trader.py --live --timeframes 3m,5m,15m
```

Live trading reads the TurboFlow USDT balance from `GET /account/assets/v2?fill_coin_sub_info=yes`; the control-panel bankroll is only the fallback. BTC/ETH 3m/5m/15m use the configured 1/4 Kelly fractions with a `$2.00` minimum.
