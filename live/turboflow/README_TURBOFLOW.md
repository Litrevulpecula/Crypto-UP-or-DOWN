# TurboFlow Runner

Default live path for prediction-market execution. HiBT remains in `live/hibt/`.

```bash
.venv/bin/python live/turboflow/run_turboflow_signal_stack.py
.venv/bin/python live/turboflow/run_turboflow_api_trader.py --timeframes 3m,5m,15m
.venv/bin/python live/turboflow/control_panel.py --host 127.0.0.1 --port 8765
```

Live order env:

```bash
export TURBOFLOW_TOKEN='...'
export TURBOFLOW_UID='...'
export TURBOFLOW_COIN_CODE='...'
export TURBOFLOW_POOL_ID='...'
.venv/bin/python live/turboflow/run_turboflow_api_trader.py --live --timeframes 3m,5m,15m
```

The runner fetches TurboFlow prediction config from `/public/pm/config?version=2` before each order and submits to `/account/pm/order/submit`.
