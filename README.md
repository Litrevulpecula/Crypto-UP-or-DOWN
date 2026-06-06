# Crypto UP or DOWN

Python workflows for Binance kline alignment, LightGBM walk-forward research, and live Polymarket/HiBT signal execution.

## 15m Live Polymarket Refresh

Production 15m live models use the walk-forward-proven Optuna parameters stored in:

```bash
config/lgbm_15m_walk_forward_optuna_params.json
```

That file is copied from `results/v2_btc_eth_cross_15m_20260606/optuna_best_params.json` and matches the 15m v2 walk-forward run:

- target horizon: 15 minutes
- train/validation/test sample: 15 minutes
- feature set: `v2`
- tie policy: `expected`
- threshold EMA alpha from research: `0.7`

Live refresh may refit the latest BTC/ETH models and reselect validation thresholds, but it must not run a fresh Optuna search. The searched hyperparameters are research artifacts that were validated by walk-forward. A new live-only search creates unbacktested parameters and should not be used for production.

Train or refresh the latest 15m live models:

```bash
.venv/bin/python live/train_live_model.py
```

The defaults are:

- `--out-dir live/models_15m`
- `--fixed-params-json config/lgbm_15m_walk_forward_optuna_params.json`
- `--target-horizon-minutes 15`
- `--train-sample-minutes 15`
- `--validation-sample-minutes 15`
- `--feature-set v2`
- `--optuna-trials 0`

`live/train_live_model.py` will refuse live Optuna unless `--allow-live-optuna` is explicitly passed. That flag is for research/debug only; do not use it for production refreshes without a separate walk-forward backtest.

Generate live signals for Polymarket/HiBT:

```bash
.venv/bin/python live/update_live_klines.py
.venv/bin/python live/write_lightgbm_signals.py --once
.venv/bin/python live/write_lightgbm_signals.py
```

`live/update_live_klines.py` is the live Binance data process. It uses spot/futures websocket kline streams and writes only closed 1-minute candles into per-symbol `1m_live.csv` overlay files under `aligned_data_oos`. `live/write_lightgbm_signals.py` reads the aligned gzip files plus those overlay files. Do not run `scripts/fetch_oos_klines.py` as a live loop; it is REST backfill only.

The largest live feature window is 240 minutes plus horizon/safety warmup. If the aligned gzip files are stale and the live overlay starts empty, the signal writer waits for a continuous recent websocket window instead of bridging across old data.

By default the signal writer loads both:

- `live/models_5m`
- `live/models_15m`

It reads each model directory's `live_model_metadata.json` and uses the model's own `feature_set`, so 5m can remain `v1` while 15m uses `v2`.

Run Polymarket live:

```bash
.venv/bin/python live/polymarket/run_poly_live.py --config live/polymarket/poly_config.json --dry-run
```

By default the trader auto-resolves BTC/ETH 5m/15m rolling markets from Gamma using each signal's `decision_time` / `timestamp`. It filters for the real rolling slugs (`btc-updown-5m-*`, `eth-updown-15m-*`, `btc-updown-15m-*`, `eth-updown-15m-*`) and requires `active=true`, `closed=false`, `acceptingOrders=true`, and `enableOrderBook=true`.

Use the finder to inspect the exact market and token IDs:

```bash
.venv/bin/python live/polymarket/poly_market_finder.py --symbol BTC --timeframe 5m
```

Manual token IDs are still accepted as fallbacks in `live/polymarket/poly_config.json`:

- `token_id_btc_5m_up`, `token_id_btc_5m_down`
- `token_id_eth_5m_up`, `token_id_eth_5m_down`
- `token_id_btc_15m_up`, `token_id_btc_15m_down`
- `token_id_eth_15m_up`, `token_id_eth_15m_down`

`max_price` is the fixed 80% odds cap. Polymarket execution compares the all-in buy price against that cap:

```text
all_in_price = price + fee_rate * price * (1 - price)
```

The trader reads the market `fee_rate_bps` from Polymarket and only falls back to `default_taker_fee_rate_bps` if the fee lookup fails. Do not add model confidence or any other live-only variable to this price gate.

Polymarket order sizing is fixed shares, not fixed USDC. Use `order_shares` in `live/polymarket/poly_config.json`; the default and minimum is `5.0` shares per signal.

For funded wallet routing, `private_key` is the signing wallet private key and `funder` is the address that actually holds the funds. The trader prefers `py_clob_client_v2` when installed and supports:

- `signature_type: 0` or `"eoa"` for a normal EOA wallet
- `signature_type: 1` or `"poly_proxy"` for a Polymarket proxy wallet
- `signature_type: 2` or `"gnosis_safe"` for a Gnosis Safe
- `signature_type: 3` or `"poly_1271"` for the new deposit wallet / funder flow

Legacy 5m fields (`token_id_btc_up`, `token_id_btc_down`, `token_id_eth_up`, `token_id_eth_down`) are still accepted as fallbacks.
