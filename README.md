# Crypto UP or DOWN

Python workflows for Binance kline alignment, LightGBM walk-forward research, and live Polymarket/HiBT signal execution.

## Live Polymarket Model Refresh

Live model refresh is monthly, not minute-by-minute or daily. The training cutoff must be the latest completed month with a full target label:

- 5m cutoff: current UTC month start minus 5 minutes, e.g. `2026-05-31T23:55:00Z` when running in June 2026.
- 15m cutoff: current UTC month start minus 15 minutes, e.g. `2026-05-31T23:45:00Z` when running in June 2026.
- Do not fetch current-minute data for training refresh.
- Do not include `1m_live.csv` overlays in training.
- Do not fit the deployed model on validation rows. Validation is only for early stopping and threshold selection.
- Do not run a new live Optuna search unless a separate walk-forward backtest validates it.

Production 15m live models use the walk-forward-proven Optuna parameters stored in:

```bash
config/lgbm_15m_walk_forward_optuna_params.json
```

That file is copied from `results/v1_15m_full_decay/optuna_best_params.json` and matches the 15m v1 walk-forward run:

- target horizon: 15 minutes
- train/validation/test sample: 15 minutes
- feature set: `v1`
- tie policy: `expected`
- threshold EMA alpha from research: `0.7`

Live refresh may refit the latest BTC/ETH models at the monthly cutoff and reselect validation thresholds, but it must not run a fresh Optuna search. The searched hyperparameters are research artifacts that were validated by walk-forward. A new live-only search creates unbacktested parameters and should not be used for production.

Refresh the latest 5m live models:

```bash
.venv/bin/python live/train_live_model.py \
  --out-dir live/models_5m \
  --target-horizon-minutes 5 \
  --train-sample-minutes 5 \
  --validation-sample-minutes 5 \
  --threshold-update-weight 0.2 \
  --fixed-params-json results/v1_5m_full_decay/optuna_best_params.json \
  --feature-set v1 \
  --optuna-trials 0
```

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
- `--feature-set v1`
- `--optuna-trials 0`

`live/train_live_model.py` will refuse live Optuna unless `--allow-live-optuna` is explicitly passed. That flag is for research/debug only; do not use it for production refreshes without a separate walk-forward backtest.

Current production Polymarket deploy is 15m only. Use the dedicated deploy script:

```bash
live/deploy_polymarket_15m_vps.sh --start
```

See `live/README_POLYMARKET_15M.md` for the exact VPS sync exclusions and tmux commands.

`live/run_polymarket_stack.py` starts the Binance live kline updater with the LightGBM signal callback enabled, then starts the Polymarket trader. The stack default is `15m=live/models_15m`. Pass `--signal-model-dir` explicitly only when intentionally overriding the production model set.

On startup it warms up the live overlay by fetching the latest 360 closed 1-minute Binance candles for BTC/ETH spot and futures into `1m_live.csv`. This covers the largest live feature requirement: 240 minutes of rolling features plus 15-minute horizon/safety warmup. During live operation, websocket updates are the primary path and a REST catch-up runs every 2 seconds over the latest 15 minutes so a stalled websocket stream cannot leave the event feature row incomplete. It does not retrain models and does not rewrite the large aligned gzip files.

`live/update_live_klines.py` is the live Binance data process. It uses spot/futures websocket kline streams and writes only closed 1-minute candles into per-symbol `1m_live.csv` overlay files under `aligned_data_oos`. When `--signal-file` is passed, the updater runs the LightGBM signal callback immediately after a new or changed closed 1-minute row is written. `live/write_lightgbm_signals.py` is a single-shot signal utility plus shared callback helpers; it is not the live timing loop. Do not run `scripts/fetch_oos_klines.py` as a live loop; it is REST backfill only.

Signal generation is data-callback driven. The callback maps each closed 1-minute row to `decision_time = open_time + 1 minute`; if that decision time is a configured Polymarket event start and all required BTC/ETH spot/futures overlay rows are present, it runs the due model(s) once and writes signals for that exact event window. The same in-process callback will not write the same timeframe/decision time twice.

Current production stack loads only:

- `live/models_15m`

`live/write_lightgbm_signals.py` still supports multiple `--model-dir timeframe=path` entries for research/debug. Do not omit the 15m-only model selection in production deploy scripts.

Run the Polymarket trader directly only for isolated debugging:

```bash
.venv/bin/python live/polymarket/run_poly_live.py --config live/polymarket/poly_config.json --dry-run
```

The trader auto-resolves BTC/ETH rolling markets from Gamma using each signal's `decision_time` / `timestamp`. The 15m production signal writer emits only 15m signals; legacy 5m token fields remain accepted as fallbacks for non-production runs.

Use the finder to inspect the exact market and token IDs:

```bash
.venv/bin/python live/polymarket/poly_market_finder.py --symbol BTC --timeframe 15m
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
