# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python workspace for Binance crypto kline alignment and experiment workflows.

- `scripts/` contains executable workflows such as `align_klines.py`, `fetch_oos_klines.py`, and model backtest utilities.
- `config/` stores reusable model settings such as `lgbm_default_params.json`.
- `raw_data/` holds downloaded Binance zip archives and checksums. Treat it as large input data.
- `aligned_data/` and `aligned_data_oos/` contain normalized gzipped 1-minute CSV timelines.
- `cache/datasets/` stores generated parquet/json dataset caches.
- `results/` and `models/` are generated experiment outputs. Keep new result directories descriptive, e.g. `results/btc_eth_target5m_YYYYMMDD/`.

## Build, Test, and Development Commands

Use the checked-in virtual environment when available:

```bash
.venv/bin/python scripts/align_klines.py --raw-root raw_data --out-root aligned_data
.venv/bin/python scripts/fetch_oos_klines.py
.venv/bin/python scripts/lightgbm_5m_direction_btc_eth.py --data-root aligned_data --out-dir results/dev_lgbm_5m_double --min-trade-fraction 0.05
.venv/bin/python scripts/lightgbm_5m_direction_btc_eth.py --data-root aligned_data --out-dir results/dev_lgbm_5m_double_optuna --min-trade-fraction 0.05 --optuna-trials 40
```

For a fresh environment, install dependencies with `python3 -m pip install -r requirements.txt`.

## Coding Style & Naming Conventions

Write Python 3 with 4-space indentation, `pathlib.Path` for paths, and explicit `argparse` CLIs for runnable scripts. Keep constants uppercase near the top of scripts, use `snake_case` for functions and variables, and prefer structured CSV/JSON/parquet reads over ad hoc text parsing. Preserve UTC timestamp handling and millisecond units unless a conversion is clearly documented.

## Testing Guidelines

No formal test suite is present in this checkout. For data alignment changes, validate row counts, missing ranges, and `alignment_report.json`.

## Commit & Pull Request Guidelines

Git history is not readable in this environment, so use clear imperative commit subjects such as `Add OOS kline fetch workflow` or `Create BTC ETH feature pipeline`. Pull requests should describe the data range used, commands run, key metrics changed, and any new generated artifacts. Link issues when applicable and avoid committing large regenerated data unless the review explicitly requires it.

## Security & Configuration Tips

Do not hard-code secrets or API credentials. The current Binance fetch script uses public endpoints, but future authenticated access should read credentials from environment variables. Keep large raw data, caches, and experimental outputs out of normal code reviews unless they are central to the change.
