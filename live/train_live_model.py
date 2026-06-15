#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = Path(__file__).resolve().parent
if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))

import lightgbm_5m_direction_btc_eth as base  # noqa: E402


DEFAULT_15M_FIXED_PARAMS_JSON = ROOT / "config" / "lgbm_15m_walk_forward_optuna_params.json"
DEFAULT_15M_OUT_DIR = LIVE_DIR / "models_15m"


def runtime(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        out_dir=args.out_dir,
        win_payoff=base.PAYOFF_WIN,
        loss_payoff=base.PAYOFF_LOSS,
        min_trade_fraction=args.min_trade_fraction,
        threshold_grid_size=args.threshold_grid_size,
        threshold_ema_alpha=args.threshold_update_weight,
        target_tie_policy=args.target_tie_policy,
        train_time_decay_half_life_days=args.train_time_decay_half_life_days,
        train_sample_minutes=args.train_sample_minutes,
        validation_sample_minutes=args.validation_sample_minutes,
        target_horizon_minutes=args.target_horizon_minutes,
    )


def split_latest(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_end = dataset.index.max()
    validation_start = validation_end - pd.Timedelta(days=args.validation_days)
    train = dataset.loc[dataset.index < validation_start]
    validation = dataset.loc[dataset.index >= validation_start]
    if args.train_window_days is not None:
        train = train.loc[train.index >= validation_start - pd.Timedelta(days=args.train_window_days)]
    train = base.sample_aligned_frame(train, args.train_sample_minutes)
    validation = base.sample_aligned_frame(validation, args.validation_sample_minutes)
    if train.empty or validation.empty:
        raise RuntimeError("Not enough data for latest train/validation split.")
    return train, validation


def parse_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def latest_completed_month_cutoff(target_horizon_minutes: int) -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    current_month_start = pd.Timestamp(year=now.year, month=now.month, day=1, tz="UTC")
    return current_month_start - pd.Timedelta(minutes=target_horizon_minutes)


def fixed_record_for_symbol(fixed_payload: dict, symbol: str) -> dict:
    payload = fixed_payload.get("params_by_symbol", fixed_payload)
    if symbol not in payload:
        raise KeyError(f"Fixed parameter file has no entry for {symbol}.")
    record = payload[symbol]
    if "metadata" in record and "best_params" in record["metadata"]:
        record = record["metadata"]
    if "best_params" not in record:
        record = {"best_params": record}
    return record


def train_final_symbol(
    symbol: str,
    columns: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: dict,
    args: argparse.Namespace,
) -> dict:
    rt = runtime(args)
    prefix = symbol.replace("USDT", "")
    validation_tie = base.payoff_tie_mask(validation, prefix, rt)
    validation_weight = base.target_sample_weight(validation, prefix, rt)
    train_weight = base.train_sample_weight(train, prefix, rt)
    selector = lgb.LGBMClassifier(**params)
    selector.fit(
        train[columns],
        train[f"{prefix}_target"].astype(int),
        sample_weight=train_weight,
        eval_set=[(validation[columns], validation[f"{prefix}_target"].astype(int))],
        eval_sample_weight=[validation_weight],
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
    )
    validation_prob = selector.predict_proba(validation[columns])[:, 1]
    validation_actual = validation[f"{prefix}_target"].to_numpy(np.int8)
    raw_thresholds = base.choose_thresholds(validation_prob, validation_actual, rt, validation_tie)
    thresholds = base.adjust_extreme_thresholds(raw_thresholds, validation_prob)

    # Deploy the selector itself: it is trained only on `train` (validation is out-of-sample),
    # so the deployed model and the thresholds picked on its validation predictions share the
    # same provenance. Retraining on validation here would let the model memorize the exact rows
    # used to calibrate thresholds, biasing live behavior. This mirrors the walk-forward flow.
    model_path = args.out_dir / "models" / f"live_{prefix}.txt"
    selector.booster_.save_model(model_path)
    return {
        "model_path": str(model_path),
        "best_iteration": int(selector.best_iteration_ or params["n_estimators"]),
        "raw_thresholds": raw_thresholds,
        "thresholds": thresholds,
        "validation_auc": base.auc_or_nan(validation_actual, validation_prob, validation_tie),
        "validation_start": validation.index.min().isoformat(),
        "validation_end": validation.index.max().isoformat(),
        "train_start": train.index.min().isoformat(),
        "train_end": train.index.max().isoformat(),
        "train_rows": int(len(train)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train latest monthly-cutoff live LightGBM models from fixed walk-forward Optuna params."
    )
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data_oos"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_15M_OUT_DIR)
    parser.add_argument("--symbols", default=",".join(base.SYMBOLS))
    parser.add_argument("--feature-set", choices=base.FEATURE_SETS, default="v1")
    parser.add_argument("--target-horizon-minutes", type=int, default=15)
    parser.add_argument("--target-tie-policy", choices=["drop", "down", "expected"], default="expected")
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--train-window-days", type=int, default=0)
    parser.add_argument("--train-time-decay-half-life-days", type=float, default=365.0)
    parser.add_argument(
        "--dataset-end",
        help=(
            "UTC max decision_time for training. Defaults to latest completed monthly cutoff "
            "(current month start minus target horizon)."
        ),
    )
    parser.add_argument("--no-monthly-cutoff", action="store_true", help="Use all available labeled rows instead.")
    parser.add_argument("--include-live-overlay", action="store_true", help="Include 1m_live.csv overlays in training data.")
    parser.add_argument("--train-sample-minutes", type=int, default=15)
    parser.add_argument("--validation-sample-minutes", type=int, default=15)
    parser.add_argument("--min-trade-fraction", type=float, default=0.05)
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument("--threshold-update-weight", type=float, default=0.7)
    parser.add_argument("--fixed-params-json", type=Path, default=DEFAULT_15M_FIXED_PARAMS_JSON)
    parser.add_argument("--early-stopping-rounds", type=int, default=250)
    args = parser.parse_args()
    if args.target_horizon_minutes <= 0:
        raise ValueError("--target-horizon-minutes must be > 0.")
    if args.train_sample_minutes <= 0 or args.validation_sample_minutes <= 0:
        raise ValueError("sample-minute arguments must be > 0.")
    if not args.fixed_params_json.exists():
        raise FileNotFoundError(args.fixed_params_json)
    if args.train_window_days <= 0:
        args.train_window_days = None
    dataset_end = parse_utc_timestamp(args.dataset_end)
    dataset_end_policy = "explicit"
    if dataset_end is None:
        if args.no_monthly_cutoff:
            dataset_end_policy = "all_available_labeled_rows"
        else:
            dataset_end = latest_completed_month_cutoff(args.target_horizon_minutes)
            dataset_end_policy = "latest_completed_month"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "models").mkdir(parents=True, exist_ok=True)
    symbols = base.parse_symbols(args.symbols)
    dataset_sample_minutes = math.gcd(args.train_sample_minutes, args.validation_sample_minutes)
    dataset = base.build_dataset(
        data_root=args.data_root,
        symbols=symbols,
        target_symbols=symbols,
        market=base.SPOT_MARKET,
        futures_market=base.FUTURES_MARKET,
        dataset_start=None,
        target_horizon_minutes=args.target_horizon_minutes,
        sample_minutes=dataset_sample_minutes,
        target_tie_policy=args.target_tie_policy,
        feature_set=args.feature_set,
        include_live_overlay=args.include_live_overlay,
    )
    if dataset_end is not None:
        dataset = dataset.loc[dataset.index <= dataset_end]
        if dataset.empty:
            raise RuntimeError(f"No labeled dataset rows at or before dataset_end={dataset_end.isoformat()}.")
    columns_by_symbol = {symbol: base.feature_columns_for_symbol(dataset, symbol) for symbol in symbols}
    with (args.out_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump({"features_by_symbol": {s: {"feature_count": len(c), "features": c} for s, c in columns_by_symbol.items()}}, handle, indent=2)
        handle.write("\n")
    train, validation = split_latest(dataset, args)
    params_report = {}
    live_report = {}
    with args.fixed_params_json.open("r", encoding="utf-8") as handle:
        fixed_payload = json.load(handle)
    for symbol in symbols:
        print(f"use fixed params {symbol}", flush=True)
        fixed_record = fixed_record_for_symbol(fixed_payload, symbol)
        params_report[symbol] = {
            "source": str(args.fixed_params_json),
            "source_trials": fixed_record.get("trials"),
            "best_value": fixed_record.get("best_value"),
            "best_params": fixed_record["best_params"],
            "best_user_attrs": fixed_record.get("best_user_attrs", {}),
        }
        print(f"train final {symbol}", flush=True)
        live_report[symbol] = train_final_symbol(
            symbol,
            columns_by_symbol[symbol],
            train,
            validation,
            params_report[symbol]["best_params"],
            args,
        )
    with (args.out_dir / "optuna_best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(params_report, handle, indent=2)
        handle.write("\n")
    metadata = {
        "mode": "latest_live_train_no_walk_forward",
        "data_root": str(args.data_root),
        "rows": int(len(dataset)),
        "dataset_start": dataset.index.min().isoformat(),
        "dataset_end": dataset.index.max().isoformat(),
        "dataset_end_policy": dataset_end_policy,
        "dataset_end_requested": None if dataset_end is None else dataset_end.isoformat(),
        "include_live_overlay": bool(args.include_live_overlay),
        "symbols": list(symbols),
        "feature_set": args.feature_set,
        "target_horizon_minutes": args.target_horizon_minutes,
        "sample_minutes": dataset_sample_minutes,
        "train_sample_minutes": args.train_sample_minutes,
        "validation_sample_minutes": args.validation_sample_minutes,
        "validation_days": args.validation_days,
        "threshold_update_weight": args.threshold_update_weight,
        "train_time_decay_half_life_days": args.train_time_decay_half_life_days,
        "fixed_params_json": str(args.fixed_params_json),
        "params_policy": (
            "production live refresh uses fixed Optuna parameters proven by walk-forward; "
            "do not run fresh live Optuna without a separate backtest"
        ),
        "live_report": live_report,
    }
    with (args.out_dir / "live_model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
