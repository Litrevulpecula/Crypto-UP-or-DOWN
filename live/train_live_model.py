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
import optuna
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


def fixed_params(args: argparse.Namespace) -> dict:
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": 1,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "objective": "binary",
        "random_state": args.random_state,
        "verbosity": -1,
    }


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": trial.suggest_float("learning_rate", args.optuna_min_learning_rate, args.optuna_max_learning_rate, log=True),
        "num_leaves": trial.suggest_int("num_leaves", args.optuna_min_num_leaves, args.optuna_max_num_leaves, step=8),
        "min_child_samples": trial.suggest_int("min_child_samples", args.optuna_min_child_samples, args.optuna_max_child_samples, step=20),
        "subsample": trial.suggest_float("subsample", args.optuna_min_subsample, args.optuna_max_subsample),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", args.optuna_min_colsample_bytree, args.optuna_max_colsample_bytree),
        "reg_alpha": trial.suggest_float("reg_alpha", args.optuna_min_reg_alpha, args.optuna_max_reg_alpha, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", args.optuna_min_reg_lambda, args.optuna_max_reg_lambda, log=True),
        "objective": "binary",
        "random_state": args.random_state,
        "verbosity": -1,
    }


def split_latest(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_end = dataset.index.max()
    validation_start = validation_end - pd.Timedelta(days=args.validation_days)
    train = dataset.loc[dataset.index < validation_start]
    validation = dataset.loc[dataset.index >= validation_start]
    if args.train_window_days is not None:
        train = train.loc[train.index >= validation_start - pd.Timedelta(days=args.train_window_days)]
    train = base.sample_aligned_frame(train, args.train_sample_minutes)
    validation = base.sample_aligned_frame(validation, args.validation_sample_minutes)
    fit_all = base.sample_aligned_frame(dataset, args.train_sample_minutes)
    if train.empty or validation.empty:
        raise RuntimeError("Not enough data for latest train/validation split.")
    return train, validation, fit_all


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


def optimize_symbol(symbol: str, columns: list[str], train: pd.DataFrame, validation: pd.DataFrame, args: argparse.Namespace) -> dict:
    rt = runtime(args)
    prefix = symbol.replace("USDT", "")
    y_train = train[f"{prefix}_target"].astype(int)
    train_weight = base.train_sample_weight(train, prefix, rt)
    y_validation = validation[f"{prefix}_target"].to_numpy(np.int8)
    validation_tie = base.payoff_tie_mask(validation, prefix, rt)
    validation_weight = base.target_sample_weight(validation, prefix, rt)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            train[columns],
            y_train,
            sample_weight=train_weight,
            eval_set=[(validation[columns], y_validation.astype(int))],
            eval_sample_weight=[validation_weight],
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
        )
        prob = model.predict_proba(validation[columns])[:, 1]
        thresholds = base.choose_thresholds(prob, y_validation, rt, validation_tie)
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or params["n_estimators"]))
        trial.set_user_attr("validation_auc", base.auc_or_nan(y_validation, prob, validation_tie))
        trial.set_user_attr("validation_thresholds", thresholds)
        return thresholds["sharpe"] if args.optuna_metric == "sharpe" else base.auc_or_nan(y_validation, prob, validation_tie)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.random_state))
    study.optimize(objective, n_trials=args.optuna_trials, timeout=args.optuna_timeout, show_progress_bar=False)
    params = fixed_params(args)
    params.update(study.best_trial.params)
    params["subsample_freq"] = 1
    params["objective"] = "binary"
    params["random_state"] = args.random_state
    params["verbosity"] = -1
    return {
        "best_value": float(study.best_value),
        "best_params": params,
        "best_user_attrs": dict(study.best_trial.user_attrs),
        "trials": len(study.trials),
    }


def train_final_symbol(
    symbol: str,
    columns: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    fit_all: pd.DataFrame,
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

    final_params = dict(params)
    final_params["n_estimators"] = int(selector.best_iteration_ or params["n_estimators"])
    final_model = lgb.LGBMClassifier(**final_params)
    final_weight = base.train_sample_weight(fit_all, prefix, rt)
    final_model.fit(fit_all[columns], fit_all[f"{prefix}_target"].astype(int), sample_weight=final_weight)
    model_path = args.out_dir / "models" / f"live_{prefix}.txt"
    final_model.booster_.save_model(model_path)
    return {
        "model_path": str(model_path),
        "best_iteration": final_params["n_estimators"],
        "raw_thresholds": raw_thresholds,
        "thresholds": thresholds,
        "validation_auc": base.auc_or_nan(validation_actual, validation_prob, validation_tie),
        "validation_start": validation.index.min().isoformat(),
        "validation_end": validation.index.max().isoformat(),
        "fit_all_start": fit_all.index.min().isoformat(),
        "fit_all_end": fit_all.index.max().isoformat(),
        "fit_all_rows": int(len(fit_all)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train latest 15m live LightGBM models from fixed walk-forward Optuna params.")
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data_oos"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_15M_OUT_DIR)
    parser.add_argument("--symbols", default=",".join(base.SYMBOLS))
    parser.add_argument("--feature-set", choices=base.FEATURE_SETS, default="v2")
    parser.add_argument("--target-horizon-minutes", type=int, default=15)
    parser.add_argument("--target-tie-policy", choices=["drop", "down", "expected"], default="expected")
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--train-window-days", type=int, default=0)
    parser.add_argument("--train-time-decay-half-life-days", type=float, default=365.0)
    parser.add_argument("--train-sample-minutes", type=int, default=15)
    parser.add_argument("--validation-sample-minutes", type=int, default=15)
    parser.add_argument("--min-trade-fraction", type=float, default=0.05)
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument("--threshold-update-weight", type=float, default=0.7)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--optuna-timeout", type=float, default=None)
    parser.add_argument("--optuna-metric", choices=["sharpe", "auc"], default="sharpe")
    parser.add_argument("--fixed-params-json", type=Path, default=DEFAULT_15M_FIXED_PARAMS_JSON)
    parser.add_argument("--allow-live-optuna", action="store_true", help="Explicitly allow a fresh live Optuna search.")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--num-leaves", type=int, default=32)
    parser.add_argument("--min-child-samples", type=int, default=120)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=1e-3)
    parser.add_argument("--reg-lambda", type=float, default=1e-3)
    parser.add_argument("--early-stopping-rounds", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--optuna-min-learning-rate", type=float, default=0.01)
    parser.add_argument("--optuna-max-learning-rate", type=float, default=0.02)
    parser.add_argument("--optuna-min-num-leaves", type=int, default=16)
    parser.add_argument("--optuna-max-num-leaves", type=int, default=64)
    parser.add_argument("--optuna-min-child-samples", type=int, default=20)
    parser.add_argument("--optuna-max-child-samples", type=int, default=260)
    parser.add_argument("--optuna-min-subsample", type=float, default=0.55)
    parser.add_argument("--optuna-max-subsample", type=float, default=0.95)
    parser.add_argument("--optuna-min-colsample-bytree", type=float, default=0.55)
    parser.add_argument("--optuna-max-colsample-bytree", type=float, default=0.95)
    parser.add_argument("--optuna-min-reg-alpha", type=float, default=1e-6)
    parser.add_argument("--optuna-max-reg-alpha", type=float, default=10.0)
    parser.add_argument("--optuna-min-reg-lambda", type=float, default=1e-6)
    parser.add_argument("--optuna-max-reg-lambda", type=float, default=10.0)
    args = parser.parse_args()
    if args.target_horizon_minutes <= 0:
        raise ValueError("--target-horizon-minutes must be > 0.")
    if args.train_sample_minutes <= 0 or args.validation_sample_minutes <= 0:
        raise ValueError("sample-minute arguments must be > 0.")
    if args.optuna_trials < 0:
        raise ValueError("--optuna-trials must be >= 0.")
    if args.fixed_params_json is None:
        if args.optuna_trials <= 0:
            raise ValueError("--fixed-params-json is required for live training unless Optuna is explicitly enabled.")
        if not args.allow_live_optuna:
            raise ValueError("Live Optuna search is disabled. Pass --allow-live-optuna only for research, not production refreshes.")
    else:
        if not args.fixed_params_json.exists():
            raise FileNotFoundError(args.fixed_params_json)
        if args.optuna_trials > 0:
            print("--fixed-params-json provided; ignoring --optuna-trials for live training.", flush=True)
            args.optuna_trials = 0
    if args.train_window_days <= 0:
        args.train_window_days = None
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
    )
    columns_by_symbol = {symbol: base.feature_columns_for_symbol(dataset, symbol) for symbol in symbols}
    with (args.out_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump({"features_by_symbol": {s: {"feature_count": len(c), "features": c} for s, c in columns_by_symbol.items()}}, handle, indent=2)
        handle.write("\n")
    train, validation, fit_all = split_latest(dataset, args)
    optuna_report = {}
    live_report = {}
    fixed_payload = None
    if args.fixed_params_json is not None:
        with args.fixed_params_json.open("r", encoding="utf-8") as handle:
            fixed_payload = json.load(handle)
    for symbol in symbols:
        if fixed_payload is None:
            print(f"optuna latest {symbol}", flush=True)
            optuna_report[symbol] = optimize_symbol(symbol, columns_by_symbol[symbol], train, validation, args)
        else:
            print(f"use fixed params {symbol}", flush=True)
            fixed_record = fixed_record_for_symbol(fixed_payload, symbol)
            optuna_report[symbol] = {
                "source": str(args.fixed_params_json),
                "live_optuna_search": False,
                "source_trials": fixed_record.get("trials"),
                "best_value": fixed_record.get("best_value"),
                "best_params": fixed_record["best_params"],
                "best_user_attrs": fixed_record.get("best_user_attrs", {}),
                "trials": 0,
            }
        print(f"train final {symbol}", flush=True)
        live_report[symbol] = train_final_symbol(
            symbol,
            columns_by_symbol[symbol],
            train,
            validation,
            fit_all,
            optuna_report[symbol]["best_params"],
            args,
        )
    with (args.out_dir / "optuna_best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(optuna_report, handle, indent=2)
        handle.write("\n")
    metadata = {
        "mode": "latest_live_train_no_walk_forward",
        "data_root": str(args.data_root),
        "rows": int(len(dataset)),
        "dataset_start": dataset.index.min().isoformat(),
        "dataset_end": dataset.index.max().isoformat(),
        "symbols": list(symbols),
        "feature_set": args.feature_set,
        "target_horizon_minutes": args.target_horizon_minutes,
        "sample_minutes": dataset_sample_minutes,
        "train_sample_minutes": args.train_sample_minutes,
        "validation_sample_minutes": args.validation_sample_minutes,
        "validation_days": args.validation_days,
        "threshold_update_weight": args.threshold_update_weight,
        "train_time_decay_half_life_days": args.train_time_decay_half_life_days,
        "fixed_params_json": str(args.fixed_params_json) if args.fixed_params_json is not None else None,
        "live_optuna_search": fixed_payload is None,
        "optuna_trials": args.optuna_trials if fixed_payload is None else 0,
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
