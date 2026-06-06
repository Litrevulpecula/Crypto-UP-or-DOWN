#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

import lightgbm_5m_direction_btc_eth as base


def parse_symbols(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item).upper() for item in value)
    return base.parse_symbols(str(value))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def timestamp(value: object) -> pd.Timestamp:
    item = pd.Timestamp(value)
    if item.tzinfo is None:
        return item.tz_localize("UTC")
    return item.tz_convert("UTC")


def resolve_model_path(source_dir: Path, value: object, fold_id: int, prefix: str) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    candidate = source_dir / path
    if candidate.exists():
        return candidate
    candidate = source_dir / "models" / f"fold_{fold_id:03d}_{prefix}.txt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path)


def load_feature_columns(source_dir: Path, dataset: pd.DataFrame, symbols: tuple[str, ...]) -> dict[str, list[str]]:
    path = source_dir / "feature_columns.json"
    if not path.exists():
        return {symbol: base.feature_columns_for_symbol(dataset, symbol) for symbol in symbols}
    payload = read_json(path)
    result = {}
    available = set(dataset.columns)
    for symbol in symbols:
        columns = payload["features_by_symbol"][symbol]["features"]
        missing = [column for column in columns if column not in available]
        if missing:
            raise ValueError(f"{symbol} has missing feature columns in rebuilt dataset: {missing[:10]}")
        result[symbol] = columns
    return result


def replay(source_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict:
    source_info = read_json(source_dir / "dataset_info.json")
    source_folds = pd.read_csv(source_dir / "folds.csv")
    symbols = parse_symbols(source_info.get("symbols", base.SYMBOLS))
    target_symbols = parse_symbols(source_info.get("target_symbols", list(symbols)))
    target_tie_policy = str(source_info.get("target_tie_policy", "down"))
    feature_set = str(source_info.get("feature_set", "v1"))
    data_root = args.data_root or Path(source_info.get("data_root", "aligned_data"))
    market = source_info.get("market", base.SPOT_MARKET)
    futures_market = source_info.get("futures_market", base.FUTURES_MARKET)

    dataset = base.build_dataset(
        data_root=data_root,
        symbols=symbols,
        target_symbols=target_symbols,
        market=market,
        futures_market=futures_market,
        dataset_start=None,
        target_horizon_minutes=base.TARGET_HORIZON_MINUTES,
        sample_minutes=base.SAMPLE_MINUTES,
        target_tie_policy=target_tie_policy,
        feature_set=feature_set,
    )
    columns_by_symbol = load_feature_columns(source_dir, dataset, target_symbols)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (source_dir / "feature_columns.json").exists():
        shutil.copy2(source_dir / "feature_columns.json", out_dir / "feature_columns.json")

    runtime = SimpleNamespace(
        out_dir=out_dir,
        win_payoff=float(source_info.get("win_payoff", base.PAYOFF_WIN)),
        loss_payoff=float(source_info.get("loss_payoff", base.PAYOFF_LOSS)),
        min_trade_fraction=args.min_trade_fraction,
        threshold_grid_size=args.threshold_grid_size,
        threshold_ema_alpha=args.threshold_ema_alpha,
        target_tie_policy=target_tie_policy,
        feature_set=feature_set,
    )

    predictions = []
    folds = []
    previous_thresholds_by_symbol: dict[str, dict] = {}

    for _, source_fold in source_folds.iterrows():
        fold_id = int(source_fold["fold"])
        validation = dataset.loc[
            (dataset.index >= timestamp(source_fold["validation_start"]))
            & (dataset.index <= timestamp(source_fold["validation_end"]))
        ]
        test = dataset.loc[
            (dataset.index >= timestamp(source_fold["test_start"]))
            & (dataset.index <= timestamp(source_fold["test_end"]))
        ]
        if validation.empty or test.empty:
            raise RuntimeError(f"Empty validation/test split for fold {fold_id}.")

        fold_predictions = pd.DataFrame(index=test.index)
        fold_predictions["fold"] = fold_id
        fold_predictions["last_kline_time"] = test["last_kline_time"].to_numpy()
        fold_record = {
            "fold": fold_id,
            "validation_start": validation.index.min().isoformat(),
            "validation_end": validation.index.max().isoformat(),
            "test_start": test.index.min().isoformat(),
            "test_end": test.index.max().isoformat(),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        }
        status_parts = []

        for symbol in target_symbols:
            prefix = symbol.replace("USDT", "")
            columns = columns_by_symbol[symbol]
            model_path = resolve_model_path(source_dir, source_fold[f"{prefix}_model_path"], fold_id, prefix)
            booster = lgb.Booster(model_file=str(model_path))

            validation_probability = booster.predict(validation[columns])
            validation_actual = validation[f"{prefix}_target"].to_numpy(dtype=np.int8)
            validation_tie = base.payoff_tie_mask(validation, prefix, runtime)
            raw_thresholds = base.choose_thresholds(validation_probability, validation_actual, runtime, validation_tie)
            adjusted_thresholds = base.adjust_extreme_thresholds(raw_thresholds, validation_probability)
            thresholds = base.smooth_thresholds(
                adjusted_thresholds,
                previous_thresholds_by_symbol.get(symbol),
                args.threshold_ema_alpha,
            )
            previous_thresholds_by_symbol[symbol] = thresholds

            probability = booster.predict(test[columns])
            actual = test[f"{prefix}_target"].to_numpy(dtype=np.int8)
            test_tie = base.payoff_tie_mask(test, prefix, runtime)
            actual_test_tie = base.target_tie_mask(test, prefix)
            position, active, wins, pnl = base.apply_thresholds(probability, actual, thresholds, runtime, test_tie)
            prediction = (position == 1).astype(np.int8)

            fold_predictions[f"{prefix}_prob_up"] = probability
            fold_predictions[f"{prefix}_pred_up"] = prediction
            fold_predictions[f"{prefix}_position"] = position
            fold_predictions[f"{prefix}_is_trade"] = active.astype(np.int8)
            fold_predictions[f"{prefix}_is_win"] = wins.astype(np.int8)
            fold_predictions[f"{prefix}_is_tie"] = actual_test_tie.astype(np.int8)
            fold_predictions[f"{prefix}_actual_up"] = actual
            fold_predictions[f"{prefix}_pnl"] = pnl
            fold_predictions[f"{prefix}_threshold"] = thresholds["used_threshold"]
            fold_predictions[f"{prefix}_short_threshold"] = thresholds["used_short_threshold"]
            fold_predictions[f"{prefix}_long_threshold"] = thresholds["used_long_threshold"]
            fold_predictions[f"{prefix}_raw_short_threshold"] = thresholds["raw_short_threshold"]
            fold_predictions[f"{prefix}_raw_long_threshold"] = thresholds["raw_long_threshold"]
            fold_predictions[f"{prefix}_adjusted_short_threshold"] = thresholds["short_threshold"]
            fold_predictions[f"{prefix}_adjusted_long_threshold"] = thresholds["long_threshold"]
            fold_predictions[f"{prefix}_spot_close"] = test[f"{prefix}_spot_close"].to_numpy()
            fold_predictions[f"{prefix}_future_close"] = test[f"{prefix}_future_close"].to_numpy()

            fold_record[f"{prefix}_auc"] = base.auc_or_nan(actual, probability, test_tie)
            fold_record[f"{prefix}_threshold_mode"] = thresholds["mode"]
            fold_record[f"{prefix}_raw_short_threshold"] = thresholds["raw_short_threshold"]
            fold_record[f"{prefix}_raw_long_threshold"] = thresholds["raw_long_threshold"]
            fold_record[f"{prefix}_short_threshold_was_extreme"] = thresholds["short_threshold_was_extreme"]
            fold_record[f"{prefix}_long_threshold_was_extreme"] = thresholds["long_threshold_was_extreme"]
            fold_record[f"{prefix}_adjusted_short_threshold"] = thresholds["short_threshold"]
            fold_record[f"{prefix}_adjusted_long_threshold"] = thresholds["long_threshold"]
            fold_record[f"{prefix}_short_threshold"] = thresholds["used_short_threshold"]
            fold_record[f"{prefix}_long_threshold"] = thresholds["used_long_threshold"]
            fold_record[f"{prefix}_threshold_ema_alpha"] = thresholds["threshold_ema_alpha"]
            fold_record[f"{prefix}_validation_probability_min"] = thresholds["validation_probability_min"]
            fold_record[f"{prefix}_validation_probability_max"] = thresholds["validation_probability_max"]
            fold_record[f"{prefix}_validation_threshold_sharpe"] = thresholds["sharpe"]
            fold_record[f"{prefix}_validation_threshold_pnl"] = thresholds["pnl"]
            fold_record[f"{prefix}_validation_threshold_trades"] = thresholds["trades"]
            fold_record[f"{prefix}_validation_threshold_trade_fraction"] = thresholds["trade_fraction"]
            fold_record[f"{prefix}_validation_threshold_win_rate"] = thresholds["win_rate"]
            fold_record[f"{prefix}_validation_threshold_tie_trades"] = thresholds["tie_trades"]
            fold_record[f"{prefix}_validation_threshold_tie_trade_fraction"] = thresholds["tie_trade_fraction"]
            fold_record[f"{prefix}_test_trades"] = int(active.sum())
            fold_record[f"{prefix}_test_trade_fraction"] = float(active.mean())
            fold_record[f"{prefix}_test_tie_trades"] = int((active & test_tie).sum())
            fold_record[f"{prefix}_test_tie_trade_fraction"] = float((active & test_tie).mean())
            strict_active = active & ~test_tie
            fold_record[f"{prefix}_win_rate"] = float(wins[strict_active].mean()) if strict_active.any() else float("nan")
            fold_record[f"{prefix}_pnl"] = float(pnl.sum())
            fold_record[f"{prefix}_test_sharpe"] = base.sharpe_ratio(pnl, base.SAMPLE_MINUTES)
            fold_record[f"{prefix}_model_path"] = str(model_path)
            status_parts.append(
                f"{prefix}_thr=({thresholds['used_short_threshold']:.2f},{thresholds['used_long_threshold']:.2f}) "
                f"raw=({thresholds['raw_short_threshold']:.2f},{thresholds['raw_long_threshold']:.2f}) "
                f"trades={int(active.sum())} pnl={float(pnl.sum()):.1f}"
            )

        predictions.append(fold_predictions.reset_index(names="decision_time"))
        folds.append(fold_record)
        print(
            f"fold={fold_id} test={fold_record['test_start']}..{fold_record['test_end']} "
            + " ".join(status_parts),
            flush=True,
        )

    pd.DataFrame(folds).to_csv(out_dir / "folds.csv", index=False)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    summary = base.summarize(prediction_frame, runtime, target_symbols)
    replay_info = {
        "source_dir": str(source_dir),
        "data_root": str(data_root),
        "symbols": list(symbols),
        "target_symbols": list(target_symbols),
        "target_tie_policy": target_tie_policy,
        "feature_set": feature_set,
        "threshold_grid_size": args.threshold_grid_size,
        "min_trade_fraction": args.min_trade_fraction,
        "threshold_ema_alpha": args.threshold_ema_alpha,
        "threshold_smoothing": f"ema old_weight={1.0 - args.threshold_ema_alpha:.6g}, new_weight={args.threshold_ema_alpha:.6g}",
        "threshold_extreme_adjustment": "short raw 0 uses validation probability min; long raw 1 uses validation probability max before EMA smoothing",
    }
    with (out_dir / "replay_info.json").open("w", encoding="utf-8") as handle:
        json.dump(replay_info, handle, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved LightGBM models with fresh threshold logic.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument("--min-trade-fraction", type=float, default=0.05)
    parser.add_argument("--threshold-ema-alpha", type=float, default=0.7)
    args = parser.parse_args()
    if args.threshold_grid_size < 2:
        raise ValueError("--threshold-grid-size must be >= 2.")
    if not np.isfinite(args.min_trade_fraction) or not 0.0 <= args.min_trade_fraction <= 1.0:
        raise ValueError("--min-trade-fraction must be in [0, 1].")
    if not np.isfinite(args.threshold_ema_alpha) or not 0.0 <= args.threshold_ema_alpha <= 1.0:
        raise ValueError("--threshold-ema-alpha must be in [0, 1].")

    summary = replay(args.source_dir, args.out_dir, args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
