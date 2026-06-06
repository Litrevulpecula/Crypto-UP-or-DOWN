#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

import lightgbm_5m_direction_btc_eth as base
import replay_lightgbm_5m_thresholds as replay_base


def parse_x_values(value: str | None) -> list[float]:
    if value is None:
        return [round(i / 10.0, 1) for i in range(11)]
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    for item in result:
        if not np.isfinite(item) or not 0.0 <= item <= 1.0:
            raise ValueError("--x-values must all be finite values in [0, 1].")
    return result


def timestamp(value: object) -> pd.Timestamp:
    item = pd.Timestamp(value)
    if item.tzinfo is None:
        return item.tz_localize("UTC")
    return item.tz_convert("UTC")


def period_mask(index: pd.DatetimeIndex, start: pd.Timestamp | None, end: pd.Timestamp | None) -> np.ndarray:
    mask = np.ones(len(index), dtype=bool)
    if start is not None:
        mask &= index >= start
    if end is not None:
        mask &= index <= end
    return mask


def make_runtime(source_info: dict, args: argparse.Namespace, alpha: float) -> SimpleNamespace:
    return SimpleNamespace(
        out_dir=args.out_dir,
        win_payoff=float(source_info.get("win_payoff", base.PAYOFF_WIN)),
        loss_payoff=float(source_info.get("loss_payoff", base.PAYOFF_LOSS)),
        min_trade_fraction=args.min_trade_fraction,
        threshold_grid_size=args.threshold_grid_size,
        threshold_ema_alpha=alpha,
        target_tie_policy=str(source_info.get("target_tie_policy", "down")),
    )


def add_symbol_metrics(
    bucket: dict,
    symbol: str,
    actual: np.ndarray,
    probability: np.ndarray,
    active: np.ndarray,
    wins: np.ndarray,
    pnl: np.ndarray,
    tie_mask: np.ndarray,
    actual_tie: np.ndarray,
) -> None:
    data = bucket[symbol]
    data["actual"].append(actual)
    data["probability"].append(probability)
    data["active"].append(active)
    data["wins"].append(wins)
    data["pnl"].append(pnl)
    data["tie_mask"].append(tie_mask)
    data["actual_tie"].append(actual_tie)


def summarize_symbol_parts(parts: dict, symbol: str) -> dict:
    active = np.concatenate(parts["active"]) if parts["active"] else np.array([], dtype=bool)
    wins = np.concatenate(parts["wins"]) if parts["wins"] else np.array([], dtype=bool)
    pnl = np.concatenate(parts["pnl"]) if parts["pnl"] else np.array([], dtype=float)
    tie_mask = np.concatenate(parts["tie_mask"]) if parts["tie_mask"] else np.array([], dtype=bool)
    actual_tie = np.concatenate(parts["actual_tie"]) if parts["actual_tie"] else np.array([], dtype=bool)
    actual = np.concatenate(parts["actual"]) if parts["actual"] else np.array([], dtype=np.int8)
    probability = np.concatenate(parts["probability"]) if parts["probability"] else np.array([], dtype=float)
    strict_active = active & ~tie_mask
    prefix = symbol.replace("USDT", "")
    return {
        "symbol": symbol,
        "prefix": prefix,
        "samples": int(len(pnl)),
        "trades": int(active.sum()),
        "trade_fraction": float(active.mean()) if len(active) else float("nan"),
        "tie_trades": int((active & actual_tie).sum()),
        "tie_trade_fraction": float((active & actual_tie).mean()) if len(active) else float("nan"),
        "wins": int(wins[strict_active].sum()) if len(wins) else 0,
        "losses": int((strict_active & ~wins).sum()) if len(wins) else 0,
        "win_rate": float(wins[strict_active].mean()) if strict_active.any() else float("nan"),
        "auc": base.auc_or_nan(actual, probability, tie_mask) if len(actual) else float("nan"),
        "pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "annualized_sharpe": base.sharpe_ratio(pnl, base.SAMPLE_MINUTES) if len(pnl) else float("nan"),
    }


def load_source_context(source_dir: Path, args: argparse.Namespace) -> dict:
    source_info = replay_base.read_json(source_dir / "dataset_info.json")
    source_folds = pd.read_csv(source_dir / "folds.csv")
    symbols = replay_base.parse_symbols(source_info.get("symbols", base.SYMBOLS))
    target_symbols = replay_base.parse_symbols(source_info.get("target_symbols", list(symbols)))
    target_tie_policy = str(source_info.get("target_tie_policy", "down"))
    feature_set = str(source_info.get("feature_set", "v1"))
    data_root = args.data_root or Path(source_info.get("data_root", "aligned_data"))
    dataset = base.build_dataset(
        data_root=data_root,
        symbols=symbols,
        target_symbols=target_symbols,
        market=source_info.get("market", base.SPOT_MARKET),
        futures_market=source_info.get("futures_market", base.FUTURES_MARKET),
        dataset_start=None,
        target_horizon_minutes=base.TARGET_HORIZON_MINUTES,
        sample_minutes=base.SAMPLE_MINUTES,
        target_tie_policy=target_tie_policy,
        feature_set=feature_set,
    )
    return {
        "source_dir": source_dir,
        "source_info": source_info,
        "source_folds": source_folds,
        "dataset": dataset,
        "target_symbols": target_symbols,
        "columns_by_symbol": replay_base.load_feature_columns(source_dir, dataset, target_symbols),
    }


def replay_context_for_alpha(
    context: dict,
    args: argparse.Namespace,
    x_old_weight: float,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> tuple[list[dict], list[pd.Series]]:
    source_dir = context["source_dir"]
    source_info = context["source_info"]
    source_folds = context["source_folds"]
    dataset = context["dataset"]
    target_symbols = context["target_symbols"]
    columns_by_symbol = context["columns_by_symbol"]
    alpha = 1.0 - x_old_weight
    runtime = make_runtime(source_info, args, alpha)
    previous_thresholds_by_symbol: dict[str, dict] = {}
    symbol_parts: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    portfolio_parts: list[pd.Series] = []

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
            raise RuntimeError(f"Empty validation/test split for fold {fold_id} in {source_dir}.")
        mask = period_mask(test.index, start, end)

        for symbol in target_symbols:
            prefix = symbol.replace("USDT", "")
            columns = columns_by_symbol[symbol]
            model_path = replay_base.resolve_model_path(source_dir, source_fold[f"{prefix}_model_path"], fold_id, prefix)
            booster = lgb.Booster(model_file=str(model_path))

            validation_probability = booster.predict(validation[columns])
            validation_actual = validation[f"{prefix}_target"].to_numpy(dtype=np.int8)
            validation_tie = base.payoff_tie_mask(validation, prefix, runtime)
            raw_thresholds = base.choose_thresholds(validation_probability, validation_actual, runtime, validation_tie)
            adjusted_thresholds = base.adjust_extreme_thresholds(raw_thresholds, validation_probability)
            thresholds = base.smooth_thresholds(
                adjusted_thresholds,
                previous_thresholds_by_symbol.get(symbol),
                alpha,
            )
            previous_thresholds_by_symbol[symbol] = thresholds

            probability = booster.predict(test[columns])
            actual = test[f"{prefix}_target"].to_numpy(dtype=np.int8)
            test_tie = base.payoff_tie_mask(test, prefix, runtime)
            actual_test_tie = base.target_tie_mask(test, prefix)
            _, active, wins, pnl = base.apply_thresholds(probability, actual, thresholds, runtime, test_tie)
            if not mask.any():
                continue
            add_symbol_metrics(
                symbol_parts,
                symbol,
                actual[mask],
                probability[mask],
                active[mask],
                wins[mask],
                pnl[mask],
                test_tie[mask],
                actual_test_tie[mask],
            )
            portfolio_parts.append(pd.Series(pnl[mask], index=test.index[mask], dtype=float))

    symbol_rows = [summarize_symbol_parts(parts, symbol) for symbol, parts in sorted(symbol_parts.items())]
    return symbol_rows, portfolio_parts


def run_sweep(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    x_values = parse_x_values(args.x_values)
    start = base.parse_utc_timestamp(args.start)
    end = base.parse_utc_timestamp(args.end)
    contexts = [load_source_context(path, args) for path in args.source_dirs]
    summary_rows = []
    symbol_rows = []

    for x_old_weight in x_values:
        all_symbol_rows = []
        portfolio_parts = []
        for context in contexts:
            rows, parts = replay_context_for_alpha(context, args, x_old_weight, start, end)
            all_symbol_rows.extend(rows)
            portfolio_parts.extend(parts)
        for row in all_symbol_rows:
            row = row.copy()
            row["x_old_threshold_weight"] = float(x_old_weight)
            row["threshold_ema_alpha_new_weight"] = float(1.0 - x_old_weight)
            symbol_rows.append(row)

        if portfolio_parts:
            portfolio = pd.concat(portfolio_parts, axis=1).fillna(0.0).sum(axis=1).sort_index().to_numpy(dtype=float)
        else:
            portfolio = np.array([], dtype=float)
        sharpes = np.array([row["annualized_sharpe"] for row in all_symbol_rows], dtype=float)
        pnls = np.array([row["pnl"] for row in all_symbol_rows], dtype=float)
        trades = np.array([row["trades"] for row in all_symbol_rows], dtype=float)
        summary_rows.append(
            {
                "x_old_threshold_weight": float(x_old_weight),
                "threshold_ema_alpha_new_weight": float(1.0 - x_old_weight),
                "symbols": len(all_symbol_rows),
                "samples": int(sum(row["samples"] for row in all_symbol_rows)),
                "trades": int(trades.sum()) if len(trades) else 0,
                "total_pnl": float(pnls.sum()) if len(pnls) else 0.0,
                "mean_symbol_sharpe": float(np.nanmean(sharpes)) if len(sharpes) else float("nan"),
                "median_symbol_sharpe": float(np.nanmedian(sharpes)) if len(sharpes) else float("nan"),
                "min_symbol_sharpe": float(np.nanmin(sharpes)) if len(sharpes) else float("nan"),
                "portfolio_sharpe": base.sharpe_ratio(portfolio, base.SAMPLE_MINUTES) if len(portfolio) else float("nan"),
            }
        )
        print(
            f"x={x_old_weight:.1f} alpha={1.0 - x_old_weight:.1f} "
            f"portfolio_sharpe={summary_rows[-1]['portfolio_sharpe']:.4g} "
            f"mean_symbol_sharpe={summary_rows[-1]['mean_symbol_sharpe']:.4g} "
            f"pnl={summary_rows[-1]['total_pnl']:.1f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows).sort_values("x_old_threshold_weight")
    by_symbol = pd.DataFrame(symbol_rows).sort_values(["x_old_threshold_weight", "symbol"])
    summary.to_csv(args.out_dir / "ema_x_sweep_summary.csv", index=False)
    by_symbol.to_csv(args.out_dir / "ema_x_sweep_symbol_metrics.csv", index=False)
    info = {
        "source_dirs": [str(path) for path in args.source_dirs],
        "x_definition": "used_threshold = x * previous_used_threshold + (1 - x) * newly_selected_threshold",
        "x_values": x_values,
        "threshold_grid_size": args.threshold_grid_size,
        "min_trade_fraction": args.min_trade_fraction,
        "start": start.isoformat() if start is not None else None,
        "end": end.isoformat() if end is not None else None,
        "selection_hint": "Prefer portfolio_sharpe for one capital pool, or mean_symbol_sharpe to weight symbols equally.",
    }
    with (args.out_dir / "ema_x_sweep_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep old-threshold EMA weight for saved LightGBM walk-forward models.")
    parser.add_argument("--source-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument("--min-trade-fraction", type=float, default=0.05)
    parser.add_argument("--x-values", default=None, help="Comma-separated old-threshold weights. Default: 0.0,0.1,...,1.0")
    parser.add_argument("--start", default=None, help="Optional UTC-inclusive evaluation start.")
    parser.add_argument("--end", default=None, help="Optional UTC-inclusive evaluation end.")
    args = parser.parse_args()
    if args.threshold_grid_size < 2:
        raise ValueError("--threshold-grid-size must be >= 2.")
    if not np.isfinite(args.min_trade_fraction) or not 0.0 <= args.min_trade_fraction <= 1.0:
        raise ValueError("--min-trade-fraction must be in [0, 1].")
    run_sweep(args)


if __name__ == "__main__":
    main()
