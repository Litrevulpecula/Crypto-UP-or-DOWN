#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm_5m_direction_btc_eth as base


DEFAULT_TARGET_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT")


def context_symbols_for_target(target: str) -> tuple[str, ...]:
    symbols = ["BTCUSDT", "ETHUSDT"]
    if target not in symbols:
        symbols.append(target)
    return tuple(symbols)


def parse_symbol_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip().upper() for item in value.split(",") if item.strip())


def parse_ts(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return base.parse_utc_timestamp(value)


def feature_group(feature: str, prefix: str) -> str:
    name = feature
    if name in base.TIME_FEATURES:
        return "time"
    if name.startswith(f"{prefix}_"):
        name = name[len(prefix) + 1 :]
    if "basis" in name or "futures_spot" in name:
        return "basis_futures"
    if "buy_ratio" in name or "taker_buy" in name or "buy_imbalance" in name or "buy_pressure" in name:
        return "order_flow"
    if "volume" in name or "quote" in name or "count" in name or "trade_size" in name or "avg_trade" in name:
        return "liquidity_volume"
    if "trend_" in name or "efficiency_ratio" in name or "up_bar_ratio" in name or "down_bar_ratio" in name:
        return "trend_quality"
    if "realized_vol" in name or "range" in name or "shadow" in name or "body" in name:
        return "volatility_shape"
    if "log_ret" in name or "ret_accel" in name:
        return "momentum"
    if (
        "kama" in name
        or "sma" in name
        or "rolling_high" in name
        or "rolling_low" in name
        or "close_position" in name
        or "vwap_dist" in name
    ):
        return "mean_reversion_location"
    return "other"


def period_start_end(labels: pd.Series) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    result = {}
    for label, values in labels.groupby(labels).groups.items():
        index = pd.DatetimeIndex(values)
        result[str(label)] = (index.min(), index.max())
    return result


def period_labels(index: pd.DatetimeIndex, period: str) -> pd.Series:
    if period == "year":
        labels = index.strftime("%Y")
    elif period == "quarter":
        labels = index.to_period("Q").astype(str)
    elif period == "month":
        labels = index.strftime("%Y-%m")
    else:
        raise ValueError("--period must be one of: year, quarter, month.")
    return pd.Series(labels, index=index, name="period")


def pearson_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def payoff_for_positions(
    actual_up: np.ndarray,
    tie: np.ndarray,
    position: np.ndarray,
    win_payoff: float,
    loss_payoff: float,
    tie_payoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = position != 0
    wins = ((position == 1) & (actual_up == 1)) | ((position == -1) & (actual_up == 0))
    pnl = np.zeros(len(position), dtype=float)
    pnl[active & tie] = tie_payoff
    strict = active & ~tie
    pnl[strict & wins] = win_payoff
    pnl[strict & ~wins] = loss_payoff
    return wins, pnl


def quantile_payoff_metrics(
    feature: np.ndarray,
    actual_up: np.ndarray,
    tie: np.ndarray,
    quantile: float,
    win_payoff: float,
    loss_payoff: float,
    tie_payoff: float,
) -> dict:
    low_cut = float(np.nanquantile(feature, quantile))
    high_cut = float(np.nanquantile(feature, 1.0 - quantile))
    low = feature <= low_cut
    high = feature >= high_cut
    if high_cut <= low_cut:
        low = np.zeros(len(feature), dtype=bool)
        high = np.zeros(len(feature), dtype=bool)
    position = np.zeros(len(feature), dtype=np.int8)
    position[low] = -1
    position[high] = 1
    wins, pnl = payoff_for_positions(actual_up, tie, position, win_payoff, loss_payoff, tie_payoff)
    active = position != 0
    strict_active = active & ~tie
    high_wins, high_pnl = payoff_for_positions(
        actual_up,
        tie,
        np.where(high, 1, 0).astype(np.int8),
        win_payoff,
        loss_payoff,
        tie_payoff,
    )
    low_wins, low_pnl = payoff_for_positions(
        actual_up,
        tie,
        np.where(low, -1, 0).astype(np.int8),
        win_payoff,
        loss_payoff,
        tie_payoff,
    )
    high_strict = high & ~tie
    low_strict = low & ~tie
    return {
        "low_cut": low_cut,
        "high_cut": high_cut,
        "quantile_trades": int(active.sum()),
        "quantile_trade_fraction": float(active.mean()) if len(active) else float("nan"),
        "quantile_tie_trades": int((active & tie).sum()),
        "factor_pnl": float(pnl.sum()),
        "factor_pnl_per_trade": float(pnl[active].mean()) if active.any() else float("nan"),
        "factor_win_rate": float(wins[strict_active].mean()) if strict_active.any() else float("nan"),
        "high_long_trades": int(high.sum()),
        "high_long_pnl": float(high_pnl.sum()),
        "high_long_pnl_per_trade": float(high_pnl[high].mean()) if high.any() else float("nan"),
        "high_long_win_rate": float(high_wins[high_strict].mean()) if high_strict.any() else float("nan"),
        "low_short_trades": int(low.sum()),
        "low_short_pnl": float(low_pnl.sum()),
        "low_short_pnl_per_trade": float(low_pnl[low].mean()) if low.any() else float("nan"),
        "low_short_win_rate": float(low_wins[low_strict].mean()) if low_strict.any() else float("nan"),
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid].astype(float), weights=weights[valid].astype(float)))


def sign_or_zero(value: float) -> int:
    if not np.isfinite(value) or value == 0:
        return 0
    return 1 if value > 0 else -1


def compute_symbol_period_metrics(dataset: pd.DataFrame, symbol: str, args: argparse.Namespace) -> pd.DataFrame:
    prefix = symbol.replace("USDT", "")
    columns = base.feature_columns_for_symbol(dataset, symbol)
    labels = period_labels(dataset.index, args.period)
    bounds = period_start_end(labels)
    target = dataset[f"{prefix}_target"].to_numpy(dtype=np.int8)
    tie = base.target_tie_mask(dataset, prefix)
    future_return = np.log(dataset[f"{prefix}_future_close"].to_numpy(dtype=float) / dataset[f"{prefix}_spot_close"].to_numpy(dtype=float))
    rows = []
    label_values = labels.to_numpy()

    for column in columns:
        values = dataset[column].to_numpy(dtype=float)
        finite_base = np.isfinite(values) & np.isfinite(future_return)
        if finite_base.sum() < args.min_rows:
            continue
        group = feature_group(column, prefix)
        for period, period_index in labels.groupby(labels).groups.items():
            del period_index
            selected = np.flatnonzero((label_values == period) & finite_base)
            if int(len(selected)) < args.min_rows:
                continue
            x = values[selected]
            y_return = future_return[selected]
            y_target = target[selected]
            y_tie = tie[selected]
            if np.nanstd(x) == 0:
                continue
            row = {
                "symbol": symbol,
                "prefix": prefix,
                "feature": column,
                "feature_group": group,
                "period": str(period),
                "period_start": bounds[str(period)][0].isoformat(),
                "period_end": bounds[str(period)][1].isoformat(),
                "rows": int(len(selected)),
                "non_tie_rows": int((~y_tie).sum()),
                "feature_mean": float(np.nanmean(x)),
                "feature_std": float(np.nanstd(x)),
                "future_return_mean": float(np.nanmean(y_return)),
                "up_rate_no_tie": float(y_target[~y_tie].mean()) if (~y_tie).any() else float("nan"),
                "spearman_ic": spearman_or_nan(x, y_return),
                "pearson_ic": pearson_or_nan(x, y_return),
                "auc": base.auc_or_nan(y_target, x, y_tie),
            }
            row["abs_auc_edge"] = abs(row["auc"] - 0.5) if np.isfinite(row["auc"]) else float("nan")
            row.update(
                quantile_payoff_metrics(
                    x,
                    y_target,
                    y_tie,
                    args.quantile,
                    args.win_payoff,
                    args.loss_payoff,
                    args.tie_payoff,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def window_mask(metrics: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Series:
    period_start = pd.to_datetime(metrics["period_start"], utc=True)
    period_end = pd.to_datetime(metrics["period_end"], utc=True)
    mask = pd.Series(True, index=metrics.index)
    if start is not None:
        mask &= period_end >= start
    if end is not None:
        mask &= period_start <= end
    return mask


def summarize_decay(metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    early_start = parse_ts(args.early_start)
    early_end = parse_ts(args.early_end)
    late_start = parse_ts(args.late_start)
    late_end = parse_ts(args.late_end)
    early_mask = window_mask(metrics, early_start, early_end)
    late_mask = window_mask(metrics, late_start, late_end)
    rows = []

    for (symbol, feature), group in metrics.groupby(["symbol", "feature"], sort=False):
        early = group.loc[early_mask.reindex(group.index, fill_value=False)]
        late = group.loc[late_mask.reindex(group.index, fill_value=False)]
        if early.empty or late.empty:
            continue
        early_ppt = weighted_mean(early["factor_pnl_per_trade"], early["quantile_trades"])
        late_ppt = weighted_mean(late["factor_pnl_per_trade"], late["quantile_trades"])
        early_ic = weighted_mean(early["spearman_ic"], early["rows"])
        late_ic = weighted_mean(late["spearman_ic"], late["rows"])
        early_auc = weighted_mean(early["auc"], early["non_tie_rows"])
        late_auc = weighted_mean(late["auc"], late["non_tie_rows"])
        orientation = sign_or_zero(early_ppt)
        if orientation == 0:
            orientation = sign_or_zero(early_ic)
        early_oriented_ppt = early_ppt * orientation if orientation else float("nan")
        late_oriented_ppt = late_ppt * orientation if orientation else float("nan")
        if not np.isfinite(early_oriented_ppt):
            status = "no_early_edge"
        elif not np.isfinite(late_oriented_ppt):
            status = "missing_late"
        elif early_oriented_ppt <= 0:
            status = "no_early_edge"
        elif late_oriented_ppt < 0:
            status = "flipped_or_negative"
        elif late_oriented_ppt < 0.25 * early_oriented_ppt:
            status = "decayed_75pct"
        elif late_oriented_ppt < 0.75 * early_oriented_ppt:
            status = "decayed"
        else:
            status = "stable_or_improved"
        rows.append(
            {
                "symbol": symbol,
                "prefix": symbol.replace("USDT", ""),
                "feature": feature,
                "feature_group": group["feature_group"].iloc[0],
                "early_periods": int(len(early)),
                "late_periods": int(len(late)),
                "early_rows": int(early["rows"].sum()),
                "late_rows": int(late["rows"].sum()),
                "early_quantile_trades": int(early["quantile_trades"].sum()),
                "late_quantile_trades": int(late["quantile_trades"].sum()),
                "orientation_from_early": int(orientation),
                "early_spearman_ic": early_ic,
                "late_spearman_ic": late_ic,
                "abs_ic_decay": abs(early_ic) - abs(late_ic) if np.isfinite(early_ic) and np.isfinite(late_ic) else float("nan"),
                "ic_sign_flip": bool(np.isfinite(early_ic) and np.isfinite(late_ic) and early_ic * late_ic < 0),
                "early_auc": early_auc,
                "late_auc": late_auc,
                "auc_edge_decay": (
                    abs(early_auc - 0.5) - abs(late_auc - 0.5)
                    if np.isfinite(early_auc) and np.isfinite(late_auc)
                    else float("nan")
                ),
                "early_factor_pnl_per_trade": early_ppt,
                "late_factor_pnl_per_trade": late_ppt,
                "early_oriented_pnl_per_trade": early_oriented_ppt,
                "late_oriented_pnl_per_trade": late_oriented_ppt,
                "oriented_pnl_decay": (
                    early_oriented_ppt - late_oriented_ppt
                    if np.isfinite(early_oriented_ppt) and np.isfinite(late_oriented_ppt)
                    else float("nan")
                ),
                "late_retention_ratio": (
                    late_oriented_ppt / early_oriented_ppt
                    if np.isfinite(early_oriented_ppt) and early_oriented_ppt > 0 and np.isfinite(late_oriented_ppt)
                    else float("nan")
                ),
                "status": status,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["oriented_pnl_decay", "abs_ic_decay"], ascending=[False, False])


def summarize_groups(decay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if decay.empty:
        return pd.DataFrame(rows)
    for (symbol, feature_group), group in decay.groupby(["symbol", "feature_group"], sort=False):
        rows.append(
            {
                "symbol": symbol,
                "prefix": symbol.replace("USDT", ""),
                "feature_group": feature_group,
                "features": int(len(group)),
                "flipped_or_negative": int((group["status"] == "flipped_or_negative").sum()),
                "decayed_or_flipped": int(group["status"].isin(["flipped_or_negative", "decayed_75pct", "decayed"]).sum()),
                "early_oriented_pnl_per_trade_mean": float(group["early_oriented_pnl_per_trade"].mean()),
                "late_oriented_pnl_per_trade_mean": float(group["late_oriented_pnl_per_trade"].mean()),
                "late_retention_ratio_median": float(group["late_retention_ratio"].median()),
                "oriented_pnl_decay_mean": float(group["oriented_pnl_decay"].mean()),
                "early_abs_ic_mean": float(group["early_spearman_ic"].abs().mean()),
                "late_abs_ic_mean": float(group["late_spearman_ic"].abs().mean()),
                "abs_ic_decay_mean": float(group["abs_ic_decay"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["oriented_pnl_decay_mean", "abs_ic_decay_mean"], ascending=[False, False])


def summarize_group_periods(period_metrics: pd.DataFrame, decay: pd.DataFrame) -> pd.DataFrame:
    if period_metrics.empty or decay.empty:
        return pd.DataFrame()
    orientation = decay[["symbol", "feature", "orientation_from_early"]]
    merged = period_metrics.merge(orientation, on=["symbol", "feature"], how="left")
    merged["orientation_from_early"] = merged["orientation_from_early"].fillna(0)
    merged["oriented_pnl_per_trade"] = merged["factor_pnl_per_trade"] * merged["orientation_from_early"]
    merged["oriented_spearman_ic"] = merged["spearman_ic"] * merged["orientation_from_early"]
    rows = []
    for (symbol, group_name, period), group in merged.groupby(["symbol", "feature_group", "period"], sort=False):
        trade_weights = group["quantile_trades"].astype(float)
        row_weights = group["rows"].astype(float)
        rows.append(
            {
                "symbol": symbol,
                "prefix": symbol.replace("USDT", ""),
                "feature_group": group_name,
                "period": period,
                "features": int(group["feature"].nunique()),
                "oriented_pnl_per_trade": weighted_mean(group["oriented_pnl_per_trade"], trade_weights),
                "oriented_spearman_ic": weighted_mean(group["oriented_spearman_ic"], row_weights),
                "factor_pnl": float(group["factor_pnl"].sum()),
                "quantile_trades": int(group["quantile_trades"].sum()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["symbol", "feature_group", "period"])


def write_top_lists(decay: pd.DataFrame, out_dir: Path, top_n: int) -> dict:
    outputs = {}
    if decay.empty:
        return outputs
    filters = {
        "top_decayed_features.csv": decay,
        "top_flipped_features.csv": decay.loc[decay["status"] == "flipped_or_negative"],
        "top_stable_features.csv": decay.loc[decay["status"] == "stable_or_improved"].sort_values(
            ["late_oriented_pnl_per_trade", "late_spearman_ic"],
            ascending=[False, False],
        ),
    }
    for filename, frame in filters.items():
        path = out_dir / filename
        frame.head(top_n).to_csv(path, index=False)
        outputs[filename] = str(path)
    return outputs


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_symbols = parse_symbol_list(args.target_symbols)
    all_period_metrics = []
    inventory_rows = []

    for symbol in target_symbols:
        context_symbols = context_symbols_for_target(symbol)
        print(f"build dataset target={symbol} context={','.join(context_symbols)}", flush=True)
        dataset = base.build_dataset(
            data_root=args.data_root,
            symbols=context_symbols,
            target_symbols=(symbol,),
            market=args.market,
            futures_market=args.futures_market,
            dataset_start=parse_ts(args.dataset_start),
            target_horizon_minutes=base.TARGET_HORIZON_MINUTES,
            sample_minutes=base.SAMPLE_MINUTES,
            target_tie_policy="expected",
            feature_set=args.feature_set,
        )
        columns = base.feature_columns_for_symbol(dataset, symbol)
        for column in columns:
            inventory_rows.append(
                {
                    "symbol": symbol,
                    "prefix": symbol.replace("USDT", ""),
                    "feature": column,
                    "feature_group": feature_group(column, symbol.replace("USDT", "")),
                }
            )
        print(f"diagnose target={symbol} rows={len(dataset)} features={len(columns)}", flush=True)
        all_period_metrics.append(compute_symbol_period_metrics(dataset, symbol, args))

    period_metrics = pd.concat(all_period_metrics, ignore_index=True)
    decay = summarize_decay(period_metrics, args)
    groups = summarize_groups(decay)
    group_periods = summarize_group_periods(period_metrics, decay)
    pd.DataFrame(inventory_rows).to_csv(args.out_dir / "feature_inventory.csv", index=False)
    period_metrics.to_csv(args.out_dir / "factor_period_metrics.csv", index=False)
    decay.to_csv(args.out_dir / "factor_decay_summary.csv", index=False)
    groups.to_csv(args.out_dir / "factor_group_decay_summary.csv", index=False)
    group_periods.to_csv(args.out_dir / "factor_group_period_oriented_metrics.csv", index=False)
    top_files = write_top_lists(decay, args.out_dir, args.top_n)

    status_counts = decay["status"].value_counts().to_dict() if not decay.empty else {}
    group_late = (
        groups.groupby("feature_group")["late_oriented_pnl_per_trade_mean"].mean().sort_values(ascending=False).to_dict()
        if not groups.empty
        else {}
    )
    report = {
        "target_symbols": list(target_symbols),
        "period": args.period,
        "quantile": args.quantile,
        "early_window": {"start": args.early_start, "end": args.early_end},
        "late_window": {"start": args.late_start, "end": args.late_end},
        "payoff": {"win": args.win_payoff, "loss": args.loss_payoff, "tie": args.tie_payoff},
        "rows": {
            "feature_inventory": int(len(inventory_rows)),
            "period_metrics": int(len(period_metrics)),
            "decay_summary": int(len(decay)),
            "group_summary": int(len(groups)),
        },
        "status_counts": status_counts,
        "group_late_oriented_pnl_per_trade_mean": group_late,
        "outputs": {
            "feature_inventory": str(args.out_dir / "feature_inventory.csv"),
            "period_metrics": str(args.out_dir / "factor_period_metrics.csv"),
            "factor_decay_summary": str(args.out_dir / "factor_decay_summary.csv"),
            "factor_group_decay_summary": str(args.out_dir / "factor_group_decay_summary.csv"),
            "factor_group_period_oriented_metrics": str(args.out_dir / "factor_group_period_oriented_metrics.csv"),
            **top_files,
        },
    }
    with (args.out_dir / "diagnosis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose factor decay for the 5-minute LightGBM feature set.")
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data"))
    parser.add_argument("--market", default=base.SPOT_MARKET)
    parser.add_argument("--futures-market", default=base.FUTURES_MARKET)
    parser.add_argument("--out-dir", type=Path, default=Path("results/factor_decay_diagnosis"))
    parser.add_argument("--target-symbols", default=",".join(DEFAULT_TARGET_SYMBOLS))
    parser.add_argument("--feature-set", choices=base.FEATURE_SETS, default="v1")
    parser.add_argument("--dataset-start", default="2022-01-01")
    parser.add_argument("--period", choices=["year", "quarter", "month"], default="year")
    parser.add_argument("--early-start", default="2023-01-01")
    parser.add_argument("--early-end", default="2023-12-31 23:59:59")
    parser.add_argument("--late-start", default="2024-01-01")
    parser.add_argument("--late-end", default=None)
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--win-payoff", type=float, default=base.PAYOFF_WIN)
    parser.add_argument("--loss-payoff", type=float, default=base.PAYOFF_LOSS)
    parser.add_argument("--tie-payoff", type=float, default=(base.PAYOFF_WIN + base.PAYOFF_LOSS) / 2.0)
    args = parser.parse_args()
    if not 0.0 < args.quantile < 0.5:
        raise ValueError("--quantile must be in (0, 0.5).")
    if args.min_rows < 10:
        raise ValueError("--min-rows must be >= 10.")
    run(args)


if __name__ == "__main__":
    main()
