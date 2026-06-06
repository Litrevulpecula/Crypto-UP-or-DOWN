#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from lightgbm_5m_direction_btc_eth import (
    FEATURE_SETS,
    FUTURES_MARKET,
    SAMPLE_MINUTES,
    SPOT_MARKET,
    SYMBOLS,
    TARGET_HORIZON_MINUTES,
    build_dataset,
    feature_columns_for_symbol,
    parse_symbols,
    parse_utc_timestamp,
    target_symbols_for_run,
)


def time_uniform_sample(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame
    positions = np.linspace(0, len(frame) - 1, max_rows, dtype=np.int64)
    positions = np.unique(positions)
    return frame.iloc[positions]


def feature_group(feature: str, symbol_prefix: str) -> str:
    local = feature
    for prefix in (f"{symbol_prefix}_spot_", f"{symbol_prefix}_"):
        if local.startswith(prefix):
            local = local[len(prefix) :]
            break
    if feature.startswith(f"{symbol_prefix}_futures_") or "basis" in local:
        return "futures_spot"
    if local.startswith("close_vs") or "rolling_" in local or "kama" in local:
        return "location"
    if "ret" in local or "vol" in local or "accel" in local or "velocity" in local:
        return "return_volatility"
    if "buy" in local or "taker" in local:
        return "order_flow"
    if "volume" in local or "count" in local:
        return "liquidity"
    if feature in {"sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek"}:
        return "time"
    return "other"


def analyze_symbol(dataset: pd.DataFrame, symbol: str, max_rows: int | None, random_state: int) -> pd.DataFrame:
    prefix = symbol.replace("USDT", "")
    features = feature_columns_for_symbol(dataset, symbol)
    target_col = f"{prefix}_target"
    tie_col = f"{prefix}_target_tie"
    symbol_frame = dataset.loc[dataset[tie_col].eq(0), features + [target_col]]
    symbol_frame = time_uniform_sample(symbol_frame, max_rows)

    x = symbol_frame[features].to_numpy(dtype=np.float64, copy=True)
    y = symbol_frame[target_col].to_numpy(dtype=np.int8, copy=True)
    mi = mutual_info_classif(
        x,
        y,
        discrete_features=False,
        n_neighbors=3,
        copy=False,
        random_state=random_state,
    )
    result = pd.DataFrame(
        {
            "symbol": symbol,
            "feature": features,
            "mutual_info": mi,
            "sample_rows": len(symbol_frame),
            "target_up_rate": float(y.mean()),
        }
    )
    result["rank"] = result["mutual_info"].rank(method="first", ascending=False).astype(int)
    result["feature_group"] = [feature_group(feature, prefix) for feature in result["feature"]]
    return result.sort_values("mutual_info", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank BTC/ETH features by mutual information with 5m direction target.")
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data"))
    parser.add_argument("--market", default=SPOT_MARKET)
    parser.add_argument("--futures-market", default=FUTURES_MARKET)
    parser.add_argument("--out-dir", type=Path, default=Path("results/feature_mutual_info_v1"))
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--target-symbols", default=None)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="v1")
    parser.add_argument("--dataset-start", default=None)
    parser.add_argument("--sample-minutes", type=int, default=SAMPLE_MINUTES)
    parser.add_argument("--max-rows-per-symbol", type=int, default=120_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)
    target_symbols = target_symbols_for_run(symbols, args.target_symbols)
    dataset = build_dataset(
        data_root=args.data_root,
        symbols=symbols,
        target_symbols=target_symbols,
        market=args.market,
        futures_market=args.futures_market,
        dataset_start=parse_utc_timestamp(args.dataset_start),
        target_horizon_minutes=TARGET_HORIZON_MINUTES,
        sample_minutes=args.sample_minutes,
        target_tie_policy="expected",
        feature_set=args.feature_set,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [
        analyze_symbol(dataset, symbol, args.max_rows_per_symbol, args.random_state)
        for symbol in target_symbols
    ]
    ranking = pd.concat(results, ignore_index=True)
    ranking.to_csv(args.out_dir / "mutual_info_by_feature.csv", index=False)

    low_info = ranking.loc[ranking["mutual_info"].le(1e-5)].copy()
    low_info.to_csv(args.out_dir / "low_mutual_info_features.csv", index=False)

    group_summary = (
        ranking.groupby(["symbol", "feature_group"], as_index=False)
        .agg(
            feature_count=("feature", "count"),
            mean_mutual_info=("mutual_info", "mean"),
            median_mutual_info=("mutual_info", "median"),
            zeroish_features=("mutual_info", lambda values: int((values <= 1e-5).sum())),
        )
        .sort_values(["symbol", "mean_mutual_info"], ascending=[True, False])
    )
    group_summary.to_csv(args.out_dir / "mutual_info_by_group.csv", index=False)

    summary = {
        "rows": int(len(dataset)),
        "start": dataset.index.min().isoformat(),
        "end": dataset.index.max().isoformat(),
        "feature_set": args.feature_set,
        "sample_minutes": args.sample_minutes,
        "max_rows_per_symbol": args.max_rows_per_symbol,
        "feature_counts": {
            symbol: int((ranking["symbol"] == symbol).sum())
            for symbol in target_symbols
        },
        "low_info_threshold": 1e-5,
        "low_info_counts": {
            symbol: int(((ranking["symbol"] == symbol) & ranking["mutual_info"].le(1e-5)).sum())
            for symbol in target_symbols
        },
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))
    print("\nTop 15:")
    print(ranking.groupby("symbol", group_keys=False).head(15).to_string(index=False))
    print("\nBottom 20:")
    print(ranking.groupby("symbol", group_keys=False).tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
