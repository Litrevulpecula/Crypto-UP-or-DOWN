#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


LIVE_DIR = Path(__file__).resolve().parent
HIBT_DIR = LIVE_DIR / "hibt"
if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))
if str(HIBT_DIR) not in sys.path:
    sys.path.insert(0, str(HIBT_DIR))

import lightgbm_5m_direction_btc_eth as base  # noqa: E402
from hibt_config import normalize_symbol  # noqa: E402


DEFAULT_MODEL_DIRS = {"5m": LIVE_DIR / "models_5m", "15m": LIVE_DIR / "models_15m"}
MODEL_BUNDLE_CACHE: dict[tuple[Path, tuple[str, ...]], dict[str, Any]] = {}
BASE_OPEN_TIME_CACHE: dict[Path, pd.Timestamp | None] = {}


def parse_model_dirs(values: list[str]) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_MODEL_DIRS)
    items = values
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model-dir must be timeframe=path, got {item!r}")
        timeframe, path = item.split("=", 1)
        normalized = normalize_timeframe(timeframe)
        if normalized is None:
            raise ValueError(f"Invalid timeframe in --model-dir: {item!r}")
        result[normalized] = Path(path)
    return result


def normalize_timeframe(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().lower().replace(" ", "")
    if not text:
        return None
    if text.endswith("min"):
        text = text[:-3] + "m"
    return text


def timeframe_minutes(value: str) -> int:
    timeframe = normalize_timeframe(value)
    if timeframe is None or not timeframe.endswith("m"):
        raise ValueError(f"Unsupported timeframe: {value!r}")
    try:
        minutes = int(timeframe[:-1])
    except ValueError as exc:
        raise ValueError(f"Unsupported timeframe: {value!r}") from exc
    if minutes <= 0:
        raise ValueError(f"Unsupported timeframe: {value!r}")
    return minutes


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_live_feature_frame(
    data_root: Path,
    symbols: tuple[str, ...],
    sample_minutes: int,
    dataset_start: pd.Timestamp | None,
    feature_set: str,
) -> pd.DataFrame:
    if feature_set not in base.FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {base.FEATURE_SETS}.")
    market_feature_set = "enhanced"
    spot_frames = {}
    futures_frames = {}
    common_index: pd.DatetimeIndex | None = None

    for symbol in symbols:
        futures = base.read_1m(
            data_root / base.FUTURES_MARKET / symbol / "1m.csv.gz",
            dataset_start,
            target_horizon_minutes=sample_minutes,
        )
        spot = base.read_1m(
            data_root / base.SPOT_MARKET / symbol / "1m.csv.gz",
            dataset_start,
            target_horizon_minutes=sample_minutes,
        )
        spot_frames[symbol] = spot
        futures_frames[symbol] = futures
        symbol_index = spot.index.intersection(futures.index)
        common_index = symbol_index if common_index is None else common_index.intersection(symbol_index)

    if common_index is None or common_index.empty:
        raise RuntimeError("No common 1m timestamps were found for live feature generation.")
    common_index = common_index.sort_values()

    parts = []
    for symbol in symbols:
        prefix = symbol.replace("USDT", "")
        spot = spot_frames[symbol].loc[common_index]
        futures = futures_frames[symbol].loc[common_index]
        symbol_parts = [
            base.add_market_features(spot, f"{prefix}_spot", market_feature_set),
            base.add_spot_futures_features(spot, futures, prefix, market_feature_set),
            spot[["close"]].rename(columns={"close": f"{prefix}_spot_close"}),
        ]
        if feature_set == "v2" and symbol == "BTCUSDT":
            if "ETHUSDT" not in symbols:
                raise ValueError("feature_set v2 requires ETHUSDT in --symbols for BTC cross features.")
            eth_spot = spot_frames["ETHUSDT"].loc[common_index]
            eth_futures = futures_frames["ETHUSDT"].loc[common_index]
            symbol_parts.append(base.add_btc_eth_cross_features(spot, futures, eth_spot, eth_futures))
        parts.extend(symbol_parts)

    frame = pd.concat(parts, axis=1)
    frame = pd.concat([frame, base.add_time_features(frame.index)], axis=1)
    frame, _removed = base.drop_exact_duplicate_columns(frame)
    frame["last_kline_time"] = frame.index
    frame.index = frame.index + pd.Timedelta(minutes=1)
    frame.index.name = "decision_time"
    frame = frame.loc[base.aligned_decision_time_mask(frame.index, sample_minutes)]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    all_nan_columns = [column for column in frame.columns if frame[column].isna().all()]
    if all_nan_columns:
        frame = frame.drop(columns=all_nan_columns)
    return frame.dropna()


def latest_dataset_start(data_root: Path, symbols: tuple[str, ...], lookback_days: float | None) -> pd.Timestamp | None:
    if lookback_days is None or lookback_days <= 0:
        return None
    latest: pd.Timestamp | None = None
    for market in (base.SPOT_MARKET, base.FUTURES_MARKET):
        for symbol in symbols:
            path = data_root / market / symbol / "1m.csv.gz"
            current = latest_open_time(path)
            if current is None:
                continue
            latest = current if latest is None else min(latest, current)
    if latest is None:
        return None
    return latest - pd.Timedelta(days=float(lookback_days))


def latest_open_time(path: Path) -> pd.Timestamp | None:
    base_latest = read_latest_open_time(path, cache=True)
    live_latest = read_latest_open_time(path.with_name("1m_live.csv"), cache=False)
    if base_latest is None:
        return live_latest
    if live_latest is None:
        return base_latest
    return max(base_latest, live_latest)


def read_latest_open_time(path: Path, cache: bool) -> pd.Timestamp | None:
    key = path.resolve()
    if cache and key in BASE_OPEN_TIME_CACHE:
        return BASE_OPEN_TIME_CACHE[key]
    if not path.exists():
        latest = None
    else:
        columns = pd.read_csv(path, usecols=["open_time"])
        latest = None if columns.empty else pd.to_datetime(int(columns["open_time"].max()), unit="ms", utc=True)
    if cache:
        BASE_OPEN_TIME_CACHE[key] = latest
    return latest


def load_model_bundle(model_dir: Path, symbols: tuple[str, ...]) -> dict[str, Any]:
    cache_key = (model_dir.resolve(), symbols)
    cached = MODEL_BUNDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    metadata = read_json(model_dir / "live_model_metadata.json")
    feature_payload = read_json(model_dir / "feature_columns.json")
    bundle = {
        "metadata": metadata,
        "features": {},
        "boosters": {},
        "thresholds": {},
    }
    for symbol in symbols:
        prefix = symbol.replace("USDT", "")
        bundle["features"][symbol] = feature_payload["features_by_symbol"][symbol]["features"]
        bundle["boosters"][symbol] = lgb.Booster(model_file=str(model_dir / "models" / f"live_{prefix}.txt"))
        bundle["thresholds"][symbol] = metadata["live_report"][symbol]["thresholds"]
    MODEL_BUNDLE_CACHE[cache_key] = bundle
    return bundle


def signal_for_prediction(
    symbol: str,
    timeframe: str,
    probability_up: float,
    thresholds: dict[str, Any],
    decision_time: pd.Timestamp,
    last_kline_time: pd.Timestamp,
    model_dir: Path,
) -> dict[str, Any] | None:
    minutes = timeframe_minutes(timeframe)
    if not base.aligned_decision_time_mask(pd.DatetimeIndex([decision_time]), minutes)[0]:
        raise RuntimeError(f"{timeframe} decision_time is not aligned to a Polymarket event start: {decision_time}")
    short_threshold = float(thresholds["short_threshold"])
    long_threshold = float(thresholds["long_threshold"])
    if probability_up >= long_threshold:
        side = "BUY"
        confidence = probability_up
    elif probability_up <= short_threshold:
        side = "SELL"
        confidence = 1.0 - probability_up
    else:
        return None

    compact_symbol = normalize_symbol(symbol)
    timestamp = decision_time.tz_convert("UTC").isoformat()
    signal_id = f"lgbm-{timeframe}-{compact_symbol.replace('-', '')}-{decision_time.strftime('%Y%m%dT%H%M%SZ')}"
    return {
        "signal_id": signal_id,
        "symbol": compact_symbol,
        "timeframe": timeframe,
        "signal": side,
        "confidence": float(confidence),
        "timestamp": timestamp,
        "decision_time": timestamp,
        "last_kline_time": last_kline_time.tz_convert("UTC").isoformat(),
        "prob_up": float(probability_up),
        "short_threshold": short_threshold,
        "long_threshold": long_threshold,
        "model_dir": str(model_dir),
    }


def generate_signals(args: argparse.Namespace) -> dict[str, Any]:
    symbols = base.parse_symbols(args.symbols)
    model_dirs = parse_model_dirs(args.model_dir)
    now = datetime.now(timezone.utc)
    signals = []
    diagnostics = []
    dataset_start = latest_dataset_start(args.data_root, symbols, args.lookback_days)

    for timeframe, model_dir in sorted(model_dirs.items(), key=lambda item: timeframe_minutes(item[0])):
        minutes = timeframe_minutes(timeframe)
        bundle = load_model_bundle(model_dir, symbols)
        model_feature_set = bundle["metadata"].get("feature_set", args.feature_set)
        frame = build_live_feature_frame(args.data_root, symbols, minutes, dataset_start, model_feature_set)
        if frame.empty:
            raise RuntimeError(f"No live feature rows generated for {timeframe}.")
        row = frame.iloc[[-1]]
        decision_time = pd.Timestamp(row.index[-1])
        age_seconds = (pd.Timestamp(now) - decision_time).total_seconds()
        stale = args.max_data_age_seconds > 0 and age_seconds > args.max_data_age_seconds

        for symbol in symbols:
            columns = bundle["features"][symbol]
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise RuntimeError(f"{timeframe} {symbol} missing feature columns: {missing[:10]}")
            probability_up = float(bundle["boosters"][symbol].predict(row[columns])[0])
            signal = None
            if not stale:
                signal = signal_for_prediction(
                    symbol,
                    timeframe,
                    probability_up,
                    bundle["thresholds"][symbol],
                    decision_time,
                    pd.Timestamp(row["last_kline_time"].iloc[0]),
                    model_dir,
                )
            if signal is not None:
                signals.append(signal)
            diagnostics.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "feature_set": model_feature_set,
                    "decision_time": decision_time.tz_convert("UTC").isoformat(),
                    "age_seconds": float(age_seconds),
                    "stale": bool(stale),
                    "prob_up": probability_up,
                    "short_threshold": float(bundle["thresholds"][symbol]["short_threshold"]),
                    "long_threshold": float(bundle["thresholds"][symbol]["long_threshold"]),
                    "signal": None if signal is None else signal["signal"],
                }
            )

    return {
        "generated_at": now.isoformat(),
        "source": "lightgbm_live_signal_writer",
        "signals": sorted(signals, key=lambda item: (item["timestamp"], item["symbol"], item["timeframe"])),
        "diagnostics": diagnostics,
    }


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def parse_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Polymarket/HiBT signals from latest LightGBM live models.")
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data_oos"))
    parser.add_argument("--signal-file", type=Path, default=LIVE_DIR / "signals.json")
    parser.add_argument("--model-dir", action="append", default=[])
    parser.add_argument("--symbols", default=",".join(base.SYMBOLS))
    parser.add_argument("--feature-set", choices=base.FEATURE_SETS, default="v1", help="Fallback when model metadata has no feature_set.")
    parser.add_argument("--lookback-days", type=float, default=1.0)
    parser.add_argument("--max-data-age-seconds", type=float, default=240.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        payload = generate_signals(args)
        write_payload(args.signal_file, payload)
        print(
            f"wrote {len(payload['signals'])} signals to {args.signal_file} "
            f"generated_at={payload['generated_at']}",
            flush=True,
        )
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
