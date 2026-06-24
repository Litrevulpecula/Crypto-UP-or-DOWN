#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import lightgbm as lgb
import numpy as np
import pandas as pd


LIVE_DIR = Path(__file__).resolve().parent
if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))

import lightgbm_5m_direction_btc_eth as base  # noqa: E402
from log_colors import colorize_line  # noqa: E402


DEFAULT_MODEL_DIRS = {
    "3m": LIVE_DIR / "models_3m",
    "5m": LIVE_DIR / "models_5m",
    "15m": LIVE_DIR / "models_15m",
}
MODEL_BUNDLE_CACHE: dict[tuple[Path, tuple[str, ...]], dict[str, Any]] = {}
BASE_OPEN_TIME_CACHE: dict[Path, pd.Timestamp | None] = {}
READ_1M_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "is_missing",
]


def normalize_symbol(value: str) -> str:
    compact = value.strip().upper().replace("_", "").replace("-", "").replace("/", "")
    if compact in {"BTCUSDT", "BTC"}:
        return "BTC-USDT"
    if compact in {"ETHUSDT", "ETH"}:
        return "ETH-USDT"
    if compact.endswith("USDT") and len(compact) > 4:
        return f"{compact[:-4]}-USDT"
    return value.strip().upper()


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
    live_rows_by_path: dict[Path, dict[int, dict[str, str]]] | None = None,
) -> pd.DataFrame:
    if feature_set not in base.FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {base.FEATURE_SETS}.")
    market_feature_set = (
        "enhanced_price_position"
        if feature_set
        in {
            "v1_price_position",
            "v1_sessions_price_position",
            "v1_sessions_price_position_phase",
            "v1_sessions_price_position_phase_btc_external",
            "v1_sessions_price_position_phase_eth_external",
            "v1_sessions_price_position_phase_peer_external",
        }
        else "enhanced"
    )
    spot_frames = {}
    futures_frames = {}
    common_index: pd.DatetimeIndex | None = None

    for symbol in symbols:
        futures = read_live_1m(data_root, base.FUTURES_MARKET, symbol, sample_minutes, dataset_start, live_rows_by_path)
        spot = read_live_1m(data_root, base.SPOT_MARKET, symbol, sample_minutes, dataset_start, live_rows_by_path)
        spot_frames[symbol] = spot
        futures_frames[symbol] = futures
        symbol_index = spot.index.intersection(futures.index)
        common_index = symbol_index if common_index is None else common_index.intersection(symbol_index)

    if common_index is None or common_index.empty:
        raise RuntimeError("No common 1m timestamps were found for live feature generation.")
    common_index = common_index.sort_values()

    parts = []
    spot_closes = {symbol: spot_frames[symbol].loc[common_index, "close"] for symbol in symbols}
    for symbol in symbols:
        prefix = symbol.replace("USDT", "")
        spot = spot_frames[symbol].loc[common_index]
        futures = futures_frames[symbol].loc[common_index]
        symbol_parts = [
            base.add_market_features(spot, f"{prefix}_spot", market_feature_set),
            base.add_spot_futures_features(spot, futures, prefix, market_feature_set),
            spot[["close"]].rename(columns={"close": f"{prefix}_spot_close"}),
        ]
        if feature_set == "v1_sessions_price_position_phase_btc_external":
            symbol_parts.append(base.add_external_symbol_features(spot_closes, symbol, "BTCUSDT"))
        if feature_set == "v1_sessions_price_position_phase_eth_external":
            symbol_parts.append(base.add_external_symbol_features(spot_closes, symbol, "ETHUSDT"))
        if feature_set == "v1_sessions_price_position_phase_peer_external":
            peer_symbol = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT"}.get(symbol)
            if peer_symbol is not None:
                symbol_parts.append(base.add_external_symbol_features(spot_closes, symbol, peer_symbol))
        parts.extend(symbol_parts)

    frame = pd.concat(parts, axis=1)
    frame, _removed = base.drop_exact_duplicate_columns(frame)
    frame = pd.concat([frame, base.add_time_features(frame.index, feature_set)], axis=1)
    frame["last_kline_time"] = frame.index
    frame.index = frame.index + pd.Timedelta(minutes=1)
    frame.index.name = "decision_time"
    frame = frame.loc[base.aligned_decision_time_mask(frame.index, sample_minutes)]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    all_nan_columns = [column for column in frame.columns if frame[column].isna().all()]
    if all_nan_columns:
        frame = frame.drop(columns=all_nan_columns)
    return frame.dropna()


def read_live_1m(
    data_root: Path,
    market: str,
    symbol: str,
    sample_minutes: int,
    dataset_start: pd.Timestamp | None,
    live_rows_by_path: dict[Path, dict[int, dict[str, str]]] | None,
) -> pd.DataFrame:
    path = data_root / market / symbol / "1m.csv.gz"
    warmup_minutes = max(base.WINDOWS + base.ROLLING_WINDOWS) + sample_minutes + 5
    live_path = path.with_name("1m_live.csv")
    rows = None if live_rows_by_path is None else live_rows_by_path.get(live_path)
    if rows is not None and live_rows_cover_warmup(rows, warmup_minutes):
        return frame_from_live_rows(rows, warmup_minutes, dataset_start)
    return base.read_1m(path, dataset_start, target_horizon_minutes=sample_minutes)


def live_rows_cover_warmup(rows: dict[int, dict[str, str]], warmup_minutes: int) -> bool:
    if len(rows) < warmup_minutes + 1:
        return False
    tail = sorted(rows)[-(warmup_minutes + 1) :]
    return all((right - left) == 60_000 for left, right in zip(tail, tail[1:]))


def frame_from_live_rows(
    rows: dict[int, dict[str, str]],
    warmup_minutes: int,
    dataset_start: pd.Timestamp | None,
) -> pd.DataFrame:
    keys = sorted(rows)[-(warmup_minutes + 1) :]
    records = [{column: rows[key].get(column, "") for column in READ_1M_COLUMNS} for key in keys]
    frame = pd.DataFrame.from_records(records, columns=READ_1M_COLUMNS)
    frame["time"] = pd.to_datetime(pd.to_numeric(frame["open_time"], errors="coerce"), unit="ms", utc=True)
    frame = frame.drop(columns=["open_time"]).set_index("time").sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame["is_missing"] = pd.to_numeric(frame["is_missing"], errors="coerce")
    frame = frame.loc[frame["is_missing"].eq(0)].drop(columns=["is_missing"])
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if dataset_start is not None:
        warmup_start = dataset_start - pd.Timedelta(minutes=warmup_minutes)
        frame = frame.loc[frame.index >= warmup_start]
    return frame.dropna()


def frame_for_timeframe(frame: pd.DataFrame, sample_minutes: int, target_decision_time: pd.Timestamp | None) -> pd.DataFrame:
    if target_decision_time is not None and target_decision_time in frame.index:
        return frame.loc[[target_decision_time]]
    aligned = frame.loc[base.aligned_decision_time_mask(frame.index, sample_minutes)]
    return aligned.iloc[[-1]] if not aligned.empty else aligned


def latest_dataset_start(
    data_root: Path,
    symbols: tuple[str, ...],
    lookback_days: float | None,
    prefer_live_overlay: bool = True,
) -> pd.Timestamp | None:
    if lookback_days is None or lookback_days <= 0:
        return None
    latest: pd.Timestamp | None = None
    for market in (base.SPOT_MARKET, base.FUTURES_MARKET):
        for symbol in symbols:
            path = data_root / market / symbol / "1m.csv.gz"
            current = latest_open_time(path, prefer_live_overlay)
            if current is None:
                continue
            latest = current if latest is None else min(latest, current)
    if latest is None:
        return None
    return latest - pd.Timedelta(days=float(lookback_days))


def latest_open_time(path: Path, prefer_live_overlay: bool = True) -> pd.Timestamp | None:
    live_latest = read_latest_open_time(path.with_name("1m_live.csv"), cache=False)
    if prefer_live_overlay and live_latest is not None:
        return live_latest
    base_latest = read_latest_open_time(path, cache=True)
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
        raise RuntimeError(f"{timeframe} decision_time is not aligned to the event start: {decision_time}")
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


def generate_signals(
    args: argparse.Namespace,
    target_decision_times: dict[str, pd.Timestamp] | None = None,
) -> dict[str, Any]:
    symbols = base.parse_symbols(args.symbols)
    model_dirs = parse_model_dirs(args.model_dir)
    if target_decision_times is None:
        selected_model_dirs = model_dirs
    else:
        selected_model_dirs = {
            timeframe: model_dirs[timeframe]
            for timeframe in target_decision_times
            if timeframe in model_dirs
        }
    signals = []
    diagnostics = []
    dataset_start = getattr(args, "dataset_start", None)
    if dataset_start is None:
        dataset_start = latest_dataset_start(args.data_root, symbols, args.lookback_days)
    frame_cache: dict[str, pd.DataFrame] = {}

    for timeframe, model_dir in sorted(selected_model_dirs.items(), key=lambda item: timeframe_minutes(item[0])):
        minutes = timeframe_minutes(timeframe)
        bundle = load_model_bundle(model_dir, symbols)
        model_feature_set = bundle["metadata"].get("feature_set", args.feature_set)
        frame = frame_cache.get(model_feature_set)
        if frame is None:
            max_minutes = max(timeframe_minutes(item) for item in selected_model_dirs)
            frame = build_live_feature_frame(
                args.data_root,
                symbols,
                max_minutes,
                dataset_start,
                model_feature_set,
                getattr(args, "live_rows_by_path", None),
            )
            frame_cache[model_feature_set] = frame
        target_decision_time = None if target_decision_times is None else target_decision_times.get(timeframe)
        row = frame_for_timeframe(frame, minutes, target_decision_time)
        if row.empty:
            raise RuntimeError(f"No live feature rows generated for {timeframe}.")
        decision_time = pd.Timestamp(row.index[-1])
        target_ready = target_decision_time is None or decision_time == target_decision_time

        for symbol in symbols:
            columns = bundle["features"][symbol]
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise RuntimeError(f"{timeframe} {symbol} missing feature columns: {missing[:10]}")
            probability_up = float(bundle["boosters"][symbol].predict(row[columns])[0])
            signal = None
            if target_ready:
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
                    "target_decision_time": None
                    if target_decision_time is None
                    else target_decision_time.tz_convert("UTC").isoformat(),
                    "decision_time_matches_target": bool(target_ready),
                    "prob_up": probability_up,
                    "short_threshold": float(bundle["thresholds"][symbol]["short_threshold"]),
                    "long_threshold": float(bundle["thresholds"][symbol]["long_threshold"]),
                    "signal": None if signal is None else signal["signal"],
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
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


def due_decision_times(timeframes: list[str], decision_time: pd.Timestamp) -> dict[str, pd.Timestamp]:
    due = {}
    index = pd.DatetimeIndex([decision_time])
    for timeframe in timeframes:
        if base.aligned_decision_time_mask(index, timeframe_minutes(timeframe))[0]:
            due[timeframe] = decision_time
    return due


def live_overlay_covers_decision(
    data_root: Path,
    market: str,
    symbol: str,
    decision_time: pd.Timestamp,
    warmup_minutes: int,
) -> bool:
    path = data_root / market / symbol / "1m_live.csv"
    if not path.exists():
        return False
    target_open_time = decision_time - pd.Timedelta(minutes=1)
    target_open_ms = int(target_open_time.timestamp() * 1000)
    open_times = pd.read_csv(path, usecols=["open_time"])["open_time"]
    open_times = pd.to_numeric(open_times, errors="coerce").dropna().astype("int64").sort_values()
    tail = open_times[open_times <= target_open_ms].tail(warmup_minutes + 1)
    if len(tail) < warmup_minutes + 1:
        return False
    return bool(tail.iloc[-1] == target_open_ms and tail.diff().dropna().eq(60_000).all())


def target_data_ready(
    args: argparse.Namespace,
    symbols: tuple[str, ...],
    target_decision_times: dict[str, pd.Timestamp],
) -> tuple[bool, list[str]]:
    missing = []
    for timeframe, decision_time in target_decision_times.items():
        warmup_minutes = max(base.WINDOWS + base.ROLLING_WINDOWS) + timeframe_minutes(timeframe) + 5
        for symbol in symbols:
            for market in (base.SPOT_MARKET, base.FUTURES_MARKET):
                if not live_overlay_covers_decision(args.data_root, market, symbol, decision_time, warmup_minutes):
                    missing.append(f"{timeframe}:{market}:{symbol}")
    return not missing, missing


def write_and_log_payload(path: Path, payload: dict[str, Any], extra: str = "") -> float:
    start = time.perf_counter()
    write_payload(path, payload)
    write_ms = (time.perf_counter() - start) * 1000.0
    suffix = f" {extra}" if extra else ""
    message = (
        f"wrote {len(payload['signals'])} signals to {path} "
        f"generated_at={payload['generated_at']} write_ms={write_ms:.3f}{suffix}"
    )
    print(colorize_line(message), flush=True)
    return write_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate event signals from latest LightGBM live models.")
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data_oos"))
    parser.add_argument("--signal-file", type=Path, default=LIVE_DIR / "signals.json")
    parser.add_argument("--model-dir", action="append", default=[])
    parser.add_argument("--symbols", default=",".join(base.SYMBOLS))
    parser.add_argument("--feature-set", choices=base.FEATURE_SETS, default="v1", help="Fallback when model metadata has no feature_set.")
    parser.add_argument("--lookback-days", type=float, default=1.0)
    args = parser.parse_args()

    payload = generate_signals(args)
    write_and_log_payload(args.signal_file, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
