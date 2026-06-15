#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

SYMBOLS = ("BTCUSDT", "ETHUSDT")
SPOT_MARKET = "binance_spot_klines"
FUTURES_MARKET = "binance_um_futures_klines"
WINDOWS = (1, 2, 3, 5, 10, 15, 30, 60, 120, 240)
ROLLING_WINDOWS = (5, 10, 30, 60, 120)
PRICE_POSITION_WINDOWS = (5, 10, 15, 30, 60, 120, 240)
PAYOFF_WIN = 0.8
PAYOFF_LOSS = -1.0
DEFAULT_TARGET_HORIZON_MINUTES = 5
TARGET_HORIZON_MINUTES = DEFAULT_TARGET_HORIZON_MINUTES
SAMPLE_MINUTES = 5
TIME_FEATURES = (
    "sin_hour",
    "cos_hour",
    "sin_dayofweek",
    "cos_dayofweek",
    "session_asia",
    "session_europe",
    "session_us",
    "session_asia_europe_overlap",
    "session_europe_us_overlap",
)
SYMBOL_COLORS = {"BTC": "#F59119", "ETH": "#647DEB"}
FEATURE_SETS = ("v1", "v1_sessions", "v1_price_position", "v1_sessions_price_position")
FEATURE_SET_DESCRIPTIONS = {
    "v1": (
        "v1 current baseline with finite-difference acceleration, KAMA location/velocity/acceleration, "
        "volatility/order-flow ratios/interactions, futures-vs-spot ratios/interactions, "
        "spot multi-window returns, same-symbol futures-vs-spot basis/ratio/interaction features, "
        "and exact duplicate feature pruning with no duplicated raw futures market features"
    ),
    "v1_sessions": (
        "v1 plus UTC trading-session indicators: Asia 00-08, Europe 07-16, US 13-22, "
        "and Asia/Europe plus Europe/US overlap flags"
    ),
    "v1_price_position": (
        "v1 plus spot close position within rolling high/low ranges over "
        "5, 10, 15, 30, 60, 120, and 240 minute windows"
    ),
    "v1_sessions_price_position": (
        "v1 plus UTC trading-session indicators and spot close position within rolling high/low ranges over "
        "5, 10, 15, 30, 60, 120, and 240 minute windows"
    ),
}
def parse_symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not symbols:
        raise ValueError("--symbols must contain at least one symbol.")
    return symbols


def read_1m(
    path: Path,
    dataset_start: pd.Timestamp | None = None,
    target_horizon_minutes: int = DEFAULT_TARGET_HORIZON_MINUTES,
    include_live_overlay: bool = True,
) -> pd.DataFrame:
    columns = [
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
    live_path = path.with_name("1m_live.csv")
    warmup_minutes = max(WINDOWS + ROLLING_WINDOWS) + target_horizon_minutes + 5
    live_frame = pd.read_csv(live_path, usecols=columns) if include_live_overlay and live_path.exists() else None
    if live_frame is not None and live_frame_covers_warmup(live_frame, warmup_minutes):
        frame = live_frame.sort_values("open_time").tail(warmup_minutes + 1)
    else:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=columns)
        if live_frame is not None:
            frame = pd.concat([frame, live_frame], ignore_index=True)
    frame["time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame.drop(columns=["open_time"]).set_index("time").sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.loc[frame["is_missing"].eq(0)].drop(columns=["is_missing"])
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if dataset_start is not None:
        warmup_start = dataset_start - pd.Timedelta(minutes=warmup_minutes)
        frame = frame.loc[frame.index >= warmup_start]
    return frame.dropna()


def resolve_1m_path(root: Path, market: str, symbol: str) -> Path:
    gz_path = root / market / symbol / "1m.csv.gz"
    if gz_path.exists():
        return gz_path
    csv_path = root / market / symbol / "1m.csv"
    if csv_path.exists():
        return csv_path
    return gz_path


def live_frame_covers_warmup(frame: pd.DataFrame, warmup_minutes: int) -> bool:
    if frame.empty or len(frame) < warmup_minutes + 1:
        return False
    open_times = pd.to_numeric(frame["open_time"], errors="coerce").dropna().astype("int64")
    tail = open_times.sort_values().tail(warmup_minutes + 1)
    return bool(tail.diff().dropna().eq(60_000).all())


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=window)
    return (series - rolling.mean()) / rolling.std()


def kama(series: pd.Series, er_window: int, fast_period: int = 2, slow_period: int = 30) -> pd.Series:
    values = series.to_numpy(dtype=float)
    change = series.diff(er_window).abs()
    volatility = series.diff().abs().rolling(er_window, min_periods=er_window).sum()
    efficiency_ratio = safe_div(change, volatility).fillna(0.0).clip(0.0, 1.0)
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc).pow(2).to_numpy(dtype=float)

    result = np.full(len(series), np.nan, dtype=float)
    if len(series) <= er_window:
        return pd.Series(result, index=series.index)

    start = er_window
    result[start] = values[start]
    for idx in range(start + 1, len(values)):
        if not np.isfinite(values[idx]) or not np.isfinite(result[idx - 1]) or not np.isfinite(smoothing_constant[idx]):
            continue
        result[idx] = result[idx - 1] + smoothing_constant[idx] * (values[idx] - result[idx - 1])
    return pd.Series(result, index=series.index)


def feature_velocity(log_price: pd.Series, window: int) -> pd.Series:
    return log_price.diff(window) / float(window)


def finite_difference_acceleration(log_price: pd.Series, fast_window: int, slow_window: int) -> pd.Series:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")
    return (feature_velocity(log_price, fast_window) - feature_velocity(log_price, slow_window)) / float(
        slow_window - fast_window
    )


def drop_exact_duplicate_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    kept_columns: list[str] = []
    fingerprints: dict[bytes, list[str]] = {}
    removed: list[dict[str, str]] = []

    for column in frame.columns:
        if column in kept_columns:
            removed.append({"removed": column, "duplicate_of": column, "reason": "duplicate_name"})
            continue

        series = frame[column]
        values_hash = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype=np.uint64, copy=False)
        digest = hashlib.blake2b(values_hash.tobytes(), digest_size=16)
        digest.update(str(series.dtype).encode("utf-8"))
        key = digest.digest()

        duplicate_of = None
        for candidate in fingerprints.get(key, []):
            if series.equals(frame[candidate]):
                duplicate_of = candidate
                break

        if duplicate_of is None:
            kept_columns.append(column)
            fingerprints.setdefault(key, []).append(column)
        else:
            removed.append({"removed": column, "duplicate_of": duplicate_of, "reason": "identical_values"})

    if not removed:
        return frame, removed
    return frame.loc[:, kept_columns], removed


def add_market_features(frame: pd.DataFrame, prefix: str, feature_set: str = "v1") -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    include_enhanced = feature_set in {"enhanced", "enhanced_price_position"}
    include_price_position = feature_set == "enhanced_price_position"
    open_ = frame["open"]
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]
    quote_volume = frame["quote_volume"]
    count = frame["count"]
    buy_volume_ratio = safe_div(frame["taker_buy_volume"], volume)
    buy_quote_ratio = safe_div(frame["taker_buy_quote_volume"], quote_volume)
    quote_volume_per_trade = safe_div(quote_volume, count)
    taker_quote_volume_per_trade = safe_div(frame["taker_buy_quote_volume"], count)
    vwap = safe_div(quote_volume, volume)
    taker_buy_vwap = safe_div(frame["taker_buy_quote_volume"], frame["taker_buy_volume"])
    log_close = np.log(close)
    log_volume = np.log(volume.replace(0, np.nan))
    log_quote_volume = np.log(quote_volume.replace(0, np.nan))
    log_ret_1m = log_close.diff()
    bar_range = high / low - 1.0
    body_size = (close / open_ - 1.0).abs()
    upper_shadow = high / pd.concat([open_, close], axis=1).max(axis=1) - 1.0
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) / low - 1.0
    close_position_in_bar = safe_div(close - low, high - low)

    out[f"{prefix}_log_ret_1m"] = log_ret_1m
    out[f"{prefix}_high_low_range"] = bar_range
    out[f"{prefix}_body_size"] = body_size
    out[f"{prefix}_upper_shadow"] = upper_shadow
    out[f"{prefix}_lower_shadow"] = lower_shadow
    out[f"{prefix}_close_position_in_bar"] = close_position_in_bar
    for window in WINDOWS:
        if window != 1:
            out[f"{prefix}_log_ret_{window}m"] = log_close.diff(window)

    out[f"{prefix}_taker_buy_volume_ratio"] = buy_volume_ratio
    out[f"{prefix}_taker_buy_quote_ratio"] = buy_quote_ratio
    if include_enhanced:
        buy_imbalance = 2.0 * buy_volume_ratio - 1.0
        out[f"{prefix}_taker_buy_quote_volume_ratio_diff"] = buy_quote_ratio - buy_volume_ratio
        out[f"{prefix}_vwap_dist"] = vwap / close - 1.0
        out[f"{prefix}_taker_buy_vwap_dist"] = taker_buy_vwap / close - 1.0
        out[f"{prefix}_ret_x_buy_imbalance_1m"] = log_ret_1m * buy_imbalance
        realized_vol_by_window: dict[int, pd.Series] = {}
        quote_volume_mean_by_window: dict[int, pd.Series] = {}
        buy_ratio_mean_by_window: dict[int, pd.Series] = {}
    else:
        realized_vol_by_window = {}
        quote_volume_mean_by_window = {}
        buy_ratio_mean_by_window = {}

    if include_price_position:
        for window in PRICE_POSITION_WINDOWS:
            rolling_high = high.rolling(window, min_periods=window).max()
            rolling_low = low.rolling(window, min_periods=window).min()
            position = safe_div(close - rolling_low, rolling_high - rolling_low)
            out[f"{prefix}_close_position_{window}m"] = position.clip(0.0, 1.0)

    for window in ROLLING_WINDOWS:
        rolling_high = high.rolling(window, min_periods=window).max()
        rolling_low = low.rolling(window, min_periods=window).min()
        kama_value = kama(close, window)
        sma = close.rolling(window, min_periods=window).mean()

        out[f"{prefix}_ret_std_{window}m"] = log_ret_1m.rolling(window, min_periods=window).std()
        out[f"{prefix}_volume_z_{window}m"] = zscore(volume, window)
        out[f"{prefix}_quote_volume_z_{window}m"] = zscore(quote_volume, window)
        out[f"{prefix}_count_z_{window}m"] = zscore(count, window)
        out[f"{prefix}_range_mean_{window}m"] = bar_range.rolling(window, min_periods=window).mean()
        out[f"{prefix}_close_vs_sma_{window}m"] = close / sma - 1.0
        close_vs_kama = close / kama_value - 1.0
        out[f"{prefix}_close_vs_kama_{window}m"] = close_vs_kama
        if include_enhanced:
            out[f"{prefix}_close_vs_kama_z_{window}m"] = zscore(close_vs_kama, window)
            out[f"{prefix}_kama_velocity_{window}m"] = feature_velocity(np.log(kama_value), window)
        out[f"{prefix}_rolling_high_dist_{window}m"] = close / rolling_high - 1.0
        out[f"{prefix}_rolling_low_dist_{window}m"] = close / rolling_low - 1.0
        realized_vol = log_ret_1m.pow(2).rolling(window, min_periods=window).sum().pow(0.5)
        out[f"{prefix}_realized_vol_{window}m"] = realized_vol
        if include_enhanced:
            realized_vol_by_window[window] = realized_vol
            quote_volume_mean_by_window[window] = quote_volume.rolling(window, min_periods=window).mean()
            buy_ratio_mean_by_window[window] = buy_volume_ratio.rolling(window, min_periods=window).mean()
            up_vol = log_ret_1m.clip(lower=0.0).pow(2).rolling(window, min_periods=window).sum().pow(0.5)
            down_vol = log_ret_1m.clip(upper=0.0).pow(2).rolling(window, min_periods=window).sum().pow(0.5)
            out[f"{prefix}_upside_realized_vol_{window}m"] = up_vol
            out[f"{prefix}_downside_realized_vol_{window}m"] = down_vol
            out[f"{prefix}_realized_vol_balance_{window}m"] = safe_div(up_vol - down_vol, up_vol + down_vol)
            out[f"{prefix}_ret_vol_scaled_{window}m"] = safe_div(
                log_close.diff(window), out[f"{prefix}_realized_vol_{window}m"]
            )
            out[f"{prefix}_volume_ret_corr_{window}m"] = log_ret_1m.rolling(
                window, min_periods=window
            ).corr(log_volume.diff())
            out[f"{prefix}_quote_volume_per_trade_z_{window}m"] = zscore(quote_volume_per_trade, window)
            out[f"{prefix}_taker_quote_volume_per_trade_z_{window}m"] = zscore(
                taker_quote_volume_per_trade, window
            )
        if window in {5, 30, 60}:
            out[f"{prefix}_buy_ratio_mean_{window}m"] = buy_volume_ratio.rolling(window, min_periods=window).mean()
        if window in {30, 60}:
            out[f"{prefix}_buy_ratio_z_{window}m"] = zscore(buy_volume_ratio, window)

    if include_enhanced:
        for window in (2, 3, 5, 10, 15, 30, 60):
            out[f"{prefix}_ret_velocity_{window}m"] = feature_velocity(log_close, window)
            out[f"{prefix}_volume_velocity_{window}m"] = feature_velocity(log_volume, window)
            out[f"{prefix}_quote_volume_velocity_{window}m"] = feature_velocity(log_quote_volume, window)

        for fast_window, slow_window in ((2, 10), (5, 15), (10, 30), (15, 60)):
            out[f"{prefix}_ret_accel_{fast_window}_{slow_window}m"] = finite_difference_acceleration(
                log_close, fast_window, slow_window
            )
            out[f"{prefix}_volume_accel_{fast_window}_{slow_window}m"] = finite_difference_acceleration(
                log_volume, fast_window, slow_window
            )

        for fast_window, slow_window in ((5, 10), (10, 30), (30, 60), (60, 120)):
            out[f"{prefix}_kama_accel_{fast_window}_{slow_window}m"] = (
                out[f"{prefix}_kama_velocity_{fast_window}m"] - out[f"{prefix}_kama_velocity_{slow_window}m"]
            ) / float(slow_window - fast_window)

        for fast_window, slow_window in ((5, 30), (10, 60), (30, 120)):
            out[f"{prefix}_realized_vol_ratio_{fast_window}_{slow_window}m"] = safe_div(
                realized_vol_by_window[fast_window], realized_vol_by_window[slow_window]
            ) - 1.0
            out[f"{prefix}_quote_volume_mean_ratio_{fast_window}_{slow_window}m"] = safe_div(
                quote_volume_mean_by_window[fast_window], quote_volume_mean_by_window[slow_window]
            ) - 1.0
            out[f"{prefix}_buy_ratio_mean_diff_{fast_window}_{slow_window}m"] = (
                buy_ratio_mean_by_window[fast_window] - buy_ratio_mean_by_window[slow_window]
            )

        for window in (5, 30, 60):
            buy_mean = buy_volume_ratio.rolling(window, min_periods=window).mean()
            buy_z = zscore(buy_volume_ratio, window)
            out[f"{prefix}_buy_pressure_x_ret_{window}m"] = (2.0 * buy_mean - 1.0) * log_close.diff(window)
            out[f"{prefix}_buy_pressure_x_quote_volume_z_{window}m"] = buy_z * out[
                f"{prefix}_quote_volume_z_{window}m"
            ]

    return out


def add_spot_futures_features(spot: pd.DataFrame, futures: pd.DataFrame, prefix: str, feature_set: str = "v1") -> pd.DataFrame:
    out = pd.DataFrame(index=spot.index)
    include_enhanced = feature_set in {"enhanced", "enhanced_price_position"}
    spot_close = spot["close"]
    futures_close = futures["close"]
    basis = futures_close / spot_close - 1.0
    spot_log = np.log(spot_close)
    futures_log = np.log(futures_close)

    futures_spot_volume_ratio = safe_div(futures["volume"], spot["volume"])
    futures_spot_quote_volume_ratio = safe_div(futures["quote_volume"], spot["quote_volume"])
    futures_spot_count_ratio = safe_div(futures["count"], spot["count"])
    spot_buy_ratio = safe_div(spot["taker_buy_volume"], spot["volume"])
    futures_buy_ratio = safe_div(futures["taker_buy_volume"], futures["volume"])
    futures_spot_buy_ratio_diff = futures_buy_ratio - spot_buy_ratio

    out[f"{prefix}_futures_basis"] = basis
    out[f"{prefix}_futures_spot_volume_ratio"] = futures_spot_volume_ratio
    out[f"{prefix}_futures_spot_buy_ratio_diff"] = futures_spot_buy_ratio_diff
    for window in (1, 5, 15, 60):
        out[f"{prefix}_basis_change_{window}m"] = basis.diff(window)
        out[f"{prefix}_futures_spot_ret_spread_{window}m"] = futures_log.diff(window) - spot_log.diff(window)
    if include_enhanced:
        for window in (5, 15, 30, 60):
            out[f"{prefix}_basis_z_{window}m"] = zscore(basis, window)
            out[f"{prefix}_futures_spot_volume_ratio_z_{window}m"] = zscore(futures_spot_volume_ratio, window)
            out[f"{prefix}_futures_spot_quote_volume_ratio_z_{window}m"] = zscore(
                futures_spot_quote_volume_ratio, window
            )
            out[f"{prefix}_futures_spot_count_ratio_z_{window}m"] = zscore(futures_spot_count_ratio, window)
        for fast_window, slow_window in ((1, 5), (5, 15), (15, 60)):
            out[f"{prefix}_basis_accel_{fast_window}_{slow_window}m"] = (
                basis.diff(fast_window) / float(fast_window) - basis.diff(slow_window) / float(slow_window)
            ) / float(slow_window - fast_window)
        for window in (5, 15, 60):
            ret_spread = out[f"{prefix}_futures_spot_ret_spread_{window}m"]
            spot_realized_vol = spot_log.diff().pow(2).rolling(window, min_periods=window).sum().pow(0.5)
            futures_realized_vol = futures_log.diff().pow(2).rolling(window, min_periods=window).sum().pow(0.5)
            out[f"{prefix}_basis_vol_scaled_{window}m"] = safe_div(basis, spot_realized_vol)
            out[f"{prefix}_futures_spot_realized_vol_ratio_{window}m"] = safe_div(
                futures_realized_vol, spot_realized_vol
            ) - 1.0
            out[f"{prefix}_basis_x_ret_spread_{window}m"] = basis * ret_spread
            out[f"{prefix}_buy_diff_x_ret_spread_{window}m"] = futures_spot_buy_ratio_diff * ret_spread
    return out


def add_time_features(index: pd.DatetimeIndex, feature_set: str = "v1") -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["sin_hour"] = np.sin(2.0 * np.pi * index.hour / 24.0)
    out["cos_hour"] = np.cos(2.0 * np.pi * index.hour / 24.0)
    out["sin_dayofweek"] = np.sin(2.0 * np.pi * index.dayofweek / 7.0)
    out["cos_dayofweek"] = np.cos(2.0 * np.pi * index.dayofweek / 7.0)
    if feature_set in {"v1_sessions", "v1_sessions_price_position"}:
        hour = index.hour + index.minute / 60.0
        out["session_asia"] = ((hour >= 0.0) & (hour < 8.0)).astype(np.float32)
        out["session_europe"] = ((hour >= 7.0) & (hour < 16.0)).astype(np.float32)
        out["session_us"] = ((hour >= 13.0) & (hour < 22.0)).astype(np.float32)
        out["session_asia_europe_overlap"] = ((hour >= 7.0) & (hour < 8.0)).astype(np.float32)
        out["session_europe_us_overlap"] = ((hour >= 13.0) & (hour < 16.0)).astype(np.float32)
    return out


def aligned_decision_time_mask(decision_time: pd.DatetimeIndex, sample_minutes: int) -> np.ndarray:
    seconds = decision_time.hour * 3600 + decision_time.minute * 60 + decision_time.second
    return (
        (seconds % (sample_minutes * 60) == 0)
        & (decision_time.microsecond == 0)
        & (decision_time.nanosecond == 0)
    )


def sample_aligned_frame(frame: pd.DataFrame, sample_minutes: int) -> pd.DataFrame:
    return frame.loc[aligned_decision_time_mask(frame.index, sample_minutes)]


def build_dataset(
    data_root: Path,
    symbols: tuple[str, ...],
    target_symbols: tuple[str, ...] | None,
    market: str,
    futures_market: str,
    dataset_start: pd.Timestamp | None,
    target_horizon_minutes: int,
    sample_minutes: int,
    target_tie_policy: str,
    feature_set: str = "v1",
    include_live_overlay: bool = True,
) -> pd.DataFrame:
    if target_tie_policy not in {"drop", "down", "expected"}:
        raise ValueError("target_tie_policy must be 'drop', 'down', or 'expected'.")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}.")
    market_feature_set = (
        "enhanced_price_position"
        if feature_set in {"v1_price_position", "v1_sessions_price_position"}
        else "enhanced"
    )
    modeled_symbols = target_symbols or symbols
    missing_targets = sorted(set(modeled_symbols) - set(symbols))
    if missing_targets:
        raise ValueError(f"target_symbols must be a subset of symbols. Missing: {missing_targets}")
    spot_frames = {}
    futures_frames = {}
    price_source_by_symbol = {}
    all_nan_feature_columns_removed = []
    common_index: pd.DatetimeIndex | None = None

    for symbol in symbols:
        futures_path = resolve_1m_path(data_root, futures_market, symbol)
        spot_path = resolve_1m_path(data_root, market, symbol)
        futures = read_1m(futures_path, dataset_start, target_horizon_minutes, include_live_overlay)
        spot = read_1m(spot_path, dataset_start, target_horizon_minutes, include_live_overlay)
        price_source_by_symbol[symbol] = market
        spot_frames[symbol] = spot
        futures_frames[symbol] = futures
        symbol_index = spot.index.intersection(futures.index)
        common_index = symbol_index if common_index is None else common_index.intersection(symbol_index)

    if common_index is None or common_index.empty:
        raise RuntimeError("No common 1m timestamps were found.")
    common_index = common_index.sort_values()

    parts = []
    for symbol in modeled_symbols:
        prefix = symbol.replace("USDT", "")
        spot = spot_frames[symbol].loc[common_index]
        futures = futures_frames[symbol].loc[common_index]
        symbol_parts = [
            add_market_features(spot, f"{prefix}_spot", market_feature_set),
            spot[["close"]].rename(columns={"close": f"{prefix}_spot_close"}),
        ]
        if price_source_by_symbol[symbol] == market:
            symbol_parts.insert(1, add_spot_futures_features(spot, futures, prefix, market_feature_set))
        parts.extend(symbol_parts)

    parts = [part for part in parts if part is not None]
    frame = pd.concat(parts, axis=1)
    frame = pd.concat([frame, add_time_features(frame.index, feature_set)], axis=1)
    frame, duplicate_feature_columns = drop_exact_duplicate_columns(frame)

    horizon_steps = target_horizon_minutes
    exact_horizon_time = pd.Series(frame.index, index=frame.index).shift(-horizon_steps)
    has_exact_horizon = exact_horizon_time == frame.index + pd.Timedelta(minutes=target_horizon_minutes)
    target_tie_counts = {}
    for symbol in modeled_symbols:
        prefix = symbol.replace("USDT", "")
        current_close = frame[f"{prefix}_spot_close"]
        future_close = current_close.shift(-horizon_steps).where(has_exact_horizon)
        tie = future_close.eq(current_close) & future_close.notna()
        frame[f"{prefix}_future_close"] = future_close
        if target_tie_policy == "drop":
            frame[f"{prefix}_target"] = np.where(
                future_close > current_close,
                1.0,
                np.where(future_close < current_close, 0.0, np.nan),
            )
        else:
            # For expected tie payoff, this 0 label is only a placeholder.
            # Training/evaluation weights remove tie rows from direction fitting.
            frame[f"{prefix}_target"] = (future_close > current_close).astype(float)
        frame[f"{prefix}_target_tie"] = tie.astype(np.int8)
        target_tie_counts[symbol] = int(tie.sum())

    frame["last_kline_time"] = frame.index
    frame.index = frame.index + pd.Timedelta(minutes=1)
    frame.index.name = "decision_time"
    frame = frame.loc[aligned_decision_time_mask(frame.index, sample_minutes)]
    if dataset_start is not None:
        frame = frame.loc[frame.index >= dataset_start]

    frame = frame.replace([np.inf, -np.inf], np.nan)
    all_nan_feature_columns_removed = [column for column in frame.columns if frame[column].isna().all()]
    if all_nan_feature_columns_removed:
        frame = frame.drop(columns=all_nan_feature_columns_removed)
    frame = frame.dropna()
    for column in frame.columns:
        if column == "last_kline_time":
            continue
        if column.endswith("_target"):
            frame[column] = frame[column].astype(np.int8)
        elif pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].astype(np.float32)
    frame.attrs["symbols"] = symbols
    frame.attrs["target_symbols"] = modeled_symbols
    frame.attrs["price_source_by_symbol"] = price_source_by_symbol
    frame.attrs["target_tie_counts"] = target_tie_counts
    frame.attrs["target_tie_policy"] = target_tie_policy
    frame.attrs["feature_set"] = feature_set
    frame.attrs["duplicate_feature_columns_removed"] = duplicate_feature_columns
    frame.attrs["all_nan_feature_columns_removed"] = all_nan_feature_columns_removed
    return frame


def feature_columns_for_symbol(dataset: pd.DataFrame, symbol: str) -> list[str]:
    prefix = symbol.replace("USDT", "")
    excluded_suffixes = ("_target", "_target_tie", "_future_close", "_spot_close")
    columns = []
    for column in dataset.columns:
        if column == "last_kline_time" or column.endswith(excluded_suffixes):
            continue
        if column.startswith(f"{prefix}_") or column in TIME_FEATURES:
            columns.append(column)
    return columns


def auc_or_nan(actual: np.ndarray, score: np.ndarray, tie_mask: np.ndarray | None = None) -> float:
    actual = np.asarray(actual)
    score = np.asarray(score)
    if tie_mask is not None:
        keep = ~np.asarray(tie_mask, dtype=bool)
        actual = actual[keep]
        score = score[keep]
    if len(np.unique(actual)) < 2:
        return float("nan")
    return float(roc_auc_score(actual.astype(int), score.astype(float)))


def annualization_factor(sample_minutes: int = SAMPLE_MINUTES) -> float:
    return float(np.sqrt(365 * 24 * 60 / sample_minutes))


def sharpe_ratio(pnl: np.ndarray, sample_minutes: int = SAMPLE_MINUTES) -> float:
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) < 2:
        return float("-inf")
    std = pnl.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return float("inf") if pnl.mean() > 0 else float("-inf")
    return float((pnl.mean() / std) * annualization_factor(sample_minutes))


def validation_sample_minutes(args: argparse.Namespace) -> int:
    return int(getattr(args, "validation_sample_minutes", SAMPLE_MINUTES))


def apply_double_thresholds(
    probability: np.ndarray,
    actual_up: np.ndarray,
    short_threshold: float,
    long_threshold: float,
    win_payoff: float,
    loss_payoff: float,
    tie_mask: np.ndarray | None = None,
    tie_payoff: float | None = None,
):
    position = np.where(probability >= long_threshold, 1, np.where(probability <= short_threshold, -1, 0)).astype(np.int8)
    active = position != 0
    if tie_mask is None:
        tie_mask = np.zeros(len(position), dtype=bool)
    else:
        tie_mask = np.asarray(tie_mask, dtype=bool)
    wins = (((position == 1) & (actual_up == 1)) | ((position == -1) & (actual_up == 0))) & ~tie_mask
    pnl = np.where(active, np.where(wins, win_payoff, loss_payoff), 0.0).astype(float)
    if tie_payoff is not None:
        pnl = np.where(active & tie_mask, tie_payoff, pnl).astype(float)
    return position, active, wins, pnl


def target_tie_payoff(args: argparse.Namespace) -> float | None:
    if getattr(args, "target_tie_policy", "down") == "expected":
        return float((args.win_payoff + args.loss_payoff) / 2.0)
    return None


def target_tie_mask(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    column = f"{prefix}_target_tie"
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    return frame[column].to_numpy(dtype=bool)


def target_sample_weight(frame: pd.DataFrame, prefix: str, args: argparse.Namespace) -> np.ndarray:
    weight = np.ones(len(frame), dtype=float)
    if getattr(args, "target_tie_policy", "down") == "expected":
        weight[target_tie_mask(frame, prefix)] = 0.0
    return weight


def train_sample_weight(frame: pd.DataFrame, prefix: str, args: argparse.Namespace) -> np.ndarray:
    weight = target_sample_weight(frame, prefix, args)
    half_life_days = getattr(args, "train_time_decay_half_life_days", None)
    if half_life_days is None:
        return weight
    if frame.empty:
        return weight
    reference_time = frame.index.max()
    age_days = (reference_time - frame.index).total_seconds().to_numpy(dtype=float) / (24.0 * 60.0 * 60.0)
    decay = np.power(0.5, age_days / float(half_life_days))
    weight = weight * decay
    positive = weight > 0
    if positive.any():
        weight[positive] /= float(weight[positive].mean())
    return weight


def payoff_tie_mask(frame: pd.DataFrame, prefix: str, args: argparse.Namespace) -> np.ndarray:
    if getattr(args, "target_tie_policy", "down") == "expected":
        return target_tie_mask(frame, prefix)
    return np.zeros(len(frame), dtype=bool)


def choose_thresholds(
    probability: np.ndarray,
    actual_up: np.ndarray,
    args: argparse.Namespace,
    tie_mask: np.ndarray | None = None,
) -> dict:
    probability = np.asarray(probability, dtype=float)
    actual_up = np.asarray(actual_up, dtype=np.int8)
    if tie_mask is None:
        tie_mask = np.zeros(len(actual_up), dtype=bool)
    else:
        tie_mask = np.asarray(tie_mask, dtype=bool)
    min_trades = max(2, int(np.ceil(len(actual_up) * args.min_trade_fraction)))
    grid = np.linspace(0.0, 1.0, args.threshold_grid_size)
    tie_payoff = target_tie_payoff(args)
    best = {
        "mode": "double",
        "threshold": 0.5,
        "short_threshold": 0.0,
        "long_threshold": 1.0,
        "sharpe": float("-inf"),
        "pnl": 0.0,
        "trades": 0,
        "trade_fraction": 0.0,
        "win_rate": 0.0,
        "tie_trades": 0,
        "tie_trade_fraction": 0.0,
        "min_trades": int(min_trades),
    }
    for short_threshold in grid:
        for long_threshold in grid:
            if short_threshold >= long_threshold:
                continue
            _, active, wins, pnl = apply_double_thresholds(
                probability,
                actual_up,
                short_threshold,
                long_threshold,
                args.win_payoff,
                args.loss_payoff,
                tie_mask,
                tie_payoff,
            )
            trades = int(active.sum())
            if trades < min_trades:
                continue
            current_sharpe = sharpe_ratio(pnl, validation_sample_minutes(args))
            total_pnl = float(pnl.sum())
            trade_fraction = float(trades / len(actual_up))
            active_ties = active & tie_mask
            strict_active = active & ~tie_mask
            if (
                current_sharpe > best["sharpe"]
                or (current_sharpe == best["sharpe"] and total_pnl > best["pnl"])
                or (
                    current_sharpe == best["sharpe"]
                    and total_pnl == best["pnl"]
                    and trade_fraction > best["trade_fraction"]
                )
            ):
                best.update(
                    {
                        "threshold": float((short_threshold + long_threshold) / 2.0),
                        "short_threshold": float(short_threshold),
                        "long_threshold": float(long_threshold),
                        "sharpe": current_sharpe,
                        "pnl": total_pnl,
                        "trades": trades,
                        "trade_fraction": trade_fraction,
                        "win_rate": float(wins[strict_active].mean()) if strict_active.any() else 0.0,
                        "tie_trades": int(active_ties.sum()),
                        "tie_trade_fraction": float(active_ties.mean()),
                    }
                )
    if not np.isfinite(best["sharpe"]):
        raise RuntimeError("Threshold search failed; reduce --min-trade-fraction or --threshold-grid-size.")
    return best


def adjust_extreme_thresholds(thresholds: dict, probability: np.ndarray) -> dict:
    adjusted = thresholds.copy()
    finite_probability = np.asarray(probability, dtype=float)
    finite_probability = finite_probability[np.isfinite(finite_probability)]
    if len(finite_probability) == 0:
        return adjusted
    min_probability = float(finite_probability.min())
    max_probability = float(finite_probability.max())
    adjusted["raw_short_threshold"] = float(thresholds["short_threshold"])
    adjusted["raw_long_threshold"] = float(thresholds["long_threshold"])
    adjusted["short_threshold_was_extreme"] = bool(thresholds["short_threshold"] <= 0.0)
    adjusted["long_threshold_was_extreme"] = bool(thresholds["long_threshold"] >= 1.0)
    if adjusted["short_threshold_was_extreme"]:
        adjusted["short_threshold"] = min_probability
    if adjusted["long_threshold_was_extreme"]:
        adjusted["long_threshold"] = max_probability
    adjusted["threshold"] = float((adjusted["short_threshold"] + adjusted["long_threshold"]) / 2.0)
    adjusted["validation_probability_min"] = min_probability
    adjusted["validation_probability_max"] = max_probability
    return adjusted
