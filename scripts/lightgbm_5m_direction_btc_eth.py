#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import optuna

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SYMBOLS = ("BTCUSDT", "ETHUSDT")
SPOT_MARKET = "binance_spot_klines"
FUTURES_MARKET = "binance_um_futures_klines"
WINDOWS = (1, 2, 3, 5, 10, 15, 30, 60, 120, 240)
ROLLING_WINDOWS = (5, 10, 30, 60, 120)
PRICE_POSITION_WINDOWS = (5, 10, 15, 30, 60, 120, 240)
EXTERNAL_SYMBOL_WINDOWS = (1, 3, 5, 15, 30, 60)
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
    "sin_minute",
    "cos_minute",
)
SYMBOL_COLORS = {"BTC": "#F59119", "ETH": "#647DEB", "SOL": "#14F195"}
FEATURE_SETS = (
    "v1",
    "v1_sessions",
    "v1_price_position",
    "v1_sessions_price_position",
    "v1_sessions_price_position_phase",
    "v1_sessions_price_position_phase_btc_external",
    "v1_sessions_price_position_phase_eth_external",
    "v1_sessions_price_position_phase_peer_external",
)
FEATURE_SET_DESCRIPTIONS = {
    "v1": (
        "v1 original baseline with finite-difference acceleration, KAMA location/velocity/acceleration, "
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
    "v1_sessions_price_position_phase": (
        "previous baseline: v1_sessions_price_position plus minute-of-hour sine/cosine phase features"
    ),
    "v1_sessions_price_position_phase_btc_external": (
        "v1_sessions_price_position_phase plus BTC spot return and target-minus-BTC relative return features"
    ),
    "v1_sessions_price_position_phase_eth_external": (
        "v1_sessions_price_position_phase plus ETH spot return and target-minus-ETH relative return features"
    ),
    "v1_sessions_price_position_phase_peer_external": (
        "new baseline: v1_sessions_price_position_phase plus BTC<->ETH peer spot return and relative return features"
    ),
}
def parse_symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not symbols:
        raise ValueError("--symbols must contain at least one symbol.")
    return symbols


def target_symbols_for_run(symbols: tuple[str, ...], value: str | None) -> tuple[str, ...]:
    if value is None:
        return symbols
    target_symbols = parse_symbols(value)
    missing = sorted(set(target_symbols) - set(symbols))
    if missing:
        raise ValueError(f"--target-symbols must be a subset of --symbols. Missing from --symbols: {missing}")
    return target_symbols


def read_1m(
    path: Path,
    dataset_start: pd.Timestamp | None = None,
    target_horizon_minutes: int = DEFAULT_TARGET_HORIZON_MINUTES,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
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
    frame = pd.read_csv(path, usecols=columns)
    frame["time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame.drop(columns=["open_time"]).set_index("time").sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.loc[frame["is_missing"].eq(0)].drop(columns=["is_missing"])
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if dataset_start is not None:
        warmup_start = dataset_start - pd.Timedelta(
            minutes=max(WINDOWS + ROLLING_WINDOWS) + target_horizon_minutes + 5
        )
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


def column_owner(column: str) -> str:
    return column.split("_", 1)[0] if "_" in column else ""


def drop_exact_duplicate_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    kept_columns: list[str] = []
    fingerprints: dict[tuple[str, bytes], list[str]] = {}
    removed: list[dict[str, str]] = []

    for column in frame.columns:
        if column in kept_columns:
            removed.append({"removed": column, "duplicate_of": column, "reason": "duplicate_name"})
            continue

        series = frame[column]
        values_hash = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype=np.uint64, copy=False)
        digest = hashlib.blake2b(values_hash.tobytes(), digest_size=16)
        digest.update(str(series.dtype).encode("utf-8"))
        key = (column_owner(column), digest.digest())

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


def add_external_symbol_features(
    closes: dict[str, pd.Series],
    target_symbol: str,
    external_symbol: str,
) -> pd.DataFrame:
    prefix = target_symbol.replace("USDT", "")
    target_log = np.log(closes[target_symbol])
    out = pd.DataFrame(index=closes[target_symbol].index)
    if target_symbol == external_symbol or external_symbol not in closes:
        return out
    external_prefix = external_symbol.replace("USDT", "")
    external_log = np.log(closes[external_symbol])
    for window in EXTERNAL_SYMBOL_WINDOWS:
        external_ret = external_log.diff(window)
        out[f"{prefix}_{external_prefix}_spot_log_ret_{window}m"] = external_ret
        out[f"{prefix}_relative_{external_prefix}_spot_log_ret_{window}m"] = target_log.diff(window) - external_ret
    return out


def add_time_features(index: pd.DatetimeIndex, feature_set: str = "v1") -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["sin_hour"] = np.sin(2.0 * np.pi * index.hour / 24.0)
    out["cos_hour"] = np.cos(2.0 * np.pi * index.hour / 24.0)
    out["sin_dayofweek"] = np.sin(2.0 * np.pi * index.dayofweek / 7.0)
    out["cos_dayofweek"] = np.cos(2.0 * np.pi * index.dayofweek / 7.0)
    if feature_set in {
        "v1_sessions",
        "v1_sessions_price_position",
        "v1_sessions_price_position_phase",
        "v1_sessions_price_position_phase_btc_external",
        "v1_sessions_price_position_phase_eth_external",
        "v1_sessions_price_position_phase_peer_external",
    }:
        hour = index.hour + index.minute / 60.0
        out["session_asia"] = ((hour >= 0.0) & (hour < 8.0)).astype(np.float32)
        out["session_europe"] = ((hour >= 7.0) & (hour < 16.0)).astype(np.float32)
        out["session_us"] = ((hour >= 13.0) & (hour < 22.0)).astype(np.float32)
        out["session_asia_europe_overlap"] = ((hour >= 7.0) & (hour < 8.0)).astype(np.float32)
        out["session_europe_us_overlap"] = ((hour >= 13.0) & (hour < 16.0)).astype(np.float32)
    if feature_set in {
        "v1_sessions_price_position_phase",
        "v1_sessions_price_position_phase_btc_external",
        "v1_sessions_price_position_phase_eth_external",
        "v1_sessions_price_position_phase_peer_external",
    }:
        out["sin_minute"] = np.sin(2.0 * np.pi * index.minute / 60.0)
        out["cos_minute"] = np.cos(2.0 * np.pi * index.minute / 60.0)
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
) -> pd.DataFrame:
    if target_tie_policy not in {"drop", "down", "expected"}:
        raise ValueError("target_tie_policy must be 'drop', 'down', or 'expected'.")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}.")
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
        futures = read_1m(futures_path, dataset_start, target_horizon_minutes)
        spot = read_1m(spot_path, dataset_start, target_horizon_minutes)
        price_source_by_symbol[symbol] = market
        spot_frames[symbol] = spot
        futures_frames[symbol] = futures
        symbol_index = spot.index.intersection(futures.index)
        common_index = symbol_index if common_index is None else common_index.intersection(symbol_index)

    if common_index is None or common_index.empty:
        raise RuntimeError("No common 1m timestamps were found.")
    common_index = common_index.sort_values()

    parts = []
    spot_closes = {symbol: spot_frames[symbol].loc[common_index, "close"] for symbol in symbols}
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
        if feature_set == "v1_sessions_price_position_phase_btc_external":
            symbol_parts.append(add_external_symbol_features(spot_closes, symbol, "BTCUSDT"))
        if feature_set == "v1_sessions_price_position_phase_eth_external":
            symbol_parts.append(add_external_symbol_features(spot_closes, symbol, "ETHUSDT"))
        if feature_set == "v1_sessions_price_position_phase_peer_external":
            peer_symbol = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT"}.get(symbol)
            if peer_symbol is not None:
                symbol_parts.append(add_external_symbol_features(spot_closes, symbol, peer_symbol))
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


def test_sample_minutes(args: argparse.Namespace) -> int:
    return int(getattr(args, "test_sample_minutes", SAMPLE_MINUTES))


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


def sample_weight_stats(weight: np.ndarray) -> dict[str, float]:
    positive = np.asarray(weight, dtype=float)
    positive = positive[positive > 0]
    if len(positive) == 0:
        return {
            "weight_min": float("nan"),
            "weight_mean": float("nan"),
            "weight_max": float("nan"),
            "weight_effective_rows": 0.0,
        }
    total = float(positive.sum())
    squared = float(np.square(positive).sum())
    return {
        "weight_min": float(positive.min()),
        "weight_mean": float(positive.mean()),
        "weight_max": float(positive.max()),
        "weight_effective_rows": float((total * total) / squared) if squared > 0 else 0.0,
    }


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


def smooth_thresholds(thresholds: dict, previous_thresholds: dict | None, alpha: float) -> dict:
    smoothed = thresholds.copy()
    if previous_thresholds is None or alpha >= 1.0:
        smoothed["used_short_threshold"] = float(thresholds["short_threshold"])
        smoothed["used_long_threshold"] = float(thresholds["long_threshold"])
    elif alpha <= 0.0:
        smoothed["used_short_threshold"] = float(previous_thresholds["used_short_threshold"])
        smoothed["used_long_threshold"] = float(previous_thresholds["used_long_threshold"])
    else:
        old_weight = 1.0 - alpha
        smoothed["used_short_threshold"] = float(
            old_weight * previous_thresholds["used_short_threshold"] + alpha * thresholds["short_threshold"]
        )
        smoothed["used_long_threshold"] = float(
            old_weight * previous_thresholds["used_long_threshold"] + alpha * thresholds["long_threshold"]
        )
    if smoothed["used_short_threshold"] >= smoothed["used_long_threshold"]:
        midpoint = float((smoothed["used_short_threshold"] + smoothed["used_long_threshold"]) / 2.0)
        smoothed["used_short_threshold"] = np.nextafter(midpoint, -np.inf)
        smoothed["used_long_threshold"] = np.nextafter(midpoint, np.inf)
    smoothed["used_threshold"] = float((smoothed["used_short_threshold"] + smoothed["used_long_threshold"]) / 2.0)
    smoothed["threshold_ema_alpha"] = float(alpha)
    return smoothed


def apply_thresholds(
    probability: np.ndarray,
    actual_up: np.ndarray,
    thresholds: dict,
    args: argparse.Namespace,
    tie_mask: np.ndarray | None = None,
):
    return apply_double_thresholds(
        probability,
        actual_up,
        thresholds["used_short_threshold"],
        thresholds["used_long_threshold"],
        args.win_payoff,
        args.loss_payoff,
        tie_mask,
        target_tie_payoff(args),
    )


def fixed_model_params(args: argparse.Namespace) -> dict:
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


def optuna_search_split(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_start = dataset.index.min() + pd.Timedelta(days=args.initial_train_days)
    train = dataset.loc[dataset.index < test_start]
    validation_start = test_start - pd.Timedelta(days=args.validation_days)
    fit_train = train.loc[train.index < validation_start]
    validation = train.loc[train.index >= validation_start]
    if args.train_window_days is not None:
        fit_train = fit_train.loc[fit_train.index >= validation_start - pd.Timedelta(days=args.train_window_days)]
    fit_train = sample_aligned_frame(fit_train, args.train_sample_minutes)
    validation = sample_aligned_frame(validation, args.validation_sample_minutes)
    if fit_train.empty or validation.empty:
        split = max(1, int(len(train) * 0.8))
        fit_train = train.iloc[:split]
        validation = train.iloc[split:]
        if args.train_window_days is not None and not validation.empty:
            fit_train = fit_train.loc[fit_train.index >= validation.index.min() - pd.Timedelta(days=args.train_window_days)]
        fit_train = sample_aligned_frame(fit_train, args.train_sample_minutes)
        validation = sample_aligned_frame(validation, args.validation_sample_minutes)
    if fit_train.empty or validation.empty:
        raise RuntimeError("Not enough rows for Optuna train/validation split.")
    return fit_train, validation


def suggest_lgbm_params(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": trial.suggest_float("learning_rate", args.optuna_min_learning_rate, args.optuna_max_learning_rate, log=True),
        "num_leaves": trial.suggest_int("num_leaves", args.optuna_min_num_leaves, args.optuna_max_num_leaves, step=8),
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            args.optuna_min_child_samples,
            args.optuna_max_child_samples,
            step=20,
        ),
        "subsample": trial.suggest_float("subsample", args.optuna_min_subsample, args.optuna_max_subsample),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", args.optuna_min_colsample_bytree, args.optuna_max_colsample_bytree
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", args.optuna_min_reg_alpha, args.optuna_max_reg_alpha, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", args.optuna_min_reg_lambda, args.optuna_max_reg_lambda, log=True),
        "objective": "binary",
        "random_state": args.random_state,
        "verbosity": -1,
    }


def suggest_float_or_fixed(
    trial: optuna.Trial,
    name: str,
    low: float,
    high: float,
    center: float,
    log: bool = False,
) -> float:
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return float(center)
    if log and low <= 0:
        return float(center)
    return float(trial.suggest_float(name, low, high, log=log))


def suggest_lgbm_params_near(trial: optuna.Trial, args: argparse.Namespace, center_params: dict) -> dict:
    learning_rate_center = float(center_params["learning_rate"])
    learning_rate_low = max(args.optuna_min_learning_rate, learning_rate_center / args.fold_optuna_learning_rate_factor)
    learning_rate_high = min(args.optuna_max_learning_rate, learning_rate_center * args.fold_optuna_learning_rate_factor)

    num_leaves_center = int(center_params["num_leaves"])
    num_leaves_low = max(args.optuna_min_num_leaves, num_leaves_center - args.fold_optuna_num_leaves_radius)
    num_leaves_high = min(args.optuna_max_num_leaves, num_leaves_center + args.fold_optuna_num_leaves_radius)

    child_center = int(center_params["min_child_samples"])
    child_low = max(args.optuna_min_child_samples, child_center - args.fold_optuna_child_samples_radius)
    child_high = min(args.optuna_max_child_samples, child_center + args.fold_optuna_child_samples_radius)

    subsample_center = float(center_params["subsample"])
    subsample_low = max(args.optuna_min_subsample, subsample_center - args.fold_optuna_sample_radius)
    subsample_high = min(args.optuna_max_subsample, subsample_center + args.fold_optuna_sample_radius)

    colsample_center = float(center_params["colsample_bytree"])
    colsample_low = max(args.optuna_min_colsample_bytree, colsample_center - args.fold_optuna_sample_radius)
    colsample_high = min(args.optuna_max_colsample_bytree, colsample_center + args.fold_optuna_sample_radius)

    reg_alpha_center = max(float(center_params["reg_alpha"]), args.optuna_min_reg_alpha)
    reg_alpha_low = max(args.optuna_min_reg_alpha, reg_alpha_center / args.fold_optuna_regularization_factor)
    reg_alpha_high = min(args.optuna_max_reg_alpha, reg_alpha_center * args.fold_optuna_regularization_factor)

    reg_lambda_center = max(float(center_params["reg_lambda"]), args.optuna_min_reg_lambda)
    reg_lambda_low = max(args.optuna_min_reg_lambda, reg_lambda_center / args.fold_optuna_regularization_factor)
    reg_lambda_high = min(args.optuna_max_reg_lambda, reg_lambda_center * args.fold_optuna_regularization_factor)

    params = dict(center_params)
    params.update(
        {
            "n_estimators": int(center_params.get("n_estimators", args.n_estimators)),
            "learning_rate": suggest_float_or_fixed(
                trial,
                "learning_rate",
                learning_rate_low,
                learning_rate_high,
                learning_rate_center,
                log=True,
            ),
            "num_leaves": trial.suggest_int("num_leaves", num_leaves_low, num_leaves_high, step=8)
            if num_leaves_low < num_leaves_high
            else num_leaves_center,
            "min_child_samples": trial.suggest_int("min_child_samples", child_low, child_high, step=20)
            if child_low < child_high
            else child_center,
            "subsample": suggest_float_or_fixed(trial, "subsample", subsample_low, subsample_high, subsample_center),
            "subsample_freq": 1,
            "colsample_bytree": suggest_float_or_fixed(
                trial,
                "colsample_bytree",
                colsample_low,
                colsample_high,
                colsample_center,
            ),
            "reg_alpha": suggest_float_or_fixed(
                trial,
                "reg_alpha",
                reg_alpha_low,
                reg_alpha_high,
                reg_alpha_center,
                log=True,
            ),
            "reg_lambda": suggest_float_or_fixed(
                trial,
                "reg_lambda",
                reg_lambda_low,
                reg_lambda_high,
                reg_lambda_center,
                log=True,
            ),
            "objective": "binary",
            "random_state": args.random_state,
            "verbosity": -1,
        }
    )
    return params


def optimize_symbol_params(
    symbol: str,
    columns: list[str],
    fit_train: pd.DataFrame,
    validation: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    prefix = symbol.replace("USDT", "")
    y_fit = fit_train[f"{prefix}_target"].astype(int)
    fit_weight = train_sample_weight(fit_train, prefix, args)
    fit_weight_stats = sample_weight_stats(fit_weight)
    y_validation = validation[f"{prefix}_target"].astype(np.int8).to_numpy()
    validation_tie = payoff_tie_mask(validation, prefix, args)
    validation_weight = target_sample_weight(validation, prefix, args)
    if y_fit[fit_weight > 0].nunique() < 2 or len(np.unique(y_validation[~validation_tie])) < 2:
        raise RuntimeError(f"Not enough target class diversity for Optuna on {symbol}.")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm_params(trial, args)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            fit_train[columns],
            y_fit,
            sample_weight=fit_weight,
            eval_set=[(validation[columns], y_validation.astype(int))],
            eval_sample_weight=[validation_weight],
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
        )
        probability = model.predict_proba(validation[columns])[:, 1]
        thresholds = choose_thresholds(probability, y_validation, args, validation_tie)
        if args.optuna_metric == "auc":
            value = auc_or_nan(y_validation, probability, validation_tie)
        else:
            value = thresholds["sharpe"]
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or params["n_estimators"]))
        trial.set_user_attr("validation_auc", auc_or_nan(y_validation, probability, validation_tie))
        trial.set_user_attr("validation_thresholds", thresholds)
        trial.set_user_attr("validation_sharpe", thresholds["sharpe"])
        trial.set_user_attr("validation_pnl", thresholds["pnl"])
        return value

    sampler = optuna.samplers.TPESampler(seed=args.random_state)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(0, args.optuna_pruner_warmup))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.optuna_trials, timeout=args.optuna_timeout, show_progress_bar=False)
    best_params = fixed_model_params(args)
    best_params.update(study.best_trial.params)
    best_params["subsample_freq"] = 1
    best_params["objective"] = "binary"
    best_params["random_state"] = args.random_state
    best_params["verbosity"] = -1
    return {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "best_user_attrs": dict(study.best_trial.user_attrs),
        "metric": args.optuna_metric,
        "trials": len(study.trials),
        "fit_train_start": fit_train.index.min().isoformat(),
        "fit_train_end": fit_train.index.max().isoformat(),
        "validation_start": validation.index.min().isoformat(),
        "validation_end": validation.index.max().isoformat(),
        "fit_weight_stats": fit_weight_stats,
    }


def optimize_symbol_params_near(
    symbol: str,
    columns: list[str],
    fit_train: pd.DataFrame,
    validation: pd.DataFrame,
    center_params: dict,
    args: argparse.Namespace,
) -> dict:
    prefix = symbol.replace("USDT", "")
    y_fit = fit_train[f"{prefix}_target"].astype(int)
    fit_weight = train_sample_weight(fit_train, prefix, args)
    fit_weight_stats = sample_weight_stats(fit_weight)
    y_validation = validation[f"{prefix}_target"].astype(np.int8).to_numpy()
    validation_tie = payoff_tie_mask(validation, prefix, args)
    validation_weight = target_sample_weight(validation, prefix, args)
    if y_fit[fit_weight > 0].nunique() < 2 or len(np.unique(y_validation[~validation_tie])) < 2:
        raise RuntimeError(f"Not enough target class diversity for fold Optuna on {symbol}.")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm_params_near(trial, args, center_params)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            fit_train[columns],
            y_fit,
            sample_weight=fit_weight,
            eval_set=[(validation[columns], y_validation.astype(int))],
            eval_sample_weight=[validation_weight],
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
        )
        probability = model.predict_proba(validation[columns])[:, 1]
        thresholds = choose_thresholds(probability, y_validation, args, validation_tie)
        value = auc_or_nan(y_validation, probability, validation_tie) if args.optuna_metric == "auc" else thresholds["sharpe"]
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or params["n_estimators"]))
        trial.set_user_attr("validation_auc", auc_or_nan(y_validation, probability, validation_tie))
        trial.set_user_attr("validation_thresholds", thresholds)
        trial.set_user_attr("validation_sharpe", thresholds["sharpe"])
        trial.set_user_attr("validation_pnl", thresholds["pnl"])
        return value

    sampler = optuna.samplers.TPESampler(seed=args.random_state)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(0, args.optuna_pruner_warmup))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.fold_optuna_trials, timeout=args.fold_optuna_timeout, show_progress_bar=False)
    best_params = dict(center_params)
    best_params.update(study.best_trial.params)
    best_params["subsample_freq"] = 1
    best_params["objective"] = "binary"
    best_params["random_state"] = args.random_state
    best_params["verbosity"] = -1
    return {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "best_user_attrs": dict(study.best_trial.user_attrs),
        "metric": args.optuna_metric,
        "trials": len(study.trials),
        "fit_train_start": fit_train.index.min().isoformat(),
        "fit_train_end": fit_train.index.max().isoformat(),
        "validation_start": validation.index.min().isoformat(),
        "validation_end": validation.index.max().isoformat(),
        "fit_weight_stats": fit_weight_stats,
    }


def load_or_optimize_params(
    dataset: pd.DataFrame,
    columns_by_symbol: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, dict]:
    symbols = dataset.attrs.get("target_symbols", dataset.attrs["symbols"])
    if args.fixed_params_json is not None:
        with args.fixed_params_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {symbol: payload[symbol]["best_params"] for symbol in symbols}

    if args.optuna_trials <= 0:
        return {symbol: fixed_model_params(args) for symbol in symbols}

    fit_train, validation = optuna_search_split(dataset, args)
    report = {}
    params_by_symbol = {}
    for symbol in symbols:
        result = optimize_symbol_params(symbol, columns_by_symbol[symbol], fit_train, validation, args)
        report[symbol] = result
        params_by_symbol[symbol] = result["best_params"]
        print(
            f"optuna {symbol} metric={result['metric']} best={result['best_value']:.6g} "
            f"trials={result['trials']}",
            flush=True,
        )
    with (args.out_dir / "optuna_best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return params_by_symbol


def walk_forward(dataset: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    symbols = dataset.attrs.get("target_symbols", dataset.attrs["symbols"])
    columns_by_symbol = {symbol: feature_columns_for_symbol(dataset, symbol) for symbol in symbols}
    params_by_symbol = load_or_optimize_params(dataset, columns_by_symbol, args)
    with (args.out_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "features_by_symbol": {
                    symbol: {"feature_count": len(columns), "features": columns}
                    for symbol, columns in columns_by_symbol.items()
                },
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    predictions = []
    folds = []
    fold_optuna_records = []
    fold_id = 0
    previous_thresholds_by_symbol: dict[str, dict] = {}
    test_start = dataset.index.min() + pd.Timedelta(days=args.initial_train_days)
    end_time = dataset.index.max()

    while test_start <= end_time:
        test_end = min(test_start + pd.Timedelta(days=args.retrain_days), end_time + pd.Timedelta(minutes=sample_gap_minutes(dataset)))
        train = dataset.loc[dataset.index < test_start]
        test = dataset.loc[(dataset.index >= test_start) & (dataset.index < test_end)]
        test = sample_aligned_frame(test, args.test_sample_minutes)
        if train.empty or test.empty:
            test_start = test_end
            continue

        validation_start = test_start - pd.Timedelta(days=args.validation_days)
        fit_train = train.loc[train.index < validation_start]
        validation = train.loc[train.index >= validation_start]
        if args.train_window_days is not None:
            fit_train = fit_train.loc[fit_train.index >= validation_start - pd.Timedelta(days=args.train_window_days)]
        fit_train = sample_aligned_frame(fit_train, args.train_sample_minutes)
        validation = sample_aligned_frame(validation, args.validation_sample_minutes)
        if fit_train.empty or validation.empty:
            split = max(1, int(len(train) * 0.8))
            fit_train = train.iloc[:split]
            validation = train.iloc[split:]
            if args.train_window_days is not None and not validation.empty:
                fit_train = fit_train.loc[fit_train.index >= validation.index.min() - pd.Timedelta(days=args.train_window_days)]
            fit_train = sample_aligned_frame(fit_train, args.train_sample_minutes)
            validation = sample_aligned_frame(validation, args.validation_sample_minutes)
        if fit_train.empty or validation.empty:
            test_start = test_end
            continue

        fold_record = {
            "fold": fold_id,
            "fit_train_start": fit_train.index.min().isoformat(),
            "fit_train_end": fit_train.index.max().isoformat(),
            "validation_start": validation.index.min().isoformat(),
            "validation_end": validation.index.max().isoformat(),
            "test_start": test.index.min().isoformat(),
            "test_end": test.index.max().isoformat(),
            "fit_train_rows": int(len(fit_train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        }
        fold_predictions = pd.DataFrame(index=test.index)
        fold_predictions["fold"] = fold_id
        fold_predictions["last_kline_time"] = test["last_kline_time"].to_numpy()

        for symbol in symbols:
            prefix = symbol.replace("USDT", "")
            columns = columns_by_symbol[symbol]
            current_params = params_by_symbol[symbol]
            if args.fold_optuna_trials > 0:
                print(f"fold-optuna fold={fold_id} {symbol} trials={args.fold_optuna_trials}", flush=True)
                fold_optuna = optimize_symbol_params_near(
                    symbol,
                    columns,
                    fit_train,
                    validation,
                    params_by_symbol[symbol],
                    args,
                )
                current_params = fold_optuna["best_params"]
                fold_optuna_records.append(
                    {
                        "fold": fold_id,
                        "symbol": symbol,
                        **fold_optuna,
                    }
                )
                fold_record[f"{prefix}_fold_optuna_best_value"] = fold_optuna["best_value"]
                fold_record[f"{prefix}_fold_optuna_trials"] = fold_optuna["trials"]
                fold_record[f"{prefix}_fold_optuna_validation_auc"] = fold_optuna["best_user_attrs"].get(
                    "validation_auc"
                )
                fold_record[f"{prefix}_fold_optuna_validation_sharpe"] = fold_optuna["best_user_attrs"].get(
                    "validation_sharpe"
                )
                print(
                    f"fold-optuna fold={fold_id} {symbol} best={fold_optuna['best_value']:.6g}",
                    flush=True,
                )
            fit_weight = train_sample_weight(fit_train, prefix, args)
            fit_weight_stats = sample_weight_stats(fit_weight)
            validation_tie = payoff_tie_mask(validation, prefix, args)
            validation_weight = target_sample_weight(validation, prefix, args)
            model = lgb.LGBMClassifier(**current_params)
            model.fit(
                fit_train[columns],
                fit_train[f"{prefix}_target"].astype(int),
                sample_weight=fit_weight,
                eval_set=[(validation[columns], validation[f"{prefix}_target"].astype(int))],
                eval_sample_weight=[validation_weight],
                callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
            )
            validation_probability = model.predict_proba(validation[columns])[:, 1]
            validation_actual = validation[f"{prefix}_target"].to_numpy(dtype=np.int8)
            raw_thresholds = choose_thresholds(validation_probability, validation_actual, args, validation_tie)
            adjusted_thresholds = adjust_extreme_thresholds(raw_thresholds, validation_probability)
            thresholds = smooth_thresholds(
                adjusted_thresholds,
                previous_thresholds_by_symbol.get(symbol),
                args.threshold_ema_alpha,
            )
            previous_thresholds_by_symbol[symbol] = thresholds
            probability = model.predict_proba(test[columns])[:, 1]
            actual = test[f"{prefix}_target"].to_numpy(dtype=np.int8)
            test_tie = payoff_tie_mask(test, prefix, args)
            actual_test_tie = target_tie_mask(test, prefix)
            position, active, wins, pnl = apply_thresholds(probability, actual, thresholds, args, test_tie)
            prediction = (position == 1).astype(np.int8)
            model_path = models_dir / f"fold_{fold_id:03d}_{prefix}.txt"
            model.booster_.save_model(model_path)

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

            fold_record[f"{prefix}_auc"] = auc_or_nan(actual, probability, test_tie)
            fold_record[f"{prefix}_threshold_mode"] = thresholds["mode"]
            fold_record[f"{prefix}_raw_short_threshold"] = thresholds["raw_short_threshold"]
            fold_record[f"{prefix}_raw_long_threshold"] = thresholds["raw_long_threshold"]
            fold_record[f"{prefix}_short_threshold_was_extreme"] = thresholds["short_threshold_was_extreme"]
            fold_record[f"{prefix}_long_threshold_was_extreme"] = thresholds["long_threshold_was_extreme"]
            fold_record[f"{prefix}_adjusted_threshold"] = thresholds["threshold"]
            fold_record[f"{prefix}_adjusted_short_threshold"] = thresholds["short_threshold"]
            fold_record[f"{prefix}_adjusted_long_threshold"] = thresholds["long_threshold"]
            fold_record[f"{prefix}_threshold"] = thresholds["used_threshold"]
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
            fold_record[f"{prefix}_fit_weight_min"] = fit_weight_stats["weight_min"]
            fold_record[f"{prefix}_fit_weight_mean"] = fit_weight_stats["weight_mean"]
            fold_record[f"{prefix}_fit_weight_max"] = fit_weight_stats["weight_max"]
            fold_record[f"{prefix}_fit_weight_effective_rows"] = fit_weight_stats["weight_effective_rows"]
            fold_record[f"{prefix}_test_trades"] = int(active.sum())
            fold_record[f"{prefix}_test_trade_fraction"] = float(active.mean())
            fold_record[f"{prefix}_test_tie_trades"] = int((active & test_tie).sum())
            fold_record[f"{prefix}_test_tie_trade_fraction"] = float((active & test_tie).mean())
            strict_active = active & ~test_tie
            fold_record[f"{prefix}_win_rate"] = float(wins[strict_active].mean()) if strict_active.any() else float("nan")
            fold_record[f"{prefix}_pnl"] = float(pnl.sum())
            fold_record[f"{prefix}_test_sharpe"] = sharpe_ratio(pnl, test_sample_minutes(args))
            fold_record[f"{prefix}_best_iteration"] = int(
                model.best_iteration_ or current_params["n_estimators"]
            )
            fold_record[f"{prefix}_model_path"] = str(model_path)

        predictions.append(fold_predictions.reset_index(names="decision_time"))
        folds.append(fold_record)
        status_parts = []
        for symbol in symbols:
            prefix = symbol.replace("USDT", "")
            status_parts.append(
                f"{prefix}_thr=({fold_record[f'{prefix}_short_threshold']:.2f},"
                f"{fold_record[f'{prefix}_long_threshold']:.2f}) "
                f"raw=({fold_record[f'{prefix}_raw_short_threshold']:.2f},"
                f"{fold_record[f'{prefix}_raw_long_threshold']:.2f}) "
                f"trades={fold_record[f'{prefix}_test_trades']} pnl={fold_record[f'{prefix}_pnl']:.1f}"
            )
        print(
            f"fold={fold_id} test={test.index.min()}..{test.index.max()} rows={len(test)} "
            + " ".join(status_parts),
            flush=True,
        )
        fold_id += 1
        if args.max_folds is not None and fold_id >= args.max_folds:
            break
        test_start = test_end

    if not predictions:
        raise RuntimeError("No walk-forward predictions were generated.")
    pd.DataFrame(folds).to_csv(args.out_dir / "folds.csv", index=False)
    if fold_optuna_records:
        with (args.out_dir / "fold_optuna_params.json").open("w", encoding="utf-8") as handle:
            json.dump(fold_optuna_records, handle, indent=2)
            handle.write("\n")
    return pd.concat(predictions, ignore_index=True)


def sample_gap_minutes(dataset: pd.DataFrame) -> int:
    gap = dataset.index.to_series().diff().dropna().median()
    if pd.isna(gap):
        return SAMPLE_MINUTES
    return max(1, int(gap.total_seconds() // 60))


def summarize(predictions: pd.DataFrame, args: argparse.Namespace, symbols: tuple[str, ...]) -> dict:
    predictions = predictions.sort_values("decision_time").reset_index(drop=True)
    target_horizon_minutes = int(getattr(args, "target_horizon_minutes", DEFAULT_TARGET_HORIZON_MINUTES))
    summary = {
        "samples": int(len(predictions)),
        "symbols": list(symbols),
        "target": (
            f"strict spot close direction {target_horizon_minutes} minutes after last available 1m spot close; "
            f"ties policy={args.target_tie_policy}"
        ),
        "payoff": f"win +{args.win_payoff}, loss {args.loss_payoff}",
        "win_payoff": args.win_payoff,
        "loss_payoff": args.loss_payoff,
        "target_tie_payoff": target_tie_payoff(args),
        "feature_set": getattr(args, "feature_set", "v1"),
        "threshold_mode": "double",
        "threshold_objective": "walk-forward validation annualized Sharpe",
        "min_trade_fraction": args.min_trade_fraction,
        "train_window_days": getattr(args, "train_window_days", None),
        "train_time_decay_half_life_days": getattr(args, "train_time_decay_half_life_days", None),
        "train_sample_minutes": getattr(args, "train_sample_minutes", SAMPLE_MINUTES),
        "validation_sample_minutes": validation_sample_minutes(args),
        "test_sample_minutes": test_sample_minutes(args),
        "threshold_smoothing": f"ema old_weight={1.0 - args.threshold_ema_alpha:.6g}, new_weight={args.threshold_ema_alpha:.6g}",
        "threshold_ema_alpha": args.threshold_ema_alpha,
        "threshold_extreme_adjustment": "short raw 0 uses validation probability min; long raw 1 uses validation probability max before EMA smoothing",
    }
    equity = pd.DataFrame({"decision_time": predictions["decision_time"]})
    for symbol in symbols:
        prefix = symbol.replace("USDT", "")
        trades = predictions[f"{prefix}_is_trade"].astype(bool)
        wins = predictions[f"{prefix}_is_win"].astype(bool)
        ties = predictions.get(f"{prefix}_is_tie", pd.Series(0, index=predictions.index)).astype(bool)
        payoff_ties = ties if getattr(args, "target_tie_policy", "down") == "expected" else pd.Series(False, index=predictions.index)
        pnl = predictions[f"{prefix}_pnl"].astype(float)
        predictions[f"{prefix}_equity"] = pnl.cumsum()
        equity[prefix] = predictions[f"{prefix}_equity"]
        strict_trades = trades & ~payoff_ties
        summary[prefix] = {
            "samples": int(len(predictions)),
            "trades": int(trades.sum()),
            "trade_fraction": float(trades.mean()),
            "tie_trades": int((trades & ties).sum()),
            "tie_trade_fraction": float((trades & ties).mean()),
            "wins": int(wins[strict_trades].sum()),
            "losses": int((strict_trades & ~wins).sum()),
            "win_rate": float(wins[strict_trades].mean()) if strict_trades.any() else float("nan"),
            "auc": auc_or_nan(
                predictions[f"{prefix}_actual_up"].to_numpy(),
                predictions[f"{prefix}_prob_up"].to_numpy(),
                payoff_ties.to_numpy(dtype=bool) if getattr(args, "target_tie_policy", "down") == "expected" else None,
            ),
            "pnl": float(pnl.sum()),
            "annualized_sharpe": sharpe_ratio(pnl.to_numpy(), test_sample_minutes(args)),
            "final_equity": float(pnl.cumsum().iloc[-1]),
        }

    predictions.to_csv(args.out_dir / "predictions.csv", index=False)
    equity.to_csv(args.out_dir / "equity_curve.csv", index=False)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    fig, ax = plt.subplots(figsize=(14, 7))
    for symbol in symbols:
        prefix = symbol.replace("USDT", "")
        ax.plot(pd.to_datetime(equity["decision_time"]), equity[prefix], label=prefix, color=SYMBOL_COLORS.get(prefix))
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title(f"LightGBM {target_horizon_minutes}m Direction Walk-Forward Equity")
    ax.set_xlabel("Decision time")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.out_dir / "equity_curve.png", dpi=160)
    plt.close(fig)
    return summary


def parse_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean LightGBM BTC/ETH direction walk-forward.")
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data"))
    parser.add_argument("--market", default=SPOT_MARKET)
    parser.add_argument("--futures-market", default=FUTURES_MARKET)
    parser.add_argument("--out-dir", type=Path, default=Path("results/lightgbm_btc_eth_target5m_clean"))
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--target-symbols", default=None)
    parser.add_argument("--target-horizon-minutes", type=int, default=DEFAULT_TARGET_HORIZON_MINUTES)
    parser.add_argument("--target-tie-policy", choices=["expected", "drop", "down"], default="expected")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="v1_sessions_price_position_phase_peer_external")
    parser.add_argument("--dataset-start", default=None)
    parser.add_argument("--initial-train-days", type=int, default=365)
    parser.add_argument("--train-window-days", type=int, default=None)
    parser.add_argument(
        "--train-time-decay-half-life-days",
        type=float,
        default=365.0,
        help="Half-life for exponential fit-sample time decay. Use 0 to disable.",
    )
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--retrain-days", type=int, default=30)
    parser.add_argument("--train-sample-minutes", type=int, default=SAMPLE_MINUTES)
    parser.add_argument("--validation-sample-minutes", type=int, default=SAMPLE_MINUTES)
    parser.add_argument("--test-sample-minutes", type=int, default=SAMPLE_MINUTES)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--num-leaves", type=int, default=32)
    parser.add_argument("--min-child-samples", type=int, default=120)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=1e-3)
    parser.add_argument("--reg-lambda", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--win-payoff", type=float, default=PAYOFF_WIN)
    parser.add_argument("--loss-payoff", type=float, default=PAYOFF_LOSS)
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument("--min-trade-fraction", type=float, default=0.05)
    parser.add_argument("--threshold-ema-alpha", type=float, default=0.7)
    parser.add_argument("--optuna-trials", type=int, default=0, help="Trials per symbol. 0 disables Optuna.")
    parser.add_argument("--optuna-timeout", type=int, default=None, help="Optional Optuna timeout in seconds per symbol.")
    parser.add_argument("--optuna-metric", choices=["sharpe", "auc"], default="sharpe")
    parser.add_argument("--optuna-pruner-warmup", type=int, default=5)
    parser.add_argument("--fixed-params-json", type=Path, default=None)
    parser.add_argument("--fold-optuna-trials", type=int, default=0, help="Local Optuna trials per fold and symbol around initial params.")
    parser.add_argument("--fold-optuna-timeout", type=int, default=None, help="Optional local fold Optuna timeout in seconds per symbol.")
    parser.add_argument("--fold-optuna-learning-rate-factor", type=float, default=1.35)
    parser.add_argument("--fold-optuna-regularization-factor", type=float, default=3.0)
    parser.add_argument("--fold-optuna-num-leaves-radius", type=int, default=16)
    parser.add_argument("--fold-optuna-child-samples-radius", type=int, default=60)
    parser.add_argument("--fold-optuna-sample-radius", type=float, default=0.10)
    parser.add_argument("--optuna-min-estimators", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--optuna-max-estimators", type=int, default=None, help=argparse.SUPPRESS)
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

    symbols = parse_symbols(args.symbols)
    target_symbols = target_symbols_for_run(symbols, args.target_symbols)
    if not np.isfinite(args.win_payoff) or args.win_payoff <= 0:
        raise ValueError("--win-payoff must be a finite value > 0.")
    if not np.isfinite(args.loss_payoff) or args.loss_payoff >= 0:
        raise ValueError("--loss-payoff must be a finite value < 0.")
    if args.optuna_trials < 0:
        raise ValueError("--optuna-trials must be >= 0.")
    if args.fold_optuna_trials < 0:
        raise ValueError("--fold-optuna-trials must be >= 0.")
    if args.threshold_grid_size < 2:
        raise ValueError("--threshold-grid-size must be >= 2.")
    if args.train_window_days is not None and args.train_window_days <= 0:
        raise ValueError("--train-window-days must be > 0 when provided.")
    if args.train_time_decay_half_life_days is not None:
        if not np.isfinite(args.train_time_decay_half_life_days) or args.train_time_decay_half_life_days < 0:
            raise ValueError("--train-time-decay-half-life-days must be finite and >= 0.")
        if args.train_time_decay_half_life_days == 0:
            args.train_time_decay_half_life_days = None
    if args.validation_days <= 0:
        raise ValueError("--validation-days must be > 0.")
    if args.retrain_days <= 0:
        raise ValueError("--retrain-days must be > 0.")
    if args.target_horizon_minutes <= 0:
        raise ValueError("--target-horizon-minutes must be > 0.")
    sample_values = [args.train_sample_minutes, args.validation_sample_minutes, args.test_sample_minutes]
    if any(value <= 0 for value in sample_values):
        raise ValueError("sample-minute arguments must be > 0.")
    if not np.isfinite(args.min_trade_fraction) or not 0.0 <= args.min_trade_fraction <= 1.0:
        raise ValueError("--min-trade-fraction must be in [0, 1].")
    if not np.isfinite(args.threshold_ema_alpha) or not 0.0 <= args.threshold_ema_alpha <= 1.0:
        raise ValueError("--threshold-ema-alpha must be in [0, 1].")
    if args.optuna_min_num_leaves > args.optuna_max_num_leaves:
        raise ValueError("--optuna-min-num-leaves must be <= --optuna-max-num-leaves.")
    if args.optuna_min_child_samples > args.optuna_max_child_samples:
        raise ValueError("--optuna-min-child-samples must be <= --optuna-max-child-samples.")
    if not 0.0 < args.optuna_min_learning_rate <= args.optuna_max_learning_rate:
        raise ValueError("Optuna learning-rate bounds must satisfy 0 < min <= max.")
    if not 0.0 < args.optuna_min_subsample <= args.optuna_max_subsample <= 1.0:
        raise ValueError("Optuna subsample bounds must satisfy 0 < min <= max <= 1.")
    if not 0.0 < args.optuna_min_colsample_bytree <= args.optuna_max_colsample_bytree <= 1.0:
        raise ValueError("Optuna colsample_bytree bounds must satisfy 0 < min <= max <= 1.")
    if args.fold_optuna_trials > 0:
        if args.fold_optuna_learning_rate_factor <= 1.0:
            raise ValueError("--fold-optuna-learning-rate-factor must be > 1.")
        if args.fold_optuna_regularization_factor <= 1.0:
            raise ValueError("--fold-optuna-regularization-factor must be > 1.")
        if args.fold_optuna_num_leaves_radius < 0 or args.fold_optuna_child_samples_radius < 0:
            raise ValueError("fold Optuna integer radii must be >= 0.")
        if args.fold_optuna_sample_radius < 0:
            raise ValueError("--fold-optuna-sample-radius must be >= 0.")

    dataset_sample_minutes = math.gcd(math.gcd(args.train_sample_minutes, args.validation_sample_minutes), args.test_sample_minutes)
    dataset_start = parse_utc_timestamp(args.dataset_start)
    dataset = build_dataset(
        data_root=args.data_root,
        symbols=symbols,
        target_symbols=target_symbols,
        market=args.market,
        futures_market=args.futures_market,
        dataset_start=dataset_start,
        target_horizon_minutes=args.target_horizon_minutes,
        sample_minutes=dataset_sample_minutes,
        target_tie_policy=args.target_tie_policy,
        feature_set=args.feature_set,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stale_replay_info = args.out_dir / "replay_info.json"
    if stale_replay_info.exists():
        stale_replay_info.unlink()
    dataset_info = {
        "rows": int(len(dataset)),
        "start": dataset.index.min().isoformat(),
        "end": dataset.index.max().isoformat(),
        "symbols": list(symbols),
        "target_symbols": list(target_symbols),
        "data_root": str(args.data_root),
        "market": args.market,
        "futures_market": args.futures_market,
        "price_source_by_symbol": dataset.attrs.get("price_source_by_symbol", {}),
        "futures_features": True,
        "sample_minutes": dataset_sample_minutes,
        "feature_set": args.feature_set,
        "feature_set_description": FEATURE_SET_DESCRIPTIONS[args.feature_set],
        "duplicate_feature_columns_removed": dataset.attrs.get("duplicate_feature_columns_removed", []),
        "all_nan_feature_columns_removed": dataset.attrs.get("all_nan_feature_columns_removed", []),
        "train_sample_minutes": args.train_sample_minutes,
        "validation_sample_minutes": args.validation_sample_minutes,
        "test_sample_minutes": args.test_sample_minutes,
        "target_horizon_minutes": args.target_horizon_minutes,
        "target_alignment": (
            "decision_time uses last_kline_time = decision_time - 1 minute; "
            f"future_close is last_kline_time + {args.target_horizon_minutes} minutes"
        ),
        "target_definition": "strict future_close > current_close is UP, strict future_close < current_close is DOWN; ties have no direction label",
        "target_tie_policy": args.target_tie_policy,
        "target_tie_payoff": target_tie_payoff(args),
        "target_tie_counts_before_drop": dataset.attrs.get("target_tie_counts", {}),
        "initial_train_days": args.initial_train_days,
        "train_window_days": args.train_window_days,
        "train_time_decay_half_life_days": args.train_time_decay_half_life_days,
        "train_time_decay_weighting": (
            "none"
            if args.train_time_decay_half_life_days is None
            else "fit sample_weight *= 0.5 ** (age_days / half_life_days), normalized to positive-weight mean 1"
        ),
        "validation_days": args.validation_days,
        "retrain_days": args.retrain_days,
        "win_payoff": args.win_payoff,
        "loss_payoff": args.loss_payoff,
        "threshold_mode": "double",
        "threshold_grid_size": args.threshold_grid_size,
        "min_trade_fraction": args.min_trade_fraction,
        "threshold_smoothing": f"ema old_weight={1.0 - args.threshold_ema_alpha:.6g}, new_weight={args.threshold_ema_alpha:.6g}",
        "threshold_ema_alpha": args.threshold_ema_alpha,
        "threshold_extreme_adjustment": "short raw 0 uses validation probability min; long raw 1 uses validation probability max before EMA smoothing",
        "threshold_objective": "validation annualized Sharpe on fixed payoff pnl",
        "optuna_trials": args.optuna_trials,
        "optuna_metric": args.optuna_metric,
        "fixed_params_json": str(args.fixed_params_json) if args.fixed_params_json is not None else None,
        "fold_optuna_trials": args.fold_optuna_trials,
        "fold_optuna_timeout": args.fold_optuna_timeout,
        "fold_optuna_learning_rate_factor": args.fold_optuna_learning_rate_factor,
        "fold_optuna_regularization_factor": args.fold_optuna_regularization_factor,
        "fold_optuna_num_leaves_radius": args.fold_optuna_num_leaves_radius,
        "fold_optuna_child_samples_radius": args.fold_optuna_child_samples_radius,
        "fold_optuna_sample_radius": args.fold_optuna_sample_radius,
    }
    with (args.out_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, indent=2)
        handle.write("\n")

    predictions = walk_forward(dataset, args)
    summary = summarize(predictions, args, target_symbols)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
