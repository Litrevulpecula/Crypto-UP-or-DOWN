#!/usr/bin/env python3
"""REST backfill for OOS 1m klines.

We merge the existing per-symbol gz with freshly fetched klines through the
latest completed minute. Writes are atomic: a temporary gzip is fully written,
integrity-checked, then moved over the destination. Spot comes from
api.binance.com and futures from fapi.binance.com. Do not run this as the live
data loop; live uses websockets via live/update_live_klines.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SRC = Path("aligned_data")
DST = Path("aligned_data_oos")
SPOT = "binance_spot_klines"
FUT = "binance_um_futures_klines"
DEFAULT_LIVE_LOOKBACK_HOURS = 12.0
OVERLAP_MINUTES = 120
# fetch through end of live data; start a bit before existing tail to dedupe-merge
FETCH_START_MS = int(pd.Timestamp("2026-04-23T00:00:00Z").timestamp() * 1000)

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def parse_symbols(value: str) -> list[str]:
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not symbols:
        raise ValueError("--symbols must contain at least one symbol.")
    return symbols


def gzip_is_readable(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        return True
    except (EOFError, OSError, gzip.BadGzipFile):
        return False


def write_gzip_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        with gzip.open(temp_path, "wt") as handle:
            frame.to_csv(handle, index=False)
        if not gzip_is_readable(temp_path):
            raise RuntimeError(f"temporary gzip failed integrity check: {temp_path}")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def readable_existing_path(dst: Path, src: Path) -> Path:
    if dst.exists():
        if gzip_is_readable(dst):
            return dst
        print(f"WARNING: {dst} is not a valid gzip; rebuilding from {src} plus fetched klines.", flush=True)
    if not src.exists():
        raise FileNotFoundError(f"missing source file for rebuild: {src}")
    if not gzip_is_readable(src):
        raise RuntimeError(f"source gzip is not readable: {src}")
    return src


def binance_ban_until(exc: urllib.error.HTTPError) -> str | None:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except Exception:
        return None
    if payload.get("code") != -1003:
        return None
    match = re.search(r"banned until (\d+)", str(payload.get("msg", "")))
    if not match:
        return None
    until_ms = int(match.group(1))
    return pd.to_datetime(until_ms, unit="ms", utc=True).isoformat()


def default_fetch_end_ms() -> int:
    latest_closed = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)
    return int(latest_closed.timestamp() * 1000)


def fetch(base: str, symbol: str, start_ms: int, end_ms: int) -> list[list]:
    rows = []
    cur = start_ms
    ua = {"User-Agent": "Mozilla/5.0 (oos-fetch)"}
    while cur < end_ms:
        q = urllib.parse.urlencode(dict(symbol=symbol, interval="1m", startTime=cur,
                                        endTime=min(cur + 1000 * 60000, end_ms), limit=1000))
        url = f"{base}?{q}"
        req = urllib.request.Request(url, headers=ua)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    d = json.loads(resp.read())
                break
            except urllib.error.HTTPError as exc:
                ban_until = binance_ban_until(exc)
                if ban_until is not None:
                    raise RuntimeError(f"{symbol} Binance API IP ban until {ban_until}; stopping fetch.") from exc
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))
        if not d:
            break
        rows.extend(d)
        cur = int(d[-1][0]) + 60000
        if len(d) < 1000:
            break
    return rows


def to_frame(raw: list[list]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
            "ignore", "is_missing", "source_file",
        ])
    df = pd.DataFrame(raw, columns=COLS)
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_volume", "taker_buy_quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("Int64")
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df["is_missing"] = 0
    df["source_file"] = "oos_fetch"
    return df[["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
               "ignore", "is_missing", "source_file"]]


def extend(
    src_root: Path,
    dst_root: Path,
    market: str,
    base: str,
    symbol: str,
    end_ms: int,
    lookback_hours: float,
) -> tuple[int, str, str]:
    src = src_root / market / symbol / "1m.csv.gz"
    dst = dst_root / market / symbol / "1m.csv.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    old_path = readable_existing_path(dst, src)
    old = pd.read_csv(old_path)
    last_open = int(old["open_time"].max())
    lookback_start_ms = end_ms - int(float(lookback_hours) * 3600_000)
    if last_open >= lookback_start_ms:
        start_ms = max(FETCH_START_MS, last_open - OVERLAP_MINUTES * 60000)
    else:
        start_ms = max(FETCH_START_MS, lookback_start_ms)
    new = to_frame(fetch(base, symbol, start_ms, end_ms))
    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time")
    write_gzip_csv_atomic(merged, dst)
    last = pd.to_datetime(int(merged["open_time"].iloc[-1]), unit="ms", utc=True)
    return len(merged), str(last), symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REST backfill for OOS Binance 1m klines.")
    parser.add_argument("--src-root", type=Path, default=SRC)
    parser.add_argument("--dst-root", type=Path, default=DST)
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument(
        "--end",
        help="UTC end timestamp, defaults to the latest completed 1m candle.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=DEFAULT_LIVE_LOOKBACK_HOURS,
        help="When existing OOS data is stale, only fetch this recent live window.",
    )
    return parser.parse_args()


def parse_end_ms(value: str | None) -> int:
    if not value:
        return default_fetch_end_ms()
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.floor("min").timestamp() * 1000)


def main() -> None:
    args = parse_args()
    end_ms = parse_end_ms(args.end)
    if args.lookback_hours <= 0:
        raise ValueError("--lookback-hours must be > 0.")
    symbols = parse_symbols(args.symbols)
    end_text = pd.to_datetime(end_ms, unit="ms", utc=True).isoformat()
    print(f"fetch_end={end_text}", flush=True)
    for symbol in symbols:
        n1, last1, _ = extend(
            args.src_root,
            args.dst_root,
            SPOT,
            "https://api.binance.com/api/v3/klines",
            symbol,
            end_ms,
            args.lookback_hours,
        )
        n2, last2, _ = extend(
            args.src_root,
            args.dst_root,
            FUT,
            "https://fapi.binance.com/fapi/v1/klines",
            symbol,
            end_ms,
            args.lookback_hours,
        )
        print(f"{symbol}: spot rows={n1} last={last1} | fut rows={n2} last={last2}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
