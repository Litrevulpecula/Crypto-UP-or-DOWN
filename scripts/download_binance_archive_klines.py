#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path


MARKETS = {
    "spot": {
        "raw_dir": "binance_spot_klines",
        "base": "https://data.binance.vision/data/spot",
    },
    "um_futures": {
        "raw_dir": "binance_um_futures_klines",
        "base": "https://data.binance.vision/data/futures/um",
    },
}


def parse_symbols(value: str) -> list[str]:
    symbols = []
    for item in value.split(","):
        symbol = item.strip().upper()
        if not symbol:
            continue
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        symbols.append(symbol)
    if not symbols:
        raise ValueError("--symbols must contain at least one symbol.")
    return symbols


def month_range(start: str, end: str) -> list[str]:
    year, month = [int(part) for part in start.split("-")]
    end_year, end_month = [int(part) for part in end.split("-")]
    out = []
    while (year, month) <= (end_year, end_month):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return out


def day_range(start: str, end: str) -> list[str]:
    current = date.fromisoformat(start)
    last = date.fromisoformat(end)
    out = []
    while current <= last:
        out.append(current.isoformat())
        current = date.fromordinal(current.toordinal() + 1)
    return out


def download_file(url: str, path: Path, retries: int, overwrite: bool) -> str:
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            tmp_path.write_bytes(data)
            tmp_path.replace(path)
            return "downloaded"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                if tmp_path.exists():
                    tmp_path.unlink()
                return "missing"
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    return "failed"


def archive_url(base: str, frequency: str, symbol: str, period: str) -> str:
    return f"{base}/{frequency}/klines/{symbol}/1m/{symbol}-1m-{period}.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance archive 1m kline zip files.")
    parser.add_argument("--raw-root", type=Path, default=Path("raw_data"))
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. DOGE,BNB,XRP,SOL.")
    parser.add_argument("--monthly-start", default="2022-01")
    parser.add_argument("--monthly-end", default="2026-03")
    parser.add_argument("--daily-start", default="2026-04-01")
    parser.add_argument("--daily-end", default="2026-04-23")
    parser.add_argument("--markets", default="spot,um_futures")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-checksums", action="store_true")
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)
    market_keys = [item.strip() for item in args.markets.split(",") if item.strip()]
    unknown_markets = sorted(set(market_keys) - set(MARKETS))
    if unknown_markets:
        raise ValueError(f"Unknown markets: {unknown_markets}")

    monthly_periods = month_range(args.monthly_start, args.monthly_end)
    daily_periods = day_range(args.daily_start, args.daily_end)
    counters = {"downloaded": 0, "exists": 0, "missing": 0, "failed": 0}

    for market_key in market_keys:
        market = MARKETS[market_key]
        for symbol in symbols:
            out_dir = args.raw_root / market["raw_dir"] / symbol / "1m"
            for frequency, periods in [("monthly", monthly_periods), ("daily", daily_periods)]:
                for period in periods:
                    filename = f"{symbol}-1m-{period}.zip"
                    url = archive_url(market["base"], frequency, symbol, period)
                    status = download_file(url, out_dir / filename, args.retries, args.overwrite)
                    counters[status] = counters.get(status, 0) + 1
                    print(f"{status:10s} {market_key:10s} {symbol:10s} {frequency:7s} {period}", flush=True)
                    if args.download_checksums and status != "missing":
                        checksum_status = download_file(
                            f"{url}.CHECKSUM",
                            out_dir / f"{filename}.CHECKSUM",
                            args.retries,
                            args.overwrite,
                        )
                        counters[checksum_status] = counters.get(checksum_status, 0) + 1

    print("SUMMARY " + " ".join(f"{key}={value}" for key, value in sorted(counters.items())), flush=True)


if __name__ == "__main__":
    main()
