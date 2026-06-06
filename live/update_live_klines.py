#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import websockets


SPOT_MARKET = "binance_spot_klines"
FUTURES_MARKET = "binance_um_futures_klines"
SPOT_WS = "wss://stream.binance.com:9443/stream"
FUTURES_WS = "wss://fstream.binance.com/stream"
CSV_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
    "is_missing",
    "source_file",
]

log = logging.getLogger("update_live_klines")


def parse_symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not symbols:
        raise ValueError("--symbols is empty")
    return symbols


def stream_url(base_url: str, symbols: tuple[str, ...]) -> str:
    streams = "/".join(f"{symbol.lower()}@kline_1m" for symbol in symbols)
    return f"{base_url}?streams={streams}"


def kline_row(kline: dict[str, Any]) -> dict[str, str]:
    open_time = int(kline["t"])
    close_time = int(kline["T"])
    if close_time != open_time + 59_999:
        raise ValueError(f"unexpected close_time for {kline.get('s')}: {open_time=} {close_time=}")
    return {
        "open_time": str(open_time),
        "open": str(kline["o"]),
        "high": str(kline["h"]),
        "low": str(kline["l"]),
        "close": str(kline["c"]),
        "volume": str(kline["v"]),
        "close_time": str(close_time),
        "quote_volume": str(kline["q"]),
        "count": str(int(kline["n"])),
        "taker_buy_volume": str(kline["V"]),
        "taker_buy_quote_volume": str(kline["Q"]),
        "ignore": str(kline.get("B", "0")),
        "is_missing": "0",
        "source_file": "binance_ws_live",
    }


def load_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows[int(row["open_time"])] = {column: row.get(column, "") for column in CSV_COLUMNS}
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def write_rows(path: Path, rows: dict[int, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
        temp_path = Path(handle.name)
    temp_path.replace(path)


@dataclass
class OverlayStore:
    data_root: Path
    keep_rows: int
    rows_by_path: dict[Path, dict[int, dict[str, str]]] = field(default_factory=dict)

    def update(self, market: str, symbol: str, row: dict[str, str]) -> Path:
        path = self.data_root / market / symbol / "1m_live.csv"
        rows = self.rows_by_path.get(path)
        if rows is None:
            rows = load_rows(path)
            self.rows_by_path[path] = rows
        rows[int(row["open_time"])] = row
        while len(rows) > self.keep_rows:
            del rows[min(rows)]
        write_rows(path, rows)
        return path


async def consume_market(store: OverlayStore, market: str, ws_base: str, symbols: tuple[str, ...]) -> None:
    url = stream_url(ws_base, symbols)
    backoff = 1.0
    while True:
        try:
            log.info("connect %s websocket", market)
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as websocket:
                backoff = 1.0
                async for message in websocket:
                    payload = json.loads(message)
                    kline = payload.get("data", {}).get("k", {})
                    if not kline.get("x"):
                        continue
                    symbol = str(kline["s"]).upper()
                    try:
                        row = kline_row(kline)
                    except ValueError as exc:
                        log.warning("skip malformed kline: %s", exc)
                        continue
                    path = store.update(market, symbol, row)
                    log.info("%s %s closed open_time=%s close=%s", market, symbol, row["open_time"], row["close"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("%s websocket disconnected: %s", market, exc)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


async def run(args: argparse.Namespace) -> None:
    symbols = parse_symbols(args.symbols)
    store = OverlayStore(args.data_root, args.keep_rows)
    await asyncio.gather(
        consume_market(store, SPOT_MARKET, SPOT_WS, symbols),
        consume_market(store, FUTURES_MARKET, FUTURES_WS, symbols),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Write live Binance closed 1m klines from websocket streams.")
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data_oos"))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--keep-rows", type=int, default=3000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.keep_rows < 300:
        raise ValueError("--keep-rows must be >= 300")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
