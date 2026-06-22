#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

import websockets

from log_colors import configure_colored_logging


SPOT_MARKET = "binance_spot_klines"
FUTURES_MARKET = "binance_um_futures_klines"
SPOT_WS = "wss://stream.binance.com:9443/stream"
FUTURES_WS = "wss://fstream.binance.com/stream"
SPOT_REST = "https://api.binance.com/api/v3/klines"
FUTURES_REST = "https://fapi.binance.com/fapi/v1/klines"
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
FEATURE_ROW_COLUMNS = tuple(column for column in CSV_COLUMNS if column != "source_file")

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


def rest_kline_row(raw: list[Any]) -> dict[str, str]:
    open_time = int(raw[0])
    close_time = int(raw[6])
    if close_time != open_time + 59_999:
        raise ValueError(f"unexpected REST close_time: {open_time=} {close_time=}")
    return {
        "open_time": str(open_time),
        "open": str(raw[1]),
        "high": str(raw[2]),
        "low": str(raw[3]),
        "close": str(raw[4]),
        "volume": str(raw[5]),
        "close_time": str(close_time),
        "quote_volume": str(raw[7]),
        "count": str(int(raw[8])),
        "taker_buy_volume": str(raw[9]),
        "taker_buy_quote_volume": str(raw[10]),
        "ignore": str(raw[11]),
        "is_missing": "0",
        "source_file": "binance_rest_live",
    }


def latest_closed_open_time_ms() -> int:
    return int((time.time() // 60) * 60 - 60) * 1000


def fetch_rest_klines(endpoint: str, symbol: str, start_ms: int, end_ms: int) -> list[list[Any]]:
    rows: list[list[Any]] = []
    current = start_ms
    headers = {"User-Agent": "Mozilla/5.0 (live-kline-warmup)"}
    while current <= end_ms:
        query = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": "1m",
                "startTime": current,
                "endTime": min(current + 999 * 60_000, end_ms),
                "limit": 1000,
            }
        )
        request = urllib.request.Request(f"{endpoint}?{query}", headers=headers)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    payload = json.loads(response.read())
                break
            except (urllib.error.URLError, urllib.error.HTTPError):
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))
        if not payload:
            break
        rows.extend(row for row in payload if start_ms <= int(row[0]) <= end_ms)
        current = int(payload[-1][0]) + 60_000
        if len(payload) < 1000:
            break
    return rows


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


def feature_row_changed(old: dict[str, str] | None, new: dict[str, str]) -> bool:
    if old is None:
        return True
    return any(old.get(column, "") != new.get(column, "") for column in FEATURE_ROW_COLUMNS)


@dataclass
class OverlayStore:
    data_root: Path
    keep_rows: int
    on_rows_updated: Callable[[str, str, list[dict[str, str]]], None] | None = None
    rows_by_path: dict[Path, dict[int, dict[str, str]]] = field(default_factory=dict)

    def update(self, market: str, symbol: str, row: dict[str, str]) -> Path:
        return self.update_many(market, symbol, [row])

    def update_many(self, market: str, symbol: str, new_rows: list[dict[str, str]]) -> Path:
        path = self.data_root / market / symbol / "1m_live.csv"
        rows = self.rows_by_path.get(path)
        if rows is None:
            rows = load_rows(path)
            self.rows_by_path[path] = rows
        changed_rows = []
        for row in new_rows:
            open_time = int(row["open_time"])
            if feature_row_changed(rows.get(open_time), row):
                rows[open_time] = row
                changed_rows.append(row)
        changed = bool(changed_rows)
        while len(rows) > self.keep_rows:
            del rows[min(rows)]
            changed = True
        if changed:
            write_rows(path, rows)
        if changed_rows and self.on_rows_updated is not None:
            try:
                self.on_rows_updated(market, symbol, changed_rows)
            except Exception:
                log.exception("signal callback failed for %s %s", market, symbol)
        return path


class SignalWriterCallback:
    def __init__(
        self,
        data_root: Path,
        signal_file: Path,
        symbols: str,
        model_dir: list[str],
        feature_set: str,
        lookback_days: float,
        store: OverlayStore,
    ) -> None:
        import write_lightgbm_signals as signal_writer

        self.signal_writer = signal_writer
        self.args = argparse.Namespace(
            data_root=data_root,
            signal_file=signal_file,
            model_dir=model_dir,
            symbols=symbols,
            feature_set=feature_set,
            lookback_days=lookback_days,
        )
        self.store = store
        self.symbols = signal_writer.base.parse_symbols(symbols)
        self.model_dirs = signal_writer.parse_model_dirs(model_dir)
        self.timeframes = sorted(self.model_dirs, key=signal_writer.timeframe_minutes)
        self.emitted: set[tuple[str, str]] = set()
        self.started_at = signal_writer.pd.Timestamp.now(tz="UTC").floor("min")
        warmup_start = time.perf_counter()
        for path in self.model_dirs.values():
            signal_writer.load_model_bundle(path, self.symbols)
        self.args.dataset_start = signal_writer.latest_dataset_start(data_root, self.symbols, lookback_days)
        warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
        log.info(
            "signal callback ready timeframes=%s warmup_ms=%.3f signal_file=%s",
            ",".join(self.timeframes),
            warmup_ms,
            signal_file,
        )

    def __call__(self, market: str, symbol: str, rows: list[dict[str, str]]) -> None:
        writer = self.signal_writer
        for row in rows:
            try:
                open_time_ms = int(row["open_time"])
            except (KeyError, TypeError, ValueError):
                continue
            decision_time = writer.pd.Timestamp(open_time_ms + 60_000, unit="ms", tz="UTC")
            if decision_time < self.started_at:
                continue
            targets = writer.due_decision_times(self.timeframes, decision_time)
            targets = {
                timeframe: target
                for timeframe, target in targets.items()
                if (timeframe, target.isoformat()) not in self.emitted
            }
            if not targets:
                continue
            ready, missing = self.target_data_ready(targets)
            if not ready:
                log.debug(
                    "signal callback waiting target=%s due=%s missing=%s",
                    decision_time.isoformat(),
                    ",".join(targets),
                    missing[:8],
                )
                continue
            generate_start = time.perf_counter()
            payload = writer.generate_signals(self.args, targets)
            generate_ms = (time.perf_counter() - generate_start) * 1000.0
            write_ms = writer.write_and_log_payload(
                self.args.signal_file,
                payload,
                extra=(
                    f"callback_market={market} callback_symbol={symbol} "
                    f"callback_source={row.get('source_file', '')} "
                    f"target_decision_time={decision_time.tz_convert('UTC').isoformat()} "
                    f"due={','.join(targets)} generate_ms={generate_ms:.3f}"
                ),
            )
            for timeframe, target in targets.items():
                self.emitted.add((timeframe, target.isoformat()))
            end_delay_ms = (writer.pd.Timestamp.now(tz="UTC") - decision_time).total_seconds() * 1000.0
            log.info(
                "signal callback emitted target=%s due=%s trigger=%s:%s source=%s generate_ms=%.3f write_ms=%.3f end_delay_ms=%.3f",
                decision_time.isoformat(),
                ",".join(targets),
                market,
                symbol,
                row.get("source_file", ""),
                generate_ms,
                write_ms,
                end_delay_ms,
            )

    def target_data_ready(self, target_decision_times: dict[str, Any]) -> tuple[bool, list[str]]:
        writer = self.signal_writer
        missing = []
        for timeframe, decision_time in target_decision_times.items():
            warmup_minutes = max(writer.base.WINDOWS + writer.base.ROLLING_WINDOWS) + writer.timeframe_minutes(timeframe) + 5
            target_open_ms = int((decision_time - writer.pd.Timedelta(minutes=1)).timestamp() * 1000)
            for symbol in self.symbols:
                for market in (writer.base.SPOT_MARKET, writer.base.FUTURES_MARKET):
                    path = self.args.data_root / market / symbol / "1m_live.csv"
                    rows = self.store.rows_by_path.get(path)
                    if rows is None:
                        missing.append(f"{timeframe}:{market}:{symbol}")
                        continue
                    tail = sorted(open_time for open_time in rows if open_time <= target_open_ms)[-(warmup_minutes + 1) :]
                    if len(tail) < warmup_minutes + 1 or tail[-1] != target_open_ms:
                        missing.append(f"{timeframe}:{market}:{symbol}")
                        continue
                    if any((right - left) != 60_000 for left, right in zip(tail, tail[1:])):
                        missing.append(f"{timeframe}:{market}:{symbol}")
        return not missing, missing


def rest_backfill(
    store: OverlayStore,
    symbols: tuple[str, ...],
    minutes: int,
    *,
    log_success: bool = True,
    markets: tuple[str, ...] = (SPOT_MARKET, FUTURES_MARKET),
) -> None:
    if minutes <= 0:
        return
    end_ms = latest_closed_open_time_ms()
    start_ms = end_ms - (minutes - 1) * 60_000
    endpoints = {
        SPOT_MARKET: SPOT_REST,
        FUTURES_MARKET: FUTURES_REST,
    }
    for market in markets:
        endpoint = endpoints[market]
        for symbol in symbols:
            raw_rows = fetch_rest_klines(endpoint, symbol, start_ms, end_ms)
            rows = [rest_kline_row(row) for row in raw_rows]
            if not rows:
                raise RuntimeError(f"{market} {symbol} REST backfill returned no rows")
            path = store.update_many(market, symbol, rows)
            logger = log.info if log_success else log.debug
            logger(
                "%s %s REST backfilled rows=%s first_open_time=%s last_open_time=%s path=%s",
                market,
                symbol,
                len(rows),
                rows[0]["open_time"],
                rows[-1]["open_time"],
                path,
            )


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
                    log.debug("%s %s closed open_time=%s close=%s path=%s", market, symbol, row["open_time"], row["close"], path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("%s websocket disconnected: %s", market, exc)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


async def periodic_rest_catchup(
    store: OverlayStore,
    symbols: tuple[str, ...],
    minutes: int,
    interval_seconds: float,
    markets: tuple[str, ...],
) -> None:
    while True:
        boundary = (int(time.time() // 60) + 1) * 60
        await asyncio.sleep(max(0.0, boundary - time.time() + interval_seconds))
        try:
            await asyncio.to_thread(rest_backfill, store, symbols, minutes, log_success=False, markets=markets)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("REST catch-up failed: %s", exc)


async def run(args: argparse.Namespace) -> None:
    symbols = parse_symbols(args.symbols)
    store = OverlayStore(args.data_root, args.keep_rows)
    rest_backfill(store, symbols, args.rest_backfill_minutes)
    if args.rest_backfill_only:
        return
    if args.signal_file is not None:
        store.on_rows_updated = SignalWriterCallback(
            args.data_root,
            args.signal_file,
            args.symbols,
            args.signal_model_dir,
            args.signal_feature_set,
            args.signal_lookback_days,
            store,
        )
    tasks = [
        consume_market(store, SPOT_MARKET, SPOT_WS, symbols),
        consume_market(store, FUTURES_MARKET, FUTURES_WS, symbols),
    ]
    if args.rest_catchup_minutes > 0 and args.rest_catchup_seconds > 0:
        tasks.append(periodic_rest_catchup(store, symbols, args.rest_catchup_minutes, args.rest_catchup_seconds, tuple(args.rest_catchup_market)))
    await asyncio.gather(*tasks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write live Binance closed 1m klines from websocket streams.")
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data_oos"))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--keep-rows", type=int, default=3000)
    parser.add_argument("--rest-backfill-minutes", type=int, default=720)
    parser.add_argument("--rest-backfill-only", action="store_true")
    parser.add_argument("--rest-catchup-minutes", type=int, default=15)
    parser.add_argument("--rest-catchup-seconds", type=float, default=10.0)
    parser.add_argument("--rest-catchup-market", action="append", default=[FUTURES_MARKET], choices=(SPOT_MARKET, FUTURES_MARKET))
    parser.add_argument("--signal-file", type=Path, default=None)
    parser.add_argument("--signal-model-dir", action="append", default=[])
    parser.add_argument("--signal-feature-set", default="v1")
    parser.add_argument("--signal-lookback-days", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.keep_rows < 300:
        raise ValueError("--keep-rows must be >= 300")
    if args.rest_backfill_minutes < 0:
        raise ValueError("--rest-backfill-minutes must be >= 0")
    if args.rest_backfill_minutes > args.keep_rows:
        raise ValueError("--rest-backfill-minutes must be <= --keep-rows")
    if args.rest_catchup_minutes < 0:
        raise ValueError("--rest-catchup-minutes must be >= 0")
    if args.rest_catchup_minutes > args.keep_rows:
        raise ValueError("--rest-catchup-minutes must be <= --keep-rows")
    if args.rest_catchup_seconds < 0:
        raise ValueError("--rest-catchup-seconds must be >= 0")
    configure_colored_logging(getattr(logging, args.log_level.upper()))
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
