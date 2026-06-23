#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from turboflow_api_client import (
    TurboFlowCredentials,
    build_order_fields,
    normalize_timeframe,
    place_order,
    return_rate_for,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SIGNAL_FILE = ROOT / "signals.json"
DEFAULT_STATE_FILE = ROOT / "runtime" / "turboflow_api_state.json"
DEFAULT_DRYRUN_STATE_FILE = ROOT / "runtime" / "turboflow_api_dryrun_state.json"
DEFAULT_LOG_FILE = ROOT / "runtime" / "turboflow_api_orders.jsonl"
DEFAULT_CONTROL_FILE = ROOT / "runtime" / "turboflow_control.json"


@dataclass(frozen=True)
class Signal:
    signal_id: str
    symbol: str
    timeframe: str
    side: str
    timestamp: str
    decision_time: str
    last_kline_time: str
    signal_generated_at: str


class State:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.processed = self._load()

    def seen(self, signal_id: str) -> bool:
        return signal_id in self.processed

    def mark(self, signal_id: str) -> None:
        self.processed.add(signal_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
            json.dump({"processed_signal_ids": sorted(self.processed)}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(self.path)

    def _load(self) -> set[str]:
        if not self.path.exists():
            return set()
        return set(json.loads(self.path.read_text(encoding="utf-8")).get("processed_signal_ids", []))


def main() -> int:
    args = parse_args()
    if not args.live and args.state_file == DEFAULT_STATE_FILE:
        args.state_file = DEFAULT_DRYRUN_STATE_FILE
    allowed_timeframes = {normalize_timeframe(item) for item in args.timeframes.split(",") if item.strip()}
    state = State(args.state_file)
    if not args.live:
        state.processed.update(read_logged_signal_ids(args.log_file))
    credentials = TurboFlowCredentials.from_env() if args.live else dry_run_credentials()
    print(
        f"TurboFlow trader start live={args.live} signal_file={args.signal_file} "
        f"timeframes={','.join(sorted(allowed_timeframes))}",
        flush=True,
    )
    events = signal_file_events(args.signal_file, args.poll_seconds)
    while True:
        try:
            control = read_control(args.control_file)
            if not control["enabled"]:
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue
            amount = selected_amount(args.amount, control["order_amount"])
            for signal in read_signals(args.signal_file):
                if signal.timeframe not in allowed_timeframes or state.seen(signal.signal_id):
                    continue
                order_started_at = datetime.now(timezone.utc).isoformat()
                rate = return_rate_for(symbol=signal.symbol, timeframe=signal.timeframe, side=signal.side)
                api_order = build_order_fields(
                    credentials=credentials,
                    amount=amount,
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
                    side=signal.side,
                    return_rate=rate,
                )
                if args.live:
                    api_order, response = place_order(
                        credentials=credentials,
                        amount=amount,
                        symbol=signal.symbol,
                        timeframe=signal.timeframe,
                        side=signal.side,
                    )
                    success = order_succeeded(response)
                else:
                    response = {"dry_run": True}
                    success = True
                record_order(args.log_file, signal, api_order, response, args.live, order_started_at, success)
                state.mark(signal.signal_id)
                status = "submitted" if args.live and success else "failed" if args.live else "dry_run"
                print(f"{status} {signal.signal_id} {signal.symbol} {signal.timeframe} {signal.side}", flush=True)
            if args.once:
                return 0
            next(events)
        except KeyboardInterrupt:
            print("stopped by user", flush=True)
            return 130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute signals through TurboFlow prediction markets.")
    parser.add_argument("--signal-file", type=Path, default=DEFAULT_SIGNAL_FILE)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--control-file", type=Path, default=DEFAULT_CONTROL_FILE)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--amount", default="3")
    parser.add_argument("--timeframes", default="3m,5m,15m")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def signal_file_events(path: Path, poll_seconds: float):
    yield
    if os.uname().sysname == "Linux":
        try:
            yield from linux_inotify_events(path)
            return
        except OSError as exc:
            print(f"inotify unavailable, fallback poll_seconds={poll_seconds}: {exc}", flush=True)
    while True:
        time.sleep(poll_seconds)
        yield


def linux_inotify_events(path: Path):
    import ctypes
    import select
    import struct

    path.parent.mkdir(parents=True, exist_ok=True)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1 failed")
    try:
        mask = 0x00000008 | 0x00000080 | 0x00000100
        wd = libc.inotify_add_watch(fd, os.fsencode(path.parent), mask)
        if wd < 0:
            raise OSError(ctypes.get_errno(), f"inotify_add_watch failed: {path.parent}")
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        while True:
            poller.poll()
            data = os.read(fd, 4096)
            offset = 0
            changed = False
            while offset + 16 <= len(data):
                _wd, _mask, _cookie, name_len = struct.unpack_from("iIII", data, offset)
                name = data[offset + 16 : offset + 16 + name_len].rstrip(b"\0").decode()
                changed = changed or name == path.name
                offset += 16 + name_len
            if changed:
                yield
    finally:
        os.close(fd)


def read_control(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"enabled": True, "order_amount": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"enabled": True, "order_amount": None}
    return {"enabled": bool(payload.get("enabled", True)), "order_amount": payload.get("order_amount")}


def selected_amount(cli_amount: Any, control_amount: Any) -> str:
    for value in (control_amount, cli_amount):
        try:
            if float(value) > 0:
                return str(value)
        except (TypeError, ValueError):
            pass
    raise ValueError(f"order amount must be positive, got {cli_amount!r}")


def read_signals(path: Path) -> list[Signal]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    generated_at = str(payload.get("generated_at") or "")
    rows = payload.get("signals") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a signals list")
    return sorted((parse_signal(row, generated_at) for row in rows), key=lambda item: item.timestamp)


def parse_signal(record: Any, generated_at: str) -> Signal:
    if not isinstance(record, dict):
        raise ValueError(f"signal record must be a JSON object: {record!r}")
    required = ("signal_id", "symbol", "timeframe", "signal", "timestamp", "decision_time", "last_kline_time")
    missing = [key for key in required if key not in record or str(record[key]).strip() == ""]
    if missing:
        raise ValueError(f"signal record missing fields: {', '.join(missing)}")
    side = str(record["signal"]).strip().upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError(f"unsupported signal side: {record['signal']!r}")
    return Signal(
        signal_id=str(record["signal_id"]),
        symbol=str(record["symbol"]).strip().upper(),
        timeframe=normalize_timeframe(str(record["timeframe"])),
        side=side,
        timestamp=str(record["timestamp"]),
        decision_time=str(record["decision_time"]),
        last_kline_time=str(record["last_kline_time"]),
        signal_generated_at=generated_at,
    )


def read_logged_signal_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    signal_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            signal = row.get("signal") if isinstance(row, dict) else None
            if isinstance(signal, dict) and signal.get("signal_id"):
                signal_ids.add(str(signal["signal_id"]))
    return signal_ids


def dry_run_credentials() -> TurboFlowCredentials:
    return TurboFlowCredentials(token="", uid="dry-run", coin_code=os.environ.get("TURBOFLOW_COIN_CODE", "USDT"), pool_id=os.environ.get("TURBOFLOW_POOL_ID", "0"))


def order_succeeded(response: dict[str, Any]) -> bool:
    status = response.get("http_status")
    if not isinstance(status, int) or status < 200 or status >= 300:
        return False
    body = response.get("body")
    return not isinstance(body, dict) or str(body.get("errno", "200")) == "200"


def record_order(
    path: Path,
    signal: Signal,
    api_order: dict[str, Any],
    response: dict[str, Any],
    live: bool,
    order_started_at: str,
    success: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": not live,
        "signal": asdict(signal),
        "order": {
            "amount": api_order["amount"],
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "signal": signal.side,
        },
        "api_order": api_order,
        "response": response,
        "order_started_at": order_started_at,
        "success": success,
        "payout_rate": float(api_order["return_rate"]),
    }
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
