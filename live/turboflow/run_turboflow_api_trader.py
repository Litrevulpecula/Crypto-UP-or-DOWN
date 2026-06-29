#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

from turboflow_api_client import (
    TurboFlowCredentials,
    account_balance,
    build_order_fields,
    normalize_timeframe,
    place_order,
    return_rate_from_map,
    return_rate_map,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SIGNAL_FILE = ROOT / "signals.json"
DEFAULT_STATE_FILE = ROOT / "runtime" / "turboflow_api_state.json"
DEFAULT_DRYRUN_STATE_FILE = ROOT / "runtime" / "turboflow_api_dryrun_state.json"
DEFAULT_LOG_FILE = ROOT / "runtime" / "turboflow_api_orders.jsonl"
DEFAULT_SETTLEMENT_FILE = ROOT / "runtime" / "turboflow_settlements.jsonl"
DEFAULT_CONTROL_FILE = ROOT / "runtime" / "turboflow_control.json"
DEFAULT_DATA_ROOT = ROOT.parent.parent / "aligned_data_oos"
DEFAULT_BANKROLL = 200.0
MIN_ORDER_AMOUNT = 2.0
BALANCE_REFRESH_SECONDS = 30.0
BALANCE_TIMEOUT_SECONDS = 2.0
CONFIG_TIMEOUT_SECONDS = 3.0
MAX_SIGNAL_AGE_SECONDS = 10.0
QUARTER_KELLY = {
    ("BTCUSDT", "3m"): 0.02207,
    ("ETHUSDT", "3m"): 0.01297,
    ("BTCUSDT", "5m"): 0.01218,
    ("ETHUSDT", "5m"): 0.01264,
    ("BTCUSDT", "15m"): 0.02069,
    ("ETHUSDT", "15m"): 0.01839,
}


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
    if args.self_test:
        self_test()
        return 0
    if not args.live and args.state_file == DEFAULT_STATE_FILE:
        args.state_file = DEFAULT_DRYRUN_STATE_FILE
    allowed_timeframes = {normalize_timeframe(item) for item in args.timeframes.split(",") if item.strip()}
    state = State(args.state_file)
    if not args.live:
        state.processed.update(read_logged_signal_ids(args.log_file))
    credentials = TurboFlowCredentials.from_env() if args.live else dry_run_credentials()
    balance = BalanceCache(credentials, args.live, args.bankroll)
    balance.refresh(args.bankroll)
    print(
        f"TurboFlow trader start live={args.live} signal_file={args.signal_file} "
        f"timeframes={','.join(sorted(allowed_timeframes))}",
        flush=True,
    )
    events = signal_file_events(args.signal_file, args.poll_seconds)
    while True:
        try:
            control = read_control(args.control_file, args.bankroll)
            if not control["enabled"]:
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue
            if not args.live:
                write_due_settlements(args.log_file, args.settlement_file, args.data_root)
            control_bankroll = effective_bankroll(args.live, args.settlement_file, control["bankroll"])
            signals = [
                signal
                for signal in read_signals(args.signal_file)
                if signal.timeframe in allowed_timeframes and not state.seen(signal.signal_id)
            ]
            if not signals:
                if args.live:
                    balance.refresh(control_bankroll)
                if args.once:
                    return 0
                next(events)
                continue
            active_signals = []
            for signal in signals:
                if is_stale_signal(signal):
                    print(f"skip stale {signal.signal_id}", flush=True)
                    state.mark(signal.signal_id)
                    continue
                active_signals.append(signal)
            if not active_signals:
                if args.once:
                    return 0
                next(events)
                continue
            rates = return_rate_map(timeout_seconds=CONFIG_TIMEOUT_SECONDS)
            for signal in active_signals:
                try:
                    bankroll, bankroll_source = balance.current(control_bankroll)
                except RuntimeError as exc:
                    print(f"skip {signal.signal_id}: {exc}", flush=True)
                    state.mark(signal.signal_id)
                    continue
                amount = order_amount(signal, bankroll)
                order_started_at = datetime.now(timezone.utc).isoformat()
                rate = return_rate_from_map(rates, symbol=signal.symbol, timeframe=signal.timeframe, side=signal.side)
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
                        return_rate=rate,
                    )
                    success = order_succeeded(response)
                else:
                    response = {"dry_run": True}
                    success = True
                record_order(args.log_file, signal, api_order, response, args.live, order_started_at, success, bankroll, bankroll_source)
                if success:
                    balance.debit(float(api_order["amount"]))
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
    parser.add_argument("--settlement-file", type=Path, default=DEFAULT_SETTLEMENT_FILE)
    parser.add_argument("--control-file", type=Path, default=DEFAULT_CONTROL_FILE)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--timeframes", default="3m,5m,15m")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def signal_file_events(path: Path, poll_seconds: float):
    yield
    if os.uname().sysname == "Linux":
        try:
            yield from linux_inotify_events(path, poll_seconds)
            return
        except OSError as exc:
            print(f"inotify unavailable, fallback poll_seconds={poll_seconds}: {exc}", flush=True)
    while True:
        time.sleep(poll_seconds)
        yield


def linux_inotify_events(path: Path, poll_seconds: float):
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
        timeout_ms = max(1000, int(poll_seconds * 1000))
        while True:
            if not poller.poll(timeout_ms):
                yield
                continue
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


def read_control(path: Path, default_bankroll: float = DEFAULT_BANKROLL) -> dict[str, Any]:
    if not path.exists():
        return {"enabled": True, "bankroll": positive_float(default_bankroll, DEFAULT_BANKROLL)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"enabled": True, "bankroll": positive_float(default_bankroll, DEFAULT_BANKROLL)}
    return {
        "enabled": bool(payload.get("enabled", True)),
        "bankroll": positive_float(payload.get("bankroll"), positive_float(default_bankroll, DEFAULT_BANKROLL)),
    }


def order_amount(signal: Signal, bankroll: Any) -> str:
    key = (normalize_symbol(signal.symbol), signal.timeframe)
    if key not in QUARTER_KELLY:
        raise ValueError(f"Kelly amount missing for {signal.symbol} {signal.timeframe}")
    amount = max(MIN_ORDER_AMOUNT, positive_float(bankroll, DEFAULT_BANKROLL) * QUARTER_KELLY[key])
    return f"{amount:.2f}"


class BalanceCache:
    def __init__(self, credentials: TurboFlowCredentials, live: bool, base_bankroll: Any) -> None:
        self.credentials = credentials
        self.live = live
        self.value = None if live else positive_float(base_bankroll, DEFAULT_BANKROLL)
        self.source = "turboflow" if live else "dry_run"
        self.refresh_at = 0.0

    def current(self, base_bankroll: Any) -> tuple[float, str]:
        if not self.live:
            self.value = positive_float(base_bankroll, DEFAULT_BANKROLL)
            self.source = "dry_run"
            return self.value, self.source
        if self.value is None:
            self.refresh(base_bankroll)
        if self.value is None:
            raise RuntimeError("TurboFlow balance unavailable; refusing live order")
        return self.value, "turboflow"

    def refresh(self, base_bankroll: Any) -> None:
        if not self.live:
            self.value = positive_float(base_bankroll, DEFAULT_BANKROLL)
            self.source = "dry_run"
            return
        now = time.monotonic()
        if now >= self.refresh_at:
            self.refresh_at = now + BALANCE_REFRESH_SECONDS
            try:
                self.value = account_balance(self.credentials, timeout_seconds=BALANCE_TIMEOUT_SECONDS)
                self.source = "turboflow"
            except RuntimeError as exc:
                self.value = None
                self.source = "unavailable"
                print(f"balance unavailable: {exc}", flush=True)

    def debit(self, amount: float) -> None:
        if self.live and self.source == "turboflow":
            self.value = max(0.0, self.value - amount)


def positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def normalize_symbol(value: str) -> str:
    text = str(value or "").strip().upper().replace("-", "").replace("/", "").replace("_", "")
    if text in {"BTC", "ETH"}:
        text += "USDT"
    return text


def effective_bankroll(live: bool, settlement_file: Path, base_bankroll: Any) -> float:
    base = positive_float(base_bankroll, DEFAULT_BANKROLL)
    if live:
        return base
    return max(MIN_ORDER_AMOUNT, base + settled_pnl(settlement_file))


def settled_pnl(settlement_file: Path) -> float:
    if not settlement_file.exists():
        return 0.0
    total = 0.0
    with settlement_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                total += float(row.get("pnl") or 0.0)
            except (TypeError, ValueError):
                continue
    return total


def write_due_settlements(log_file: Path, settlement_file: Path, data_root: Path) -> None:
    records = [row for row in read_order_records(log_file) if row.get("success") and not stale_record(row)]
    settled_ids = read_settled_signal_ids(settlement_file)
    due = [row for row in records if signal_id_for(row) not in settled_ids and order_is_due(row)]
    if not due:
        return
    closes = {symbol: read_live_closes(data_root, symbol) for symbol in symbols_for(due)}
    rows = [settlement for row in due if (settlement := settle_order(row, closes)) is not None]
    if rows:
        append_jsonl(settlement_file, rows)


def read_order_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def read_settled_signal_ids(path: Path) -> set[str]:
    return {str(row.get("signal_id")) for row in read_order_records(path) if row.get("signal_id")}


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def order_is_due(row: dict[str, Any]) -> bool:
    signal = row.get("signal") if isinstance(row.get("signal"), dict) else {}
    try:
        minutes = int(str(signal.get("timeframe", "")).lower().removesuffix("m"))
        return datetime.now(timezone.utc) >= parse_dt(row.get("order_started_at")) + timedelta(minutes=minutes + 1)
    except (TypeError, ValueError):
        return False


def settle_order(row: dict[str, Any], closes: dict[str, dict[int, float]]) -> dict[str, Any] | None:
    signal = row.get("signal") if isinstance(row.get("signal"), dict) else {}
    order = row.get("order") if isinstance(row.get("order"), dict) else {}
    try:
        minutes = int(str(signal.get("timeframe", "")).lower().removesuffix("m"))
        order_time = parse_dt(row.get("order_started_at"))
        start_time = completed_kline_open_time(order_time)
        end_time = completed_kline_open_time(order_time + timedelta(minutes=minutes))
        start_close = closes.get(normalize_symbol(signal.get("symbol", "")), {}).get(to_ms(start_time))
        end_close = closes.get(normalize_symbol(signal.get("symbol", "")), {}).get(to_ms(end_time))
    except (TypeError, ValueError):
        return None
    if start_close is None or end_close is None or start_close == end_close:
        return None
    up = str(signal.get("side", "")).upper() == "BUY"
    win = end_close > start_close if up else end_close < start_close
    amount = positive_float(order.get("amount"), 0.0)
    payout_rate = positive_float(row.get("payout_rate"), 0.0)
    return {
        "signal_id": signal_id_for(row),
        "settled_at": datetime.now(timezone.utc).isoformat(),
        "outcome": "win" if win else "loss",
        "pnl": amount * payout_rate if win else -amount,
    }


def signal_id_for(row: dict[str, Any]) -> str:
    signal = row.get("signal") if isinstance(row.get("signal"), dict) else {}
    order = row.get("order") if isinstance(row.get("order"), dict) else {}
    return str(signal.get("signal_id") or order.get("signal_id") or "")


def symbols_for(records: list[dict[str, Any]]) -> set[str]:
    return {normalize_symbol((row.get("signal") or {}).get("symbol", "")) for row in records}


def read_live_closes(data_root: Path, symbol: str) -> dict[int, float]:
    path = data_root / "binance_spot_klines" / symbol / "1m_live.csv"
    closes: dict[int, float] = {}
    if not path.exists():
        return closes
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                closes[int(row["open_time"])] = float(row["close"])
            except (KeyError, TypeError, ValueError):
                continue
    return closes


def completed_kline_open_time(value: datetime) -> datetime:
    return value.replace(second=0, microsecond=0) - timedelta(minutes=1)


def to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def is_stale_signal(signal: Signal) -> bool:
    try:
        age = (datetime.now(timezone.utc) - parse_dt(signal.decision_time)).total_seconds()
    except ValueError:
        return True
    return age > MAX_SIGNAL_AGE_SECONDS


def stale_record(row: dict[str, Any]) -> bool:
    signal = row.get("signal") if isinstance(row.get("signal"), dict) else {}
    try:
        age = (parse_dt(row.get("order_started_at")) - parse_dt(signal.get("decision_time"))).total_seconds()
    except (TypeError, ValueError):
        return False
    return age > MAX_SIGNAL_AGE_SECONDS


def parse_dt(value: Any) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def self_test() -> None:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        data_root = root / "aligned_data_oos"
        settlements = root / "settlements.jsonl"
        decision = datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc)
        live = data_root / "binance_spot_klines" / "BTCUSDT" / "1m_live.csv"
        live.parent.mkdir(parents=True)
        live.write_text(
            "open_time,close\n"
            f"{to_ms(decision - timedelta(minutes=1))},100\n"
            f"{to_ms(decision + timedelta(minutes=4))},101\n",
            encoding="utf-8",
        )
        log = root / "orders.jsonl"
        row = {
            "success": True,
            "signal": {
                "signal_id": "x",
                "symbol": "BTCUSDT",
                "timeframe": "5m",
                "side": "BUY",
                "decision_time": decision.isoformat(),
                "last_kline_time": (decision - timedelta(minutes=1)).isoformat(),
            },
            "order_started_at": (decision + timedelta(seconds=1)).isoformat(),
            "order": {"amount": "3"},
            "payout_rate": 0.8,
        }
        settlements.write_text(
            json.dumps({"signal_id": "x", "pnl": 2.4}) + "\n"
            + json.dumps({"signal_id": "y", "pnl": -3.0}) + "\n",
            encoding="utf-8",
        )
        log.write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert abs(effective_bankroll(False, settlements, 200.0) - 199.4) < 1e-9
        stale_row = {**row, "order_started_at": (decision + timedelta(seconds=20)).isoformat()}
        log.write_text(json.dumps(row) + "\n" + json.dumps(stale_row) + "\n", encoding="utf-8")
        assert abs(effective_bankroll(False, settlements, 200.0) - 199.4) < 1e-9
        assert effective_bankroll(True, settlements, 200.0) == 200.0
        assert abs(settle_order(row, {"BTCUSDT": read_live_closes(data_root, "BTCUSDT")})["pnl"] - 2.4) < 1e-9


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
    bankroll: float,
    bankroll_source: str,
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
        "bankroll": bankroll,
        "bankroll_source": bankroll_source,
    }
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
