#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from hibt_config import HibtConfig, normalize_symbol
from hibt_signal_reader import TradingSignal

if TYPE_CHECKING:
    from hibt_browser import HibtBrowser, OrderResult


@dataclass
class JournalState:
    processed_signals: list[str] = field(default_factory=list)
    traded_candles: list[str] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    consecutive_failures: int = 0


class TradeJournal:
    def __init__(self, state_path: Path, log_path: Path) -> None:
        self.state_path = state_path
        self.log_path = log_path
        self.state = self._load()

    def can_trade(self, signal: TradingSignal, config: HibtConfig) -> tuple[bool, str]:
        if signal.key in self.state.processed_signals:
            return False, f"signal already processed: {signal.key}"
        if normalize_symbol(signal.symbol) not in config.symbols:
            return False, f"symbol {signal.symbol} not allowed"
        if not signal.is_trade:
            return False, f"signal is {signal.side}"
        if signal.confidence < config.risk.min_confidence:
            return False, f"confidence {signal.confidence:.4f} < {config.risk.min_confidence:.4f}"
        age = (datetime.now(timezone.utc) - signal.timestamp.astimezone(timezone.utc)).total_seconds()
        if age > config.risk.max_signal_age_seconds:
            return False, f"signal too old: {age:.1f}s"
        if self.state.consecutive_failures >= config.risk.max_consecutive_failures:
            return False, f"consecutive failures reached {self.state.consecutive_failures}"
        if config.risk.one_order_per_symbol_per_candle:
            key = candle_key(signal, config.risk.candle_minutes)
            if key in self.state.traded_candles:
                return False, f"already traded candle {key}"
        day_orders = self._orders_for_day()
        if len(day_orders) >= config.risk.max_orders_per_day:
            return False, "daily order limit reached"
        symbol_orders = [item for item in day_orders if item.get("symbol") == normalize_symbol(signal.symbol)]
        if len(symbol_orders) >= config.risk.max_orders_per_symbol_per_day:
            return False, f"daily symbol order limit reached for {signal.symbol}"
        last_symbol_order = next((item for item in reversed(self.state.orders) if item.get("symbol") == normalize_symbol(signal.symbol)), None)
        if last_symbol_order:
            last_ts = datetime.fromisoformat(last_symbol_order["timestamp"])
            cooldown = (datetime.now(timezone.utc) - last_ts.astimezone(timezone.utc)).total_seconds()
            if cooldown < config.risk.cooldown_seconds:
                return False, f"cooldown active: {cooldown:.1f}s < {config.risk.cooldown_seconds}s"
        return True, "ok"

    def record_skip(self, signal: TradingSignal, reason: str) -> None:
        self._mark_processed(signal)
        self._save()
        self._append_log("skip", signal, {"reason": reason})

    def record_order(self, signal: TradingSignal, result: OrderResult, config: HibtConfig) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": normalize_symbol(signal.symbol),
            "side": signal.side,
            "signal_timestamp": signal.timestamp.isoformat(),
            "confidence": signal.confidence,
            "status": result.status,
            "message": result.message,
            "dry_run": config.dry_run,
        }
        self.state.orders.append(payload)
        self._mark_processed(signal)
        if result.status in {"dry_run", "submitted", "submitted_without_modal", "needs_manual_confirm"}:
            self.state.consecutive_failures = 0
            if config.risk.one_order_per_symbol_per_candle and result.status != "dry_run":
                self.state.traded_candles.append(candle_key(signal, config.risk.candle_minutes))
                self.state.traded_candles = self.state.traded_candles[-5000:]
        self._save()
        self._append_log("order", signal, payload)

    def record_failure(self, signal: TradingSignal | None, error: Exception) -> None:
        self.state.consecutive_failures += 1
        if signal is not None:
            self._mark_processed(signal)
        self._save()
        self._append_log("failure", signal, {"error": str(error), "type": type(error).__name__})

    def _orders_for_day(self) -> list[dict[str, Any]]:
        tz = ZoneInfo("Asia/Shanghai")
        today = datetime.now(tz).date()
        result = []
        for item in self.state.orders:
            ts = datetime.fromisoformat(item["timestamp"])
            if ts.astimezone(tz).date() == today and item.get("status") not in {"dry_run"}:
                result.append(item)
        return result

    def _load(self) -> JournalState:
        if not self.state_path.exists():
            return JournalState()
        with self.state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return JournalState(
            processed_signals=payload.get("processed_signals", []),
            traded_candles=payload.get("traded_candles", []),
            orders=payload.get("orders", []),
            consecutive_failures=payload.get("consecutive_failures", 0),
        )

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.state), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _append_log(self, event: str, signal: TradingSignal | None, payload: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "signal": asdict(signal) if signal is not None else None,
            "payload": payload,
        }
        if signal is not None:
            record["signal"]["timestamp"] = signal.timestamp.isoformat()
        with self.log_path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, default=str)
            handle.write("\n")

    def _mark_processed(self, signal: TradingSignal) -> None:
        if signal.key not in self.state.processed_signals:
            self.state.processed_signals.append(signal.key)
            self.state.processed_signals = self.state.processed_signals[-10_000:]


class HibtTrader:
    def __init__(self, config: HibtConfig) -> None:
        self.config = config
        self.journal = TradeJournal(config.state_path, config.log_path)

    def handle_signal(self, browser: "HibtBrowser", signal: TradingSignal) -> "OrderResult | None":
        allowed, reason = self.journal.can_trade(signal, self.config)
        if not allowed:
            self.journal.record_skip(signal, reason)
            print(f"skip {signal.symbol} {signal.side}: {reason}", flush=True)
            return None
        try:
            result = browser.execute_order(signal.symbol, signal.side, duration_label_for_signal(signal, self.config))
            self.journal.record_order(signal, result, self.config)
            print(f"{result.status} {signal.symbol} {signal.side}: {result.message}", flush=True)
            return result
        except Exception as exc:
            self.journal.record_failure(signal, exc)
            setattr(exc, "hibt_signal_recorded", True)
            raise


def candle_key(signal: TradingSignal, candle_minutes: int) -> str:
    ts = signal.timestamp.astimezone(timezone.utc)
    timeframe = normalize_timeframe(signal.timeframe)
    minutes = timeframe_minutes(timeframe) or candle_minutes
    minute = (ts.minute // minutes) * minutes
    start = ts.replace(minute=minute, second=0, microsecond=0)
    return f"{normalize_symbol(signal.symbol)}:{timeframe or 'na'}:{start.isoformat()}"


def duration_label_for_signal(signal: TradingSignal, config: HibtConfig) -> str:
    timeframe = normalize_timeframe(signal.timeframe)
    if timeframe and timeframe in config.duration_labels:
        return config.duration_labels[timeframe]
    return config.duration_label


def normalize_timeframe(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().lower().replace(" ", "")
    if not text:
        return None
    if text.endswith("min"):
        text = text[:-3] + "m"
    return text


def timeframe_minutes(value: str | None) -> int | None:
    timeframe = normalize_timeframe(value)
    if timeframe is None or not timeframe.endswith("m"):
        return None
    try:
        minutes = int(timeframe[:-1])
    except ValueError:
        return None
    return minutes if minutes > 0 else None
