#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hibt_config import normalize_symbol


BUY_SIGNALS = {"BUY", "UP", "LONG", "CALL", "BULL", "RISE", "1"}
SELL_SIGNALS = {"SELL", "DOWN", "SHORT", "PUT", "BEAR", "FALL", "-1"}
HOLD_SIGNALS = {"HOLD", "WAIT", "NONE", "FLAT", "NO_TRADE", "0", ""}


@dataclass(frozen=True)
class TradingSignal:
    symbol: str
    side: str
    raw_signal: str
    confidence: float
    timestamp: datetime
    source: str = "file"
    event_id: str | None = None
    timeframe: str | None = None

    @property
    def is_trade(self) -> bool:
        return self.side in {"up", "down"}

    @property
    def key(self) -> str:
        if self.event_id:
            return f"{self.symbol}:event:{self.event_id}"
        timestamp = self.timestamp.astimezone(timezone.utc).isoformat()
        timeframe = self.timeframe or "na"
        return f"{self.symbol}:{timeframe}:{timestamp}:{self.side}:{self.raw_signal}:{self.confidence:.8f}"


class SignalReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._last_signature: tuple[int, int] | None = None

    def read(self) -> list[TradingSignal]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return parse_signal_payload(payload, source=str(self.path))

    def read_if_changed(self) -> list[TradingSignal]:
        if not self.path.exists():
            self._last_signature = None
            return []
        stat = self.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._last_signature:
            return []
        self._last_signature = signature
        return self.read()


def parse_signal_payload(payload: Any, source: str = "payload") -> list[TradingSignal]:
    records: list[dict[str, Any]]
    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("signals"), list):
        records = [item for item in payload["signals"] if isinstance(item, dict)]
    elif isinstance(payload, dict) and _looks_like_single_signal(payload):
        records = [payload]
    elif isinstance(payload, dict):
        records = []
        for key, value in payload.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("symbol", key)
                records.append(item)
    else:
        raise ValueError(f"Unsupported signal payload in {source}: {type(payload)!r}")

    signals = [_parse_one(item, source) for item in records]
    return sorted(signals, key=lambda item: item.timestamp)


def _looks_like_single_signal(payload: dict[str, Any]) -> bool:
    return "symbol" in payload and any(key in payload for key in ("signal", "side", "direction"))


def _parse_one(item: dict[str, Any], source: str) -> TradingSignal:
    symbol = normalize_symbol(str(item.get("symbol", "")))
    raw_signal = str(item.get("signal", item.get("side", item.get("direction", "")))).strip()
    side = normalize_side(raw_signal)
    confidence = float(item.get("confidence", item.get("probability", item.get("score", 1.0))))
    timestamp = parse_timestamp(item.get("timestamp", item.get("time", item.get("created_at"))))
    signal_id = item.get("id", item.get("signal_id", item.get("event_id")))
    timeframe = normalize_timeframe(item.get("timeframe", item.get("interval", item.get("period", item.get("horizon")))))
    signal = TradingSignal(
        symbol=symbol,
        side=side,
        raw_signal=raw_signal,
        confidence=confidence,
        timestamp=timestamp,
        source=source,
        event_id=str(signal_id) if signal_id is not None else None,
        timeframe=timeframe,
    )
    return signal


def normalize_side(value: str) -> str:
    normalized = value.strip().upper().replace(" ", "_")
    if normalized in BUY_SIGNALS:
        return "up"
    if normalized in SELL_SIGNALS:
        return "down"
    if normalized in HOLD_SIGNALS:
        return "hold"
    raise ValueError(f"Unknown trading signal: {value!r}")


def normalize_timeframe(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return None
    aliases = {
        "3": "3m",
        "3m": "3m",
        "3min": "3m",
        "3mins": "3m",
        "3minute": "3m",
        "3minutes": "3m",
        "3分钟": "3m",
        "5": "5m",
        "5m": "5m",
        "5min": "5m",
        "5mins": "5m",
        "5minute": "5m",
        "5minutes": "5m",
        "5分钟": "5m",
        "10": "10m",
        "10m": "10m",
        "10min": "10m",
        "10mins": "10m",
        "10minute": "10m",
        "10minutes": "10m",
        "10分钟": "10m",
        "15": "15m",
        "15m": "15m",
        "15min": "15m",
        "15mins": "15m",
        "15minute": "15m",
        "15minutes": "15m",
        "15分钟": "15m",
    }
    return aliases.get(text, text)


def parse_timestamp(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
