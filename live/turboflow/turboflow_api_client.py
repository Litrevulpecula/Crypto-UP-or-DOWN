#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


API_BASE = "https://apis.turboflow.xyz"
CONFIG_PATH = "/public/pm/config?version=2"
SUBMIT_PATH = "/account/pm/order/submit"
TIMEFRAME_SECONDS = {"30s": 30, "1m": 60, "3m": 180, "5m": 300, "10m": 600, "15m": 900, "1h": 3600}
PAIR_IDS = {"ETH-USDT": "5", "BTC-USDT": "6"}
ORDER_WAY = {"BUY": 1, "SELL": 3}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class TurboFlowCredentials:
    token: str
    uid: str
    coin_code: str
    pool_id: str

    @classmethod
    def from_env(cls) -> "TurboFlowCredentials":
        values = {
            "token": os.environ.get("TURBOFLOW_TOKEN") or "",
            "uid": os.environ.get("TURBOFLOW_UID") or "",
            "coin_code": os.environ.get("TURBOFLOW_COIN_CODE") or "",
            "pool_id": os.environ.get("TURBOFLOW_POOL_ID") or "",
        }
        missing = [f"TURBOFLOW_{name.upper()}" for name, value in values.items() if not value]
        if missing:
            raise RuntimeError(f"missing TurboFlow env vars: {', '.join(missing)}")
        return cls(**values)


def normalize_timeframe(value: str) -> str:
    text = str(value or "").strip().lower().replace("min", "m")
    if text.isdigit():
        text = f"{text}m"
    if text not in TIMEFRAME_SECONDS:
        raise ValueError(f"unsupported TurboFlow timeframe: {value!r}")
    return text


def timeframe_seconds(value: str) -> int:
    return TIMEFRAME_SECONDS[normalize_timeframe(value)]


def pair_id_for(symbol: str) -> str:
    text = str(symbol or "").strip().upper().replace("/", "-").replace("_", "-")
    if text in {"BTC", "ETH"}:
        text = f"{text}-USDT"
    if text.endswith("USDT") and "-" not in text:
        text = f"{text[:-4]}-USDT"
    try:
        return PAIR_IDS[text]
    except KeyError as exc:
        raise ValueError(f"unsupported TurboFlow symbol: {symbol!r}") from exc


def return_rate_for(*, symbol: str, timeframe: str, side: str, timeout_seconds: float = 8.0) -> str:
    response = request_json("GET", CONFIG_PATH, timeout_seconds=timeout_seconds)
    body = response.get("body")
    rows = ((body.get("data") or {}).get("data") or []) if isinstance(body, dict) else []
    pair_id = pair_id_for(symbol)
    duration = timeframe_seconds(timeframe)
    for row in rows:
        if str(row.get("pair_id")) != pair_id:
            continue
        for config in row.get("order_configs") or []:
            if int(config.get("duration") or 0) != duration:
                continue
            key = "bid_return_rate" if side == "BUY" else "ask_return_rate"
            value = config.get(key)
            if value in (None, ""):
                raise RuntimeError(f"TurboFlow return rate missing: pair_id={pair_id} duration={duration} side={side}")
            return str(value)
    raise RuntimeError(f"TurboFlow config missing: pair_id={pair_id} duration={duration}")


def build_order_fields(
    *,
    credentials: TurboFlowCredentials,
    amount: str,
    symbol: str,
    timeframe: str,
    side: str,
    return_rate: str,
) -> dict[str, Any]:
    signal_side = str(side or "").strip().upper()
    if signal_side not in ORDER_WAY:
        raise ValueError(f"unsupported TurboFlow side: {side!r}")
    return {
        "account_id": credentials.uid,
        "amount": str(amount),
        "coin_code": credentials.coin_code,
        "duration": timeframe_seconds(timeframe),
        "order_way": ORDER_WAY[signal_side],
        "pair_id": pair_id_for(symbol),
        "pool_id": credentials.pool_id,
        "return_rate": float(return_rate),
    }


def place_order(
    *,
    credentials: TurboFlowCredentials,
    amount: str,
    symbol: str,
    timeframe: str,
    side: str,
    timeout_seconds: float = 15.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rate = return_rate_for(symbol=symbol, timeframe=timeframe, side=side)
    payload = build_order_fields(
        credentials=credentials,
        amount=amount,
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        return_rate=rate,
    )
    return payload, request_json(
        "POST",
        SUBMIT_PATH,
        payload=payload,
        credentials=credentials,
        timeout_seconds=timeout_seconds,
    )


def request_json(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    credentials: TurboFlowCredentials | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    base = os.environ.get("TURBOFLOW_API_BASE") or API_BASE
    url = urllib.parse.urljoin(base, path)
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=request_headers(credentials))
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return {"http_status": response.status, "body": parse_json_or_text(raw)}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {"http_status": exc.code, "body": parse_json_or_text(raw)}
    except urllib.error.URLError as exc:
        return {"http_status": None, "body": {"error": str(exc.reason)}}


def request_headers(credentials: TurboFlowCredentials | None) -> dict[str, str]:
    headers = {
        "accept": "application/json, text/plain, */*",
        "content-type": "application/json",
        "origin": "https://www.turboflow.xyz",
        "referer": "https://www.turboflow.xyz/",
        "user-agent": USER_AGENT,
        "Biz-pf": "6",
        "LANG": "zh-cn",
    }
    if credentials is not None:
        headers["Authorization"] = f"Bearer {credentials.token}"
        headers["Uid"] = credentials.uid
    return headers


def parse_json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
