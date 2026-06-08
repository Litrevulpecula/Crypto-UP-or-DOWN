#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


API_URL = "https://api-ws.taichuwuji.com/event/event-order/place"
TIME_UNITS = {"3m": 3, "5m": 5, "10m": 10, "15m": 15}
SYMBOLS = {"BTC-USDT": "btc_usdt", "ETH-USDT": "eth_usdt"}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class HibtApiCredentials:
    v: str
    authorization: str
    x_auth_token: str

    @classmethod
    def from_env(cls) -> "HibtApiCredentials":
        v = os.environ.get("HIBT_API_V") or ""
        authorization = os.environ.get("HIBT_AUTHORIZATION") or ""
        x_auth_token = os.environ.get("HIBT_X_AUTH_TOKEN") or ""
        missing = []
        if not v:
            missing.append("HIBT_API_V")
        if not authorization:
            missing.append("HIBT_AUTHORIZATION")
        if not x_auth_token:
            missing.append("HIBT_X_AUTH_TOKEN")
        if missing:
            raise RuntimeError(f"missing HiBT API env vars: {', '.join(missing)}")
        return cls(v=normalize_v(v), authorization=authorization, x_auth_token=x_auth_token)


def normalize_timeframe(value: str) -> str:
    text = str(value or "").strip().lower()
    if text not in TIME_UNITS:
        raise ValueError(f"unsupported HiBT timeframe: {value!r}")
    return text


def normalize_v(value: str) -> str:
    return urllib.parse.unquote(str(value or "").strip())


def timeframe_to_time_unit(value: str) -> int:
    return TIME_UNITS[normalize_timeframe(value)]


def symbol_to_api(value: str) -> str:
    symbol = str(value or "").strip().upper()
    if symbol not in SYMBOLS:
        raise ValueError(f"unsupported HiBT symbol: {value!r}")
    return SYMBOLS[symbol]


def place_order(
    *,
    credentials: HibtApiCredentials,
    amount: str,
    direction: int,
    symbol: str,
    timeframe: str,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    query = urllib.parse.urlencode({"v": credentials.v})
    url = f"{API_URL}?{query}"
    payload = urllib.parse.urlencode(
        {
            "amount": str(amount),
            "direction": str(direction),
            "symbol": symbol_to_api(symbol),
            "timeUnit": str(timeframe_to_time_unit(timeframe)),
            "langCode": "zh_CN",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers=request_headers(credentials),
    )
    return send_request(request, timeout_seconds)


def check_auth(*, credentials: HibtApiCredentials, timeout_seconds: float = 20.0) -> dict[str, Any]:
    query = urllib.parse.urlencode({"v": credentials.v})
    payload = urllib.parse.urlencode(
        {
            "amount": "0",
            "direction": "9",
            "symbol": "__probe__",
            "timeUnit": "999",
            "langCode": "zh_CN",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{API_URL}?{query}",
        data=payload,
        method="POST",
        headers=request_headers(credentials),
    )
    return send_request(request, timeout_seconds)


def request_headers(credentials: HibtApiCredentials) -> dict[str, str]:
    return {
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9",
        "client-type": "web",
        "content-type": "application/x-www-form-urlencoded",
        "future_source": "1",
        "hc-language": "zh_CN",
        "hc-platform": "web",
        "lang": "zh_CN",
        "origin": "https://hibt29.com",
        "platform": "PC",
        "referer": "https://hibt29.com/",
        "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "cross-site",
        "user-agent": USER_AGENT,
        "authorization": credentials.authorization,
        "x-auth-token": credentials.x_auth_token,
    }


def send_request(request: urllib.request.Request, timeout_seconds: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "http_status": response.status,
                "body": _parse_json_or_text(body),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "http_status": exc.code,
            "body": _parse_json_or_text(body),
        }
    except urllib.error.URLError as exc:
        return {
            "http_status": None,
            "body": {"error": str(exc.reason)},
        }


def _parse_json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
