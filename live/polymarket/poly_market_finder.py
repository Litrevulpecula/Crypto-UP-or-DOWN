#!/usr/bin/env python3
"""Find active Polymarket crypto Up/Down markets and print CLOB token IDs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

GAMMA_API = "https://gamma-api.polymarket.com"

SYMBOL_SLUGS = {
    "BTC": "btc",
    "BTCUSDT": "btc",
    "BTC-USD": "btc",
    "BTC-USD-SWAP-LIN": "btc",
    "ETH": "eth",
    "ETHUSDT": "eth",
    "ETH-USD": "eth",
    "ETH-USD-SWAP-LIN": "eth",
}


def normalize_symbol_key(value: str) -> str:
    text = value.upper().strip().replace("/", "-")
    compact = text.replace("-", "")
    if text in SYMBOL_SLUGS:
        return SYMBOL_SLUGS[text]
    if compact in SYMBOL_SLUGS:
        return SYMBOL_SLUGS[compact]
    if "BTC" in compact:
        return "btc"
    if "ETH" in compact:
        return "eth"
    raise ValueError(f"Unsupported Polymarket crypto symbol: {value!r}")


def normalize_timeframe(value: str | None) -> str:
    if value is None:
        raise ValueError("timeframe is required")
    text = str(value).strip().lower().replace(" ", "")
    if text.endswith("mins"):
        text = text[:-4] + "m"
    elif text.endswith("min"):
        text = text[:-3] + "m"
    elif text.endswith("minutes"):
        text = text[:-7] + "m"
    elif text.endswith("minute"):
        text = text[:-6] + "m"
    elif text.isdigit():
        text = f"{text}m"
    if text not in {"5m", "15m"}:
        raise ValueError(f"Unsupported Polymarket timeframe: {value!r}")
    return text


def parse_utc_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        timestamp = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        timestamp = datetime.fromisoformat(text)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def parse_json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def fetch_gamma_markets(gamma_api: str = GAMMA_API, limit: int = 200) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "limit": str(limit),
            "active": "true",
            "closed": "false",
            "order": "startDate",
            "ascending": "false",
            "tag_slug": "crypto",
        }
    )
    url = f"{gamma_api.rstrip('/')}/markets?{query}"
    request = Request(url, headers={"User-Agent": "Crypto_UP_or_DOWN/1.0"})
    last_error: URLError | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=10) as response:
                payload = json.load(response)
            break
        except URLError as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    else:
        raise last_error if last_error else RuntimeError("Gamma markets request failed")
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Gamma markets payload: {type(payload).__name__}")
    return [item for item in payload if isinstance(item, dict)]


def is_tradable_market(market: dict[str, Any]) -> bool:
    return (
        bool(market.get("active"))
        and not bool(market.get("closed"))
        and not bool(market.get("archived"))
        and bool(market.get("acceptingOrders"))
        and bool(market.get("enableOrderBook"))
    )


def market_window(market: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    start = parse_utc_timestamp(market.get("eventStartTime") or market.get("startTime"))
    end = parse_utc_timestamp(market.get("endDate"))
    if start is None:
        events = market.get("events")
        if isinstance(events, list) and events and isinstance(events[0], dict):
            start = parse_utc_timestamp(events[0].get("startTime"))
    return start, end


def outcome_token_map(market: dict[str, Any]) -> dict[str, str]:
    outcomes = parse_json_array(market.get("outcomes"))
    token_ids = parse_json_array(market.get("clobTokenIds"))
    if len(outcomes) != len(token_ids):
        return {}
    return {str(outcome).strip().lower(): str(token_id) for outcome, token_id in zip(outcomes, token_ids)}


def summarize_market(market: dict[str, Any]) -> dict[str, Any]:
    token_map = outcome_token_map(market)
    start, end = market_window(market)
    return {
        "question": market.get("question"),
        "slug": market.get("slug"),
        "condition_id": market.get("conditionId"),
        "event_start_time": None if start is None else start.isoformat(),
        "end_time": None if end is None else end.isoformat(),
        "accepting_orders": bool(market.get("acceptingOrders")),
        "enable_order_book": bool(market.get("enableOrderBook")),
        "best_bid": market.get("bestBid"),
        "best_ask": market.get("bestAsk"),
        "fee_type": market.get("feeType"),
        "fee_schedule": market.get("feeSchedule"),
        "token_id_up": token_map.get("up"),
        "token_id_down": token_map.get("down"),
    }


def find_crypto_market(
    symbol: str,
    timeframe: str,
    target_time: datetime | str | None = None,
    markets: list[dict[str, Any]] | None = None,
    gamma_api: str = GAMMA_API,
    limit: int = 200,
    require_exact_start: bool = False,
) -> dict[str, Any] | None:
    symbol_key = normalize_symbol_key(symbol)
    tf = normalize_timeframe(timeframe)
    target = parse_utc_timestamp(target_time) or datetime.now(timezone.utc)
    slug_prefix = f"{symbol_key}-updown-{tf}-"
    candidates = []

    for market in markets if markets is not None else fetch_gamma_markets(gamma_api=gamma_api, limit=limit):
        slug = str(market.get("slug") or "")
        if not slug.startswith(slug_prefix):
            continue
        if not is_tradable_market(market):
            continue
        token_map = outcome_token_map(market)
        if "up" not in token_map or "down" not in token_map:
            continue
        start, end = market_window(market)
        if start is None or end is None:
            continue
        if require_exact_start and start != target:
            continue
        if not require_exact_start and not (start <= target < end):
            continue
        if start <= target < end:
            candidates.append((start, market))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_markets(query: str, active_only: bool = True, gamma_api: str = GAMMA_API, limit: int = 200) -> list[dict[str, Any]]:
    q_lower = query.lower()
    results = []
    for market in fetch_gamma_markets(gamma_api=gamma_api, limit=limit):
        searchable = " ".join(
            str(market.get(key) or "")
            for key in ("question", "slug", "title", "description")
        ).lower()
        if q_lower not in searchable:
            continue
        if active_only and not is_tradable_market(market):
            continue
        results.append(summarize_market(market))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Find active Polymarket crypto Up/Down markets.")
    parser.add_argument("--symbol", choices=("BTC", "ETH"), help="Crypto symbol to resolve.")
    parser.add_argument("--timeframe", "-t", help="Market timeframe, e.g. 5m or 15m.")
    parser.add_argument("--at", help="UTC timestamp for the target market window. Defaults to now.")
    parser.add_argument("--query", "-q", help="Free-text search over current Gamma markets.")
    parser.add_argument("--all", action="store_true", help="Include non-tradable results for --query.")
    parser.add_argument("--limit", type=int, default=200, help="Gamma markets request limit.")
    parser.add_argument("--gamma-api", default=GAMMA_API, help="Gamma API base URL.")
    args = parser.parse_args()

    if args.symbol or args.timeframe:
        if not args.symbol or not args.timeframe:
            parser.error("--symbol and --timeframe must be used together")
        market = find_crypto_market(
            args.symbol,
            args.timeframe,
            target_time=args.at,
            gamma_api=args.gamma_api,
            limit=args.limit,
        )
        if market is None:
            print("No matching active market found.", file=sys.stderr)
            return 1
        print(json.dumps(summarize_market(market), indent=2))
        return 0

    if not args.query:
        parser.error("use --symbol/--timeframe or --query")

    results = find_markets(args.query, active_only=not args.all, gamma_api=args.gamma_api, limit=args.limit)
    if not results:
        print("No markets found.", file=sys.stderr)
        return 1
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
