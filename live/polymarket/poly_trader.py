#!/usr/bin/env python3
"""Polymarket CLOB trader — reads signals.json from the shared signal pipeline
and executes market orders on Polymarket's 5-minute / 15-minute crypto markets.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from py_clob_client_v2.client import ClobClient
    from py_clob_client_v2.clob_types import ApiCreds, MarketOrderArgs, OrderArgs, OrderPayload, OrderType

    CLOB_CLIENT_VERSION = 2
except ImportError:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, MarketOrderArgs, OrderArgs, OrderType

    OrderPayload = None
    CLOB_CLIENT_VERSION = 1

try:
    from poly_market_finder import find_crypto_market, outcome_token_map, parse_utc_timestamp, summarize_market
except ImportError:
    from .poly_market_finder import find_crypto_market, outcome_token_map, parse_utc_timestamp, summarize_market

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LIVE_DIR = Path(__file__).resolve().parent
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon
BUY_SIGNALS = {"BUY", "UP", "LONG", "CALL", "BULL", "RISE", "1"}
SELL_SIGNALS = {"SELL", "DOWN", "SHORT", "PUT", "BEAR", "FALL", "-1"}
HOLD_SIGNALS = {"HOLD", "WAIT", "NONE", "FLAT", "NO_TRADE", "0", ""}
MIN_ORDER_SHARES = 5.0


@dataclass
class PolyConfig:
    private_key: str = ""
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    funder: str = ""
    signature_type: int | str | None = None
    signal_path: Path = LIVE_DIR.parent / "signals.json"
    state_path: Path = LIVE_DIR / "runtime" / "poly_state.json"
    log_path: Path = LIVE_DIR / "runtime" / "poly_orders.jsonl"
    order_shares: float = MIN_ORDER_SHARES
    max_price: float = 0.544
    use_market_fee_rate: bool = True
    default_taker_fee_rate_bps: int = 700
    poll_seconds: float = 2.0
    dry_run: bool = True
    auto_resolve_markets: bool = True
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    market_lookup_limit: int = 200
    market_slug_btc_5m: str = ""
    market_slug_eth_5m: str = ""
    market_slug_btc_15m: str = ""
    market_slug_eth_15m: str = ""
    token_id_btc_5m_up: str = ""
    token_id_btc_5m_down: str = ""
    token_id_eth_5m_up: str = ""
    token_id_eth_5m_down: str = ""
    token_id_btc_15m_up: str = ""
    token_id_btc_15m_down: str = ""
    token_id_eth_15m_up: str = ""
    token_id_eth_15m_down: str = ""
    # Legacy 5m aliases kept for existing local configs.
    token_id_btc_up: str = ""
    token_id_btc_down: str = ""
    token_id_eth_up: str = ""
    token_id_eth_down: str = ""

    @classmethod
    def load(cls, path: Path | None) -> "PolyConfig":
        config = cls()
        if path and path.exists():
            with path.open("r") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(config, k):
                    if isinstance(getattr(config, k), Path):
                        value_path = Path(v)
                        if not value_path.is_absolute():
                            value_path = path.parent / value_path
                        setattr(config, k, value_path)
                    else:
                        setattr(config, k, v)
        return config


class PolyTrader:
    def __init__(self, config: PolyConfig) -> None:
        self.config = config
        self.client: ClobClient | None = None
        self._consumed_ids: set[str] = set()
        self._market_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._load_state()

    def connect(self) -> None:
        cfg = self.config
        creds = None
        if cfg.api_key:
            creds = ApiCreds(
                api_key=cfg.api_key,
                api_secret=cfg.api_secret,
                api_passphrase=cfg.api_passphrase,
            )
        signature_type = normalize_signature_type(cfg.signature_type)
        funder = cfg.funder.strip() or None
        self.client = ClobClient(
            host=HOST,
            chain_id=CHAIN_ID,
            key=cfg.private_key,
            creds=creds,
            signature_type=signature_type,
            funder=funder,
        )
        if not creds and cfg.private_key:
            if CLOB_CLIENT_VERSION == 2:
                self.client.set_api_creds(self.client.create_or_derive_api_key())
            else:
                self.client.set_api_creds(self.client.create_or_derive_api_creds())
        log.info(
            "Connected to Polymarket CLOB (dry_run=%s sdk_v=%s signature_type=%s funder=%s)",
            cfg.dry_run,
            CLOB_CLIENT_VERSION,
            signature_type,
            short_id(funder),
        )

    def resolve_token_id(
        self,
        symbol: str,
        direction: str,
        timeframe: str | None = None,
        target_time: Any = None,
    ) -> str | None:
        sym = symbol.upper().replace("-", "").replace("/", "")
        tf = normalize_timeframe(timeframe) or "5m"
        side = direction.lower()

        if self.config.auto_resolve_markets:
            token = self.resolve_live_market_token_id(symbol, side, tf, target_time)
            if token:
                return token

        if "BTC" in sym:
            token = getattr(self.config, f"token_id_btc_{tf}_{side}", "")
            if token:
                return token
            if tf == "5m":
                return self.config.token_id_btc_up if side == "up" else self.config.token_id_btc_down
        if "ETH" in sym:
            token = getattr(self.config, f"token_id_eth_{tf}_{side}", "")
            if token:
                return token
            if tf == "5m":
                return self.config.token_id_eth_up if side == "up" else self.config.token_id_eth_down
        return None

    def resolve_live_market_token_id(self, symbol: str, side: str, timeframe: str, target_time: Any) -> str | None:
        target = parse_utc_timestamp(target_time) or datetime.now(timezone.utc)
        cache_key = (symbol.upper(), timeframe, target.isoformat())
        cached = self._market_cache.get(cache_key)
        if cached:
            return outcome_token_map(cached).get(side)

        try:
            market = find_crypto_market(
                symbol,
                timeframe,
                target_time=target,
                gamma_api=self.config.gamma_api_url,
                limit=int(self.config.market_lookup_limit),
                require_exact_start=True,
            )
        except Exception as e:
            log.warning("Failed to resolve Polymarket %s %s market at %s: %s", symbol, timeframe, target.isoformat(), e)
            return None
        if not market:
            log.warning("No active Polymarket %s %s market found for %s", symbol, timeframe, target.isoformat())
            return None

        token = outcome_token_map(market).get(side)
        if not token:
            log.warning("Polymarket market %s missing %s token", market.get("slug"), side)
            return None

        self._market_cache[cache_key] = market
        summary = summarize_market(market)
        log.info(
            "Resolved Polymarket market: %s %s %s slug=%s start=%s end=%s token=%s",
            symbol,
            timeframe,
            side,
            summary.get("slug"),
            summary.get("event_start_time"),
            summary.get("end_time"),
            short_id(token),
        )
        return token

    def get_best_bid(self, token_id: str) -> float | None:
        try:
            book = self.client.get_order_book(token_id)
            if book and book.bids:
                return float(book.bids[0].price)
        except Exception as e:
            log.warning("Failed to get bid for %s: %s", token_id[:12], e)
        return None

    def get_market_price(self, token_id: str) -> float | None:
        try:
            book = self.client.get_order_book(token_id)
            if book and book.asks:
                return float(book.asks[0].price)
        except Exception as e:
            log.warning("Failed to get price for %s: %s", token_id[:12], e)
        return None

    def get_taker_fee_rate_bps(self, token_id: str) -> int:
        if not self.config.use_market_fee_rate:
            return int(self.config.default_taker_fee_rate_bps)
        try:
            return int(self.client.get_fee_rate_bps(token_id))
        except Exception as e:
            fallback = int(self.config.default_taker_fee_rate_bps)
            log.warning("Failed to get market fee rate for %s: %s; using fallback %d bps", token_id[:12], e, fallback)
            return fallback

    def _place_limit_order(self, token_id: str, price: float, size: float, fee_rate_bps: int) -> str | None:
        kwargs = {
            "token_id": token_id,
            "price": price,
            "size": size,
            "side": "BUY",
        }
        if CLOB_CLIENT_VERSION == 1:
            kwargs["fee_rate_bps"] = fee_rate_bps
        order_args = OrderArgs(
            **kwargs,
        )
        order = self.client.create_order(order_args)
        resp = self._post_order(order, OrderType.GTC)
        order_id = resp.get("orderID") if isinstance(resp, dict) else None
        log.info("Limit order posted: price=%.3f size=%.1f id=%s", price, size, order_id)
        return order_id

    def _post_order(self, order: Any, order_type: OrderType) -> dict | None:
        if CLOB_CLIENT_VERSION == 2:
            return self.client.post_order(order, order_type=order_type)
        return self.client.post_order(order, orderType=order_type)

    def _cancel_order(self, order_id: str) -> Any:
        if CLOB_CLIENT_VERSION == 2:
            return self.client.cancel_order(OrderPayload(orderID=order_id))
        return self.client.cancel(order_id)

    def _place_market_order(self, token_id: str, amount: float, fee_rate_bps: int) -> dict | None:
        kwargs = {
            "token_id": token_id,
            "amount": amount,
            "side": "BUY",
        }
        if CLOB_CLIENT_VERSION == 1:
            kwargs["fee_rate_bps"] = fee_rate_bps
        order_args = MarketOrderArgs(**kwargs)
        resp = self.client.create_market_order(order_args)
        return self._post_order(resp, OrderType.FOK)

    def _get_order_status(self, order_id: str) -> tuple[str, float]:
        """返回 (status, size_matched)"""
        try:
            info = self.client.get_order(order_id)
            if isinstance(info, dict):
                status = info.get("status", "")
                matched = float(info.get("size_matched", 0))
                return status, matched
        except Exception:
            pass
        return "", 0.0

    def _wait_and_watch(
        self,
        order_id: str,
        token_id: str,
        limit_price: float,
        total_size: float,
        initial_ask: float,
        taker_fee_rate_bps: int,
    ) -> dict | None:
        """监控限价单，ask 上跳时吃掉未成交部分"""
        cfg = self.config

        while True:
            status, matched = self._get_order_status(order_id)
            if status == "MATCHED":
                return {"status": "filled_limit", "price": limit_price, "size": matched}
            if status in ("CANCELED", "EXPIRED"):
                return None

            ask = self.get_market_price(token_id)
            if ask and ask > initial_ask:
                self._cancel_order(order_id)
                remaining = total_size - matched
                if remaining <= 0:
                    return {"status": "filled_limit", "price": limit_price, "size": matched}
                ask_all_in = buy_all_in_price(ask, taker_fee_rate_bps)
                if ask_all_in <= cfg.max_price:
                    remaining_usdc = remaining * ask
                    log.info(
                        "Ask jumped %.3f -> %.3f all_in=%.5f fee_bps=%d, market buy remaining %.2f shares ($%.2f)",
                        initial_ask,
                        ask,
                        ask_all_in,
                        taker_fee_rate_bps,
                        remaining,
                        remaining_usdc,
                    )
                    result = self._place_market_order(token_id, remaining_usdc, taker_fee_rate_bps)
                    return {"status": "filled_mixed", "limit_price": limit_price,
                            "limit_filled": matched, "market_price": ask, "market_all_in_price": ask_all_in,
                            "taker_fee_rate_bps": taker_fee_rate_bps, "market_result": result}
                else:
                    log.info("Ask jumped to %.3f all_in=%.5f > max %.3f, keep limit fill only",
                             ask, ask_all_in, cfg.max_price)
                    if matched > 0:
                        return {"status": "partial_limit", "price": limit_price, "size": matched}
                    return None

            time.sleep(2.0)

    def execute_signal(self, signal: dict[str, Any]) -> bool:
        raw_signal = signal.get("signal", signal.get("side", signal.get("direction", "")))
        direction = normalize_signal_side(raw_signal)
        if direction == "hold":
            return False
        symbol = signal.get("symbol", "")
        timeframe = normalize_timeframe(signal.get("timeframe", signal.get("interval", signal.get("period"))))
        target_time = signal.get("decision_time", signal.get("timestamp", signal.get("time", signal.get("created_at"))))
        token_id = self.resolve_token_id(symbol, direction, timeframe, target_time)
        if not token_id:
            log.warning("No token_id for %s %s %s", symbol, timeframe or "5m", direction)
            return False

        cfg = self.config
        bid = self.get_best_bid(token_id)
        ask = self.get_market_price(token_id)
        if bid is None or ask is None:
            log.warning("Cannot get book, skipping")
            return False

        limit_price = bid
        taker_fee_rate_bps = self.get_taker_fee_rate_bps(token_id)
        limit_all_in = buy_all_in_price(limit_price, taker_fee_rate_bps)
        if limit_all_in > cfg.max_price:
            log.info("Bid %.3f all_in=%.5f > max %.3f, skip", bid, limit_all_in, cfg.max_price)
            return False

        size = float(cfg.order_shares)
        if size < MIN_ORDER_SHARES:
            log.warning("order_shares %.2f < minimum %.2f, skipping", size, MIN_ORDER_SHARES)
            return False

        if cfg.dry_run:
            ask_all_in = buy_all_in_price(ask, taker_fee_rate_bps)
            log.info(
                "[DRY RUN] %s %s %s shares=%.2f bid=%.3f limit_all_in=%.5f ask=%.3f ask_all_in=%.5f fee_bps=%d",
                symbol,
                timeframe or "5m",
                direction,
                size,
                bid,
                limit_all_in,
                ask,
                ask_all_in,
                taker_fee_rate_bps,
            )
            self._log_order(signal, token_id, limit_price, {
                "dry_run": True,
                "order_shares": size,
                "limit_notional": size * limit_price,
                "ask_notional": size * ask,
                "limit_all_in_price": limit_all_in,
                "ask_all_in_price": ask_all_in,
                "taker_fee_rate_bps": taker_fee_rate_bps,
            })
            return True

        log.info("Posting limit buy %.2f shares at %.3f (bid=%.3f ask=%.3f)", size, limit_price, bid, ask)
        order_id = self._place_limit_order(token_id, limit_price, size, taker_fee_rate_bps)
        if not order_id:
            return False

        result = self._wait_and_watch(order_id, token_id, limit_price, size, ask, taker_fee_rate_bps)
        if result:
            log.info("Executed: %s", result)
            self._log_order(signal, token_id, limit_price, result)
            return True

        return False

    def run_loop(self) -> None:
        log.info("Starting signal poll loop (interval=%.1fs)", self.config.poll_seconds)
        while True:
            try:
                self._poll_signals()
            except KeyboardInterrupt:
                log.info("Shutting down")
                break
            except Exception as e:
                log.exception("Poll loop error: %s", e)
            time.sleep(self.config.poll_seconds)

    def _poll_signals(self) -> None:
        sig_path = Path(self.config.signal_path)
        if not sig_path.exists():
            return
        try:
            data = json.loads(sig_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        signals = extract_signal_records(data)
        for sig in signals:
            sig_id = signal_identifier(sig)
            if not sig_id or sig_id in self._consumed_ids:
                continue
            log.info("New signal: %s %s %s %s", sig.get("symbol"), sig.get("timeframe", "5m"), sig.get("signal"), sig_id)
            self.execute_signal(sig)
            self._consumed_ids.add(sig_id)
            self._save_state()

    def _load_state(self) -> None:
        sp = Path(self.config.state_path)
        if sp.exists():
            try:
                state = json.loads(sp.read_text())
                self._consumed_ids = set(state.get("consumed_ids", []))
            except (json.JSONDecodeError, OSError):
                pass

    def _save_state(self) -> None:
        sp = Path(self.config.state_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({
            "consumed_ids": list(self._consumed_ids)[-500:],
        }))

    def _log_order(self, signal: dict, token_id: str, price: float, result: Any) -> None:
        lp = Path(self.config.log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "signal": signal,
            "token_id": token_id,
            "price": price,
            "result": result,
        }
        with lp.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def extract_signal_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("signals"), list):
        return [item for item in payload["signals"] if isinstance(item, dict)]
    if isinstance(payload, dict) and looks_like_signal(payload):
        return [payload]
    if isinstance(payload, dict):
        records = []
        for key, value in payload.items():
            if isinstance(value, dict) and looks_like_signal(value):
                item = dict(value)
                item.setdefault("symbol", key)
                records.append(item)
        return records
    return []


def looks_like_signal(payload: dict[str, Any]) -> bool:
    return "symbol" in payload and any(key in payload for key in ("signal", "side", "direction"))


def signal_identifier(signal: dict[str, Any]) -> str:
    explicit = signal.get("signal_id", signal.get("id", signal.get("event_id")))
    if explicit:
        return str(explicit)
    timestamp = signal.get("timestamp", signal.get("time", signal.get("created_at")))
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    symbol = signal.get("symbol", "")
    timeframe = normalize_timeframe(signal.get("timeframe", signal.get("interval", signal.get("period")))) or "5m"
    raw_signal = signal.get("signal", signal.get("side", signal.get("direction", "")))
    return f"{symbol}:{timeframe}:{timestamp}:{raw_signal}"


def normalize_signature_type(value: int | str | None) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "0": 0,
            "eoa": 0,
            "wallet": 0,
            "1": 1,
            "proxy": 1,
            "poly_proxy": 1,
            "polymarket_proxy": 1,
            "proxy_wallet": 1,
            "2": 2,
            "gnosis": 2,
            "gnosis_safe": 2,
            "safe": 2,
            "3": 3,
            "poly_1271": 3,
            "1271": 3,
            "deposit": 3,
            "deposit_wallet": 3,
            "funder": 3,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported signature_type: {value!r}")
        result = aliases[normalized]
    else:
        result = int(value)
    if result not in {0, 1, 2, 3}:
        raise ValueError("signature_type must be 0, 1, 2, or 3.")
    if result == 3 and CLOB_CLIENT_VERSION < 2:
        raise ValueError("signature_type 3 requires py_clob_client_v2.")
    return result


def short_id(value: str | None) -> str:
    if not value:
        return ""
    text = str(value)
    if len(text) <= 14:
        return text
    return f"{text[:8]}...{text[-6:]}"


def normalize_signal_side(value: Any) -> str:
    normalized = str(value).strip().upper().replace(" ", "_")
    if normalized in BUY_SIGNALS:
        return "up"
    if normalized in SELL_SIGNALS:
        return "down"
    if normalized in HOLD_SIGNALS:
        return "hold"
    raise ValueError(f"Unknown trading signal: {value!r}")


def fee_rate_decimal(fee_rate_bps: int | float) -> float:
    value = float(fee_rate_bps)
    if value > 1.0:
        return value / 10000.0
    return value


def polymarket_buy_fee_per_share(price: float, fee_rate_bps: int | float) -> float:
    price = float(price)
    return fee_rate_decimal(fee_rate_bps) * price * (1.0 - price)


def buy_all_in_price(price: float, fee_rate_bps: int | float) -> float:
    price = float(price)
    return price + polymarket_buy_fee_per_share(price, fee_rate_bps)


def normalize_timeframe(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return None
    if text.isdigit():
        return f"{text}m"
    if text.endswith("mins"):
        text = text[:-4] + "m"
    elif text.endswith("min"):
        text = text[:-3] + "m"
    elif text.endswith("minutes"):
        text = text[:-7] + "m"
    elif text.endswith("minute"):
        text = text[:-6] + "m"
    if text in {"5", "15"}:
        text = f"{text}m"
    return text
