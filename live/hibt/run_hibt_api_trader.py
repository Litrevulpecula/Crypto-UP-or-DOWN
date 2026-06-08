#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import smtplib
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import urllib.error
import urllib.request

from hibt_api_client import HibtApiCredentials, check_auth, normalize_timeframe, place_order


HIBT_DIR = Path(__file__).resolve().parent
DEFAULT_SIGNAL_FILE = HIBT_DIR / "signals.json"
DEFAULT_STATE_FILE = HIBT_DIR / "runtime" / "hibt_api_state.json"
DEFAULT_LOG_FILE = HIBT_DIR / "runtime" / "hibt_api_orders.jsonl"


@dataclass(frozen=True)
class Signal:
    signal_id: str
    symbol: str
    timeframe: str
    side: str
    timestamp: str


class State:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.processed = self._load()

    def seen(self, signal_id: str) -> bool:
        return signal_id in self.processed

    def mark(self, signal_id: str) -> None:
        processed = set(self.processed)
        processed.add(signal_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"processed_signal_ids": sorted(processed)}
        with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(self.path)
        self.processed = processed

    def _load(self) -> set[str]:
        if not self.path.exists():
            return set()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return set(payload.get("processed_signal_ids", []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute HiBT signals through the web option-order API.")
    parser.add_argument("--signal-file", type=Path, default=DEFAULT_SIGNAL_FILE)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--amount", action="append", default=[], help="Amount as timeframe=value, e.g. 15m=3.")
    parser.add_argument("--default-amount", default="3")
    parser.add_argument("--timeframes", default="15m", help="Comma-separated allowed timeframes.")
    parser.add_argument("--direction-up", type=int, default=1)
    parser.add_argument("--direction-down", type=int, default=2)
    parser.add_argument("--auth-check-seconds", type=float, default=300.0)
    parser.add_argument("--alert-min-seconds", type=float, default=3600.0)
    parser.add_argument("--live", action="store_true", help="Actually send API orders. Default is dry-run.")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed_timeframes = {normalize_timeframe(item) for item in args.timeframes.split(",") if item.strip()}
    amounts = parse_amounts(args.amount, args.default_amount)
    state = State(args.state_file)
    dry_run_seen: set[str] = set()
    credentials = None if not args.live else HibtApiCredentials.from_env()
    alerts = AlertManager.from_env(args.alert_min_seconds)
    last_auth_check = 0.0
    print(
        f"HiBT API trader start live={args.live} signal_file={args.signal_file} "
        f"timeframes={','.join(sorted(allowed_timeframes))} alerts={alerts.mode}",
        flush=True,
    )
    while True:
        try:
            if args.live and credentials is not None and args.auth_check_seconds > 0:
                now = time.monotonic()
                if now - last_auth_check >= args.auth_check_seconds:
                    last_auth_check = now
                    response = check_auth(credentials=credentials)
                    if response_auth_failed(response):
                        alerts.send(
                            "hibt-auth-expired",
                            "HiBT token may be expired",
                            alert_body("periodic auth check failed", response),
                        )
            for signal in read_signals(args.signal_file):
                if signal.timeframe not in allowed_timeframes:
                    continue
                if state.seen(signal.signal_id) or signal.signal_id in dry_run_seen:
                    continue
                amount = amounts.get(signal.timeframe, amounts["default"])
                direction = args.direction_up if signal.side == "BUY" else args.direction_down
                order = {
                    "amount": amount,
                    "direction": direction,
                    "symbol": signal.symbol,
                    "timeframe": signal.timeframe,
                    "signal_id": signal.signal_id,
                    "signal": signal.side,
                }
                if args.live:
                    response = place_order(
                        credentials=credentials,
                        amount=amount,
                        direction=direction,
                        symbol=signal.symbol,
                        timeframe=signal.timeframe,
                    )
                    success = order_succeeded(response)
                    record_order(args.log_file, signal, order, response, dry_run=False)
                    state.mark(signal.signal_id)
                    if success:
                        print(f"submitted {signal.signal_id} {signal.symbol} {signal.timeframe} {signal.side}", flush=True)
                    else:
                        alerts.send(
                            f"hibt-order-failed-{signal.signal_id}",
                            "HiBT order failed",
                            order_alert_body(signal, order, response),
                        )
                        if response_auth_failed(response):
                            alerts.send(
                                "hibt-auth-expired",
                                "HiBT token may be expired",
                                alert_body(f"order auth failed for {signal.signal_id}", response),
                            )
                        print(
                            f"failed {signal.signal_id} {signal.symbol} {signal.timeframe} {signal.side} "
                            f"status={response.get('http_status')}",
                            flush=True,
                        )
                else:
                    dry_run_seen.add(signal.signal_id)
                    record_order(args.log_file, signal, order, {"dry_run": True}, dry_run=True)
                    print(f"dry_run {signal.signal_id} {signal.symbol} {signal.timeframe} {signal.side}", flush=True)
            if args.once:
                return 0
            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            print("stopped by user", flush=True)
            return 130


def parse_amounts(items: list[str], default: str) -> dict[str, str]:
    result = {"default": str(default)}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--amount must be timeframe=value, got {item!r}")
        timeframe, amount = item.split("=", 1)
        result[normalize_timeframe(timeframe)] = amount
    return result


def read_signals(path: Path) -> list[Signal]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("signals"), list):
        raise ValueError(f"{path} must be a JSON object with a signals list")
    signals = [parse_signal(record) for record in payload["signals"]]
    return sorted(signals, key=lambda item: item.timestamp)


def parse_signal(record: Any) -> Signal:
    if not isinstance(record, dict):
        raise ValueError(f"signal record must be a JSON object: {record!r}")
    required = ("signal_id", "symbol", "timeframe", "signal", "timestamp")
    missing = [key for key in required if key not in record or str(record[key]).strip() == ""]
    if missing:
        raise ValueError(f"signal record missing fields: {', '.join(missing)}")
    return Signal(
        signal_id=str(record["signal_id"]),
        symbol=str(record["symbol"]).strip().upper(),
        timeframe=normalize_timeframe(str(record["timeframe"])),
        side=parse_signal_side(record["signal"]),
        timestamp=str(record["timestamp"]),
    )


def parse_signal_side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text not in {"BUY", "SELL"}:
        raise ValueError(f"unsupported signal side: {value!r}")
    return text


def order_succeeded(response: dict[str, Any]) -> bool:
    status = response.get("http_status")
    if not isinstance(status, int) or status < 200 or status >= 300:
        return False
    body = response.get("body")
    if isinstance(body, dict):
        if body.get("success") is False:
            return False
        code = body.get("code")
        if code is not None and str(code) not in {"0", "200"}:
            return False
    return True


def response_auth_failed(response: dict[str, Any]) -> bool:
    status = response.get("http_status")
    if status in {401, 403}:
        return True
    if isinstance(status, int) and 200 <= status < 300:
        body = response.get("body")
        if isinstance(body, dict):
            if body.get("success") is True:
                return False
            code = body.get("code")
            if code is not None and str(code) in {"0", "200"}:
                return False
            text = json.dumps(
                {
                    "code": body.get("code"),
                    "message": body.get("message"),
                    "msg": body.get("msg"),
                    "error": body.get("error"),
                },
                ensure_ascii=False,
            ).lower()
            auth_markers = ("unauthorized", "forbidden", "token", "login", "登录", "未登录", "过期")
            return any(marker in text for marker in auth_markers)
        return False
    body = response.get("body")
    text = json.dumps(body, ensure_ascii=False).lower() if isinstance(body, (dict, list)) else str(body).lower()
    auth_markers = ("unauthorized", "forbidden", "token", "login", "登录", "未登录", "过期")
    return any(marker in text for marker in auth_markers)


def alert_body(reason: str, response: dict[str, Any]) -> str:
    body = response.get("body")
    if isinstance(body, (dict, list)):
        body_text = json.dumps(body, ensure_ascii=False)
    else:
        body_text = str(body)
    body_text = re.sub(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "[jwt-redacted]", body_text)
    return (
        f"reason: {reason}\n"
        f"http_status: {response.get('http_status')}\n"
        f"body: {body_text[:1200]}\n"
        f"logged_at: {datetime.now(timezone.utc).isoformat()}\n"
    )


def order_alert_body(signal: Signal, order: dict[str, Any], response: dict[str, Any]) -> str:
    body = response.get("body")
    if isinstance(body, (dict, list)):
        body_text = json.dumps(body, ensure_ascii=False)
    else:
        body_text = str(body)
    body_text = re.sub(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "[jwt-redacted]", body_text)
    return (
        "HiBT order submission failed. This signal has been marked processed, so it will not be retried.\n"
        f"signal_id: {signal.signal_id}\n"
        f"symbol: {signal.symbol}\n"
        f"timeframe: {signal.timeframe}\n"
        f"side: {signal.side}\n"
        f"order: {json.dumps(order, ensure_ascii=False)}\n"
        f"http_status: {response.get('http_status')}\n"
        f"body: {body_text[:1200]}\n"
        f"logged_at: {datetime.now(timezone.utc).isoformat()}\n"
    )


def record_order(path: Path, signal: Signal, order: dict[str, Any], response: Any, *, dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "signal": asdict(signal),
        "order": order,
        "response": response,
    }
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


class AlertManager:
    def __init__(
        self,
        *,
        mode: str,
        min_seconds: float,
        email_config: dict[str, str],
        webhook_url: str,
    ) -> None:
        self.mode = mode
        self.min_seconds = min_seconds
        self.email_config = email_config
        self.webhook_url = webhook_url
        self.last_sent: dict[str, float] = {}

    @classmethod
    def from_env(cls, min_seconds: float) -> "AlertManager":
        email_config = {
            "host": os.environ.get("HIBT_ALERT_SMTP_HOST", ""),
            "port": os.environ.get("HIBT_ALERT_SMTP_PORT", "587"),
            "user": os.environ.get("HIBT_ALERT_SMTP_USER", ""),
            "password": os.environ.get("HIBT_ALERT_SMTP_PASSWORD", ""),
            "from": os.environ.get("HIBT_ALERT_EMAIL_FROM", ""),
            "to": os.environ.get("HIBT_ALERT_EMAIL_TO", ""),
            "tls": os.environ.get("HIBT_ALERT_SMTP_TLS", "1"),
        }
        webhook_url = os.environ.get("HIBT_ALERT_WEBHOOK_URL", "")
        email_enabled = all(email_config[key] for key in ("host", "user", "password", "from", "to"))
        modes = []
        if email_enabled:
            modes.append("email")
        if webhook_url:
            modes.append("webhook")
        return cls(
            mode="+".join(modes) if modes else "disabled",
            min_seconds=min_seconds,
            email_config=email_config,
            webhook_url=webhook_url,
        )

    def send(self, key: str, subject: str, body: str) -> None:
        if self.mode == "disabled":
            print(f"alert skipped key={key} mode=disabled subject={subject}", flush=True)
            return
        now = time.monotonic()
        previous = self.last_sent.get(key)
        if previous is not None and now - previous < self.min_seconds:
            return
        self.last_sent[key] = now
        if "email" in self.mode:
            self._send_email(subject, body)
        if "webhook" in self.mode:
            self._send_webhook(subject, body)

    def _send_email(self, subject: str, body: str) -> None:
        message = EmailMessage()
        message["From"] = self.email_config["from"]
        message["To"] = self.email_config["to"]
        message["Subject"] = subject
        message.set_content(body)
        try:
            with smtplib.SMTP(self.email_config["host"], int(self.email_config["port"]), timeout=20) as smtp:
                if self.email_config["tls"] not in {"0", "false", "False"}:
                    smtp.starttls()
                smtp.login(self.email_config["user"], self.email_config["password"])
                smtp.send_message(message)
            print(f"alert email sent subject={subject}", flush=True)
        except Exception as exc:
            print(f"alert email failed subject={subject} error={exc}", flush=True)

    def _send_webhook(self, subject: str, body: str) -> None:
        payload = json.dumps({"subject": subject, "body": body}).encode("utf-8")
        request = urllib.request.Request(
            self.webhook_url,
            data=payload,
            method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                response.read()
            print(f"alert webhook sent subject={subject}", flush=True)
        except urllib.error.URLError as exc:
            print(f"alert webhook failed subject={subject} error={exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
