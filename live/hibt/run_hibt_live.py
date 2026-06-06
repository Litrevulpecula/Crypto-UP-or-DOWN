#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hibt_config import HibtConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HiBT BTC/ETH web runner driven by a local signal file.")
    parser.add_argument("--config", type=Path, default=None, help="JSON config path. Defaults are safe dry-run values.")
    parser.add_argument("--signal-file", type=Path, default=None, help="Override config.signal_path.")
    parser.add_argument("--once", action="store_true", help="Read current signal file once and exit.")
    parser.add_argument("--login", action="store_true", help="Open HiBT and keep the browser open for manual login.")
    parser.add_argument("--live", action="store_true", help="Enable live button clicking. Default is dry-run.")
    parser.add_argument("--confirm-order", action="store_true", help="Click the final confirmation modal after trade button.")
    parser.add_argument(
        "--allow-direct-submit",
        action="store_true",
        help="Dangerous: allow live mode when HiBT second confirmation is disabled.",
    )
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = HibtConfig.load(args.config)
    if args.signal_file is not None:
        config.signal_path = args.signal_file
    if args.live:
        config.dry_run = False
    if args.confirm_order:
        config.click_confirm_order = True
    if args.allow_direct_submit:
        config.risk.allow_direct_submit_without_confirmation = True
        config.risk.require_second_confirmation_enabled = False
    if args.print_config:
        print(json.dumps(config.to_jsonable(), ensure_ascii=False, indent=2))
        return 0

    try:
        from hibt_browser import HibtBrowser, SafetyHalt
        from hibt_signal_reader import SignalReader
        from hibt_trader import HibtTrader
    except ModuleNotFoundError as exc:
        if exc.name == "playwright":
            print(
                "missing dependency: playwright. Install with: "
                "pip install -r requirements_hibt.txt && python -m playwright install --with-deps chromium",
                flush=True,
            )
            return 4
        raise

    reader = SignalReader(config.signal_path)
    trader = HibtTrader(config)
    print(
        f"HiBT runner start dry_run={config.dry_run} confirm_order={config.click_confirm_order} "
        f"signal_path={config.signal_path}",
        flush=True,
    )

    with HibtBrowser(config) as browser:
        if args.login:
            url = browser.open_for_login("BTC-USDT")
            print(f"login browser opened at {url}; press Ctrl+C here after login is complete", flush=True)
            try:
                while True:
                    time.sleep(5)
            except KeyboardInterrupt:
                print("login session closed", flush=True)
                return 0
        while True:
            try:
                signals = reader.read()
                for signal in signals:
                    result = trader.handle_signal(browser, signal)
                    if result and config.stop_after_first_trade:
                        return 0
                if args.once:
                    return 0
                time.sleep(config.poll_seconds)
            except KeyboardInterrupt:
                print("stopped by user", flush=True)
                return 130
            except SafetyHalt as exc:
                print(f"safety halt: {exc}", flush=True)
                return 2
            except Exception as exc:
                if not getattr(exc, "hibt_signal_recorded", False):
                    trader.journal.record_failure(None, exc)
                print(f"error: {type(exc).__name__}: {exc}", flush=True)
                if trader.journal.state.consecutive_failures >= config.risk.max_consecutive_failures:
                    print("too many consecutive failures; exiting", flush=True)
                    return 3
                if args.once:
                    return 1
                time.sleep(max(config.poll_seconds, 5.0))


if __name__ == "__main__":
    raise SystemExit(main())
