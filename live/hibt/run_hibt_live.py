#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LIVE_DIR = Path(__file__).resolve().parents[1]
if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))

from log_colors import colorize_line  # noqa: E402
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
                colorize_line(
                    "missing dependency: playwright. Install with: "
                    "pip install -r requirements_hibt.txt && python -m playwright install --with-deps chromium"
                ),
                flush=True,
            )
            return 4
        raise

    reader = SignalReader(config.signal_path)
    trader = HibtTrader(config)
    message = (
        f"HiBT runner start dry_run={config.dry_run} confirm_order={config.click_confirm_order} "
        f"signal_path={config.signal_path}"
    )
    print(colorize_line(message), flush=True)

    with HibtBrowser(config) as browser:
        if args.login:
            url = browser.open_for_login("BTC-USDT")
            print(
                colorize_line(f"login browser opened at {url}; press Ctrl+C here after login is complete"),
                flush=True,
            )
            try:
                while True:
                    time.sleep(5)
            except KeyboardInterrupt:
                print(colorize_line("login session closed"), flush=True)
                return 0
        while True:
            try:
                signals = reader.read() if args.once else reader.read_if_changed()
                for signal in signals:
                    start = time.perf_counter()
                    result = trader.handle_signal(browser, signal)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    print(
                        colorize_line(
                            f"DONE HiBT signal {signal.symbol} {signal.timeframe or 'na'} "
                            f"{signal.side} elapsed_ms={elapsed_ms:.1f}"
                        ),
                        flush=True,
                    )
                    if result and config.stop_after_first_trade:
                        return 0
                if args.once:
                    return 0
                time.sleep(config.poll_seconds)
            except KeyboardInterrupt:
                print(colorize_line("stopped by user"), flush=True)
                return 130
            except SafetyHalt as exc:
                print(colorize_line(f"safety halt: {exc}"), flush=True)
                return 2
            except Exception as exc:
                if not getattr(exc, "hibt_signal_recorded", False):
                    trader.journal.record_failure(None, exc)
                print(colorize_line(f"error: {type(exc).__name__}: {exc}"), flush=True)
                if trader.journal.state.consecutive_failures >= config.risk.max_consecutive_failures:
                    print(colorize_line("too many consecutive failures; exiting"), flush=True)
                    return 3
                if args.once:
                    return 1
                time.sleep(max(config.poll_seconds, 5.0))


if __name__ == "__main__":
    raise SystemExit(main())
