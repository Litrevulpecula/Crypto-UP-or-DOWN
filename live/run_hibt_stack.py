#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from log_colors import colorize_line


ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = Path(__file__).resolve().parent
HIBT_DIR = LIVE_DIR / "hibt"
DEFAULT_SIGNAL_MODEL_DIRS = ["15m=live/models_15m"]


def command_name(command: list[str]) -> str:
    script = Path(command[1]).name if len(command) > 1 else Path(command[0]).name
    return script.replace(".py", "")


def start_process(command: list[str]) -> subprocess.Popen:
    print(colorize_line(f"start {command_name(command)}: {' '.join(command)}"), flush=True)
    return subprocess.Popen(command, cwd=ROOT, start_new_session=True)


def run_startup_command(command: list[str]) -> int:
    print(colorize_line(f"run {command_name(command)}: {' '.join(command)}"), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def kill_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def stop_all(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        stop_process(process)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes):
            return
        time.sleep(0.2)
    for process in processes:
        kill_process(process)


def raise_keyboard_interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Binance live klines, 15m LightGBM signals, and HiBT trader.")
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data_oos"))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--signal-file", type=Path, default=HIBT_DIR / "signals.json")
    parser.add_argument("--hibt-config", type=Path, default=HIBT_DIR / "hibt_config.vps.json")
    parser.add_argument("--kline-log-level", default="INFO")
    parser.add_argument("--kline-keep-rows", type=int, default=3000)
    parser.add_argument("--rest-backfill-minutes", type=int, default=360)
    parser.add_argument("--rest-catchup-minutes", type=int, default=15)
    parser.add_argument("--rest-catchup-seconds", type=float, default=2.0)
    parser.add_argument(
        "--signal-model-dir",
        action="append",
        default=None,
        help=(
            "Signal model directory as timeframe=path. Repeatable. "
            "Default HiBT stack is 15m=live/models_15m."
        ),
    )
    parser.add_argument("--live", action="store_true", help="Force HiBT live mode regardless of config dry_run.")
    parser.add_argument("--confirm-order", action="store_true", help="Force HiBT final confirmation click.")
    parser.add_argument("--no-trader", action="store_true", help="Run data and signals only.")
    args = parser.parse_args()

    python = sys.executable
    if args.rest_backfill_minutes > 0:
        return_code = run_startup_command(
            [
                python,
                str(LIVE_DIR / "update_live_klines.py"),
                "--data-root",
                str(args.data_root),
                "--symbols",
                args.symbols,
                "--keep-rows",
                str(args.kline_keep_rows),
                "--rest-backfill-minutes",
                str(args.rest_backfill_minutes),
                "--rest-backfill-only",
                "--log-level",
                args.kline_log_level,
            ]
        )
        if return_code != 0:
            return return_code

    commands = [
        [
            python,
            str(LIVE_DIR / "update_live_klines.py"),
            "--data-root",
            str(args.data_root),
            "--symbols",
            args.symbols,
            "--keep-rows",
            str(args.kline_keep_rows),
            "--rest-backfill-minutes",
            "0",
            "--rest-catchup-minutes",
            str(args.rest_catchup_minutes),
            "--rest-catchup-seconds",
            str(args.rest_catchup_seconds),
            "--signal-file",
            str(args.signal_file),
            "--log-level",
            args.kline_log_level,
        ],
    ]
    signal_model_dirs = args.signal_model_dir if args.signal_model_dir is not None else DEFAULT_SIGNAL_MODEL_DIRS
    for model_dir in signal_model_dirs:
        commands[0].extend(["--signal-model-dir", model_dir])
    if not args.no_trader:
        trader_command = [
            python,
            str(HIBT_DIR / "run_hibt_live.py"),
            "--config",
            str(args.hibt_config),
            "--signal-file",
            str(args.signal_file),
        ]
        if args.live:
            trader_command.append("--live")
        if args.confirm_order:
            trader_command.append("--confirm-order")
        commands.append(trader_command)

    processes: list[subprocess.Popen] = []
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    try:
        for command in commands:
            processes.append(start_process(command))
            time.sleep(1.0)
        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    print(colorize_line(f"{command_name(process.args)} exited with code {return_code}"), flush=True)
                    stop_all(processes)
                    return int(return_code)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print(colorize_line("stopping HiBT live stack"), flush=True)
        stop_all(processes)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
