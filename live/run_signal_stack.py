#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "live"
DEFAULT_SIGNAL_FILE = LIVE_DIR / "signals.json"
DEFAULT_SIGNAL_MODEL_DIRS = ["15m=live/models_15m"]


def command_label(command: list[str]) -> str:
    return Path(command[1]).name if len(command) > 1 else Path(command[0]).name


def start_process(command: list[str]) -> subprocess.Popen:
    print(f"start {command_label(command)}: {' '.join(command)}", flush=True)
    return subprocess.Popen(command, cwd=ROOT, start_new_session=True)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def raise_keyboard_interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def clear_signal_file(path: Path) -> None:
    payload = {
        "generated_at": None,
        "source": "lightgbm_signal_stack",
        "signals": [],
        "diagnostics": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"cleared stale signal file: {path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LightGBM signal generation for event-contract routers.")
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data_oos"))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--signal-file", type=Path, default=DEFAULT_SIGNAL_FILE)
    parser.add_argument("--signal-model-dir", action="append", default=None)
    parser.add_argument("--kline-log-level", default="INFO")
    parser.add_argument("--kline-keep-rows", type=int, default=3000)
    parser.add_argument("--rest-backfill-minutes", type=int, default=360)
    parser.add_argument("--rest-catchup-minutes", type=int, default=15)
    parser.add_argument("--rest-catchup-seconds", type=float, default=2.0)
    parser.add_argument("--keep-existing-signal-file", action="store_true")
    args = parser.parse_args()

    if not args.keep_existing_signal_file:
        clear_signal_file(args.signal_file)

    command = [
        sys.executable,
        str(LIVE_DIR / "update_live_klines.py"),
        "--data-root",
        str(args.data_root),
        "--symbols",
        args.symbols,
        "--keep-rows",
        str(args.kline_keep_rows),
        "--log-level",
        args.kline_log_level,
        "--rest-backfill-minutes",
        str(args.rest_backfill_minutes),
        "--rest-catchup-minutes",
        str(args.rest_catchup_minutes),
        "--rest-catchup-seconds",
        str(args.rest_catchup_seconds),
        "--signal-file",
        str(args.signal_file),
    ]
    signal_model_dirs = args.signal_model_dir if args.signal_model_dir is not None else DEFAULT_SIGNAL_MODEL_DIRS
    for model_dir in signal_model_dirs:
        command.extend(["--signal-model-dir", model_dir])

    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    process: subprocess.Popen | None = None
    try:
        process = start_process(command)
        while True:
            return_code = process.poll()
            if return_code is not None:
                print(f"{command_label(process.args)} exited with code {return_code}", flush=True)
                return int(return_code)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopping signal stack", flush=True)
        if process is not None:
            stop_process(process)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
