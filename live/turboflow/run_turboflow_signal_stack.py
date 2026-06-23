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


ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "live"
TURBOFLOW_DIR = LIVE_DIR / "turboflow"
DEFAULT_SIGNAL_MODEL_DIRS = ["3m=live/models_3m", "5m=live/models_5m", "15m=live/models_15m"]


def clear_signal_file(path: Path) -> None:
    payload = {"generated_at": None, "source": "turboflow_signal_stack", "signals": [], "diagnostics": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"cleared stale signal file: {path}", flush=True)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local LightGBM signal generation for TurboFlow execution.")
    parser.add_argument("--data-root", type=Path, default=Path("aligned_data_oos"))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--signal-file", type=Path, default=TURBOFLOW_DIR / "signals.json")
    parser.add_argument("--signal-model-dir", action="append", default=None)
    parser.add_argument("--kline-log-level", default="INFO")
    parser.add_argument("--kline-keep-rows", type=int, default=3000)
    parser.add_argument("--rest-backfill-minutes", type=int, default=360)
    parser.add_argument("--rest-catchup-minutes", type=int, default=2)
    parser.add_argument("--rest-catchup-seconds", type=float, default=0.05)
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
    for item in args.signal_model_dir or DEFAULT_SIGNAL_MODEL_DIRS:
        command.extend(["--signal-model-dir", item])

    process: subprocess.Popen | None = None
    try:
        print(f"start update_live_klines.py: {' '.join(command)}", flush=True)
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        while True:
            code = process.poll()
            if code is not None:
                return int(code)
            time.sleep(1)
    except KeyboardInterrupt:
        print("stopping TurboFlow signal stack", flush=True)
        if process is not None:
            stop_process(process)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
