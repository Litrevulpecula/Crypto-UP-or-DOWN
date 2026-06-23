#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


HIBT_DIR = Path(__file__).resolve().parents[1] / "hibt"
sys.path.insert(0, str(HIBT_DIR))

import control_panel as base  # noqa: E402


ROOT = Path(__file__).resolve().parent
base.ROOT = ROOT
base.DEFAULT_CONTROL_FILE = ROOT / "runtime" / "turboflow_control.json"
base.DEFAULT_LOG_FILE = ROOT / "runtime" / "turboflow_api_orders.jsonl"
base.INDEX_HTML = (
    base.INDEX_HTML
    .replace("HiBT Execution", "TurboFlow Execution")
    .replace(">HiBT <", ">TurboFlow <")
    .replace("hibt_control.json", "turboflow_control.json")
    .replace("HiBT trader", "TurboFlow trader")
)


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
