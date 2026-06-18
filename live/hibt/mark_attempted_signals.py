#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


HIBT_DIR = Path(__file__).resolve().parent
STATE_FILE = HIBT_DIR / "runtime" / "hibt_api_state.json"
ORDER_LOG = HIBT_DIR / "runtime" / "hibt_api_orders.jsonl"


def main() -> int:
    processed = set()
    if STATE_FILE.exists():
        processed.update(json.loads(STATE_FILE.read_text(encoding="utf-8")).get("processed_signal_ids", []))
    if ORDER_LOG.exists():
        for line in ORDER_LOG.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                signal_id = json.loads(line).get("signal", {}).get("signal_id")
            except json.JSONDecodeError:
                continue
            if signal_id:
                processed.add(signal_id)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"processed_signal_ids": sorted(processed)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"processed_signal_ids={len(processed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
