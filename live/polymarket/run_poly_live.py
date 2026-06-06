#!/usr/bin/env python3
"""Entry point for Polymarket live trading."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poly_trader import PolyConfig, PolyTrader


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket CLOB trader")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path(__file__).resolve().parent / "poly_config.json",
        help="Path to config JSON",
    )
    parser.add_argument("--dry-run", action="store_true", help="Override dry_run=True")
    args = parser.parse_args()

    config = PolyConfig.load(args.config)
    if args.dry_run:
        config.dry_run = True

    if not config.private_key:
        print("ERROR: private_key is required in config", file=sys.stderr)
        sys.exit(1)

    trader = PolyTrader(config)
    trader.connect()
    trader.run_loop()


if __name__ == "__main__":
    main()
