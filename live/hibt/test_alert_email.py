#!/usr/bin/env python3
from __future__ import annotations

from run_hibt_api_trader import AlertManager


def main() -> int:
    AlertManager.from_env(0).send(
        "hibt-alert-test",
        "HiBT alert test",
        "HiBT order-failure/token-expiry email alert is configured.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
