#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


LIVE_DIR = Path(__file__).resolve().parent


@dataclass
class FingerprintConfig:
    enabled: bool = True
    user_agent: str | None = None
    accept_language: str = "zh-CN,zh;q=0.9,en;q=0.8"
    languages: list[str] = field(default_factory=lambda: ["zh-CN", "zh", "en"])
    platform: str = "Win32"
    hardware_concurrency: int = 8
    device_memory: int = 8
    webgl_vendor: str = "Google Inc. (NVIDIA)"
    webgl_renderer: str = (
        "ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0, D3D11)"
    )
    color_scheme: str = "light"
    device_scale_factor: float = 1.0
    screen_width: int = 1920
    screen_height: int = 1080
    screen_color_depth: int = 24
    canvas_noise: bool = True
    audio_noise: bool = True
    block_webrtc_leak: bool = True
    human_typing: bool = True
    min_type_delay_ms: int = 55
    max_type_delay_ms: int = 165
    viewport_jitter_px: int = 3


@dataclass
class BrowserConfig:
    user_data_dir: Path = LIVE_DIR / "runtime" / "hibt-chrome-profile"
    headless: bool = False
    slow_mo_ms: int = 80
    locale: str = "zh-CN"
    timezone_id: str = "Asia/Shanghai"
    executable_path: str | None = None
    channel: str | None = None
    proxy_server: str | None = None
    cdp_mode: bool = True
    cdp_port: int = 9222
    viewport_width: int = 1700
    viewport_height: int = 950
    navigation_timeout_ms: int = 30_000
    action_timeout_ms: int = 10_000
    min_action_delay_seconds: float = 0.25
    max_action_delay_seconds: float = 1.10
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)


@dataclass
class RiskConfig:
    min_confidence: float = 0.0
    min_payout_rate_percent: float = 80.0
    max_signal_age_seconds: int = 240
    one_order_per_symbol_per_candle: bool = True
    candle_minutes: int = 5
    cooldown_seconds: int = 120
    max_orders_per_day: int = 30
    max_orders_per_symbol_per_day: int = 18
    max_consecutive_failures: int = 3
    block_when_open_position_visible: bool = True
    require_second_confirmation_enabled: bool = False
    allow_direct_submit_without_confirmation: bool = True
    require_amount_available: bool = True
    min_available_usdt: float = 3.0


@dataclass
class Selectors:
    order_panel: str = ".future-order"
    current_symbol: str = ".option-coin-row-left__label"
    active_symbol_card: str = ".symbol-card.active .txt-symbol"
    symbol_cards: str = ".symbol-card"
    time_units: str = ".future-order .unit"
    active_time_unit: str = ".future-order .unit.active"
    amount_input: str = ".future-order .market-input-unit-type input.el-input__inner"
    buy_up_button: str = ".future-order button.btn-trade-success"
    buy_down_button: str = ".future-order button.btn-trade-danger"
    position_container: str = ".position-container"
    visible_overlay: str = ".el-overlay:visible, .el-message-box:visible, .el-dialog:visible"
    visible_overlay_buttons: str = ".el-overlay:visible button, .el-message-box:visible button, .el-dialog:visible button"


@dataclass
class HibtConfig:
    base_url: str = "https://hibt.com/zh-cn/options"
    symbols: tuple[str, ...] = ("BTC-USDT", "ETH-USDT")
    amount_usdt: str = "3"
    duration_label: str = "5分钟"
    duration_labels: dict[str, str] = field(default_factory=lambda: {"3m": "3分钟", "5m": "5分钟"})
    signal_path: Path = LIVE_DIR / "signals.json"
    state_path: Path = LIVE_DIR / "runtime" / "hibt_state.json"
    log_path: Path = LIVE_DIR / "runtime" / "hibt_orders.jsonl"
    poll_seconds: float = 2.0
    dry_run: bool = True
    click_confirm_order: bool = False
    stop_after_first_trade: bool = False
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    selectors: Selectors = field(default_factory=Selectors)

    @classmethod
    def load(cls, path: Path | None) -> "HibtConfig":
        config = cls()
        if path is None:
            return config
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        _merge_dataclass(config, payload)
        _resolve_paths(config)
        return config

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        return _stringify_paths(payload)


def _merge_dataclass(target: Any, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if not hasattr(target, key):
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(target, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif isinstance(current, Path):
            setattr(target, key, _path(value))
        elif isinstance(current, tuple):
            setattr(target, key, tuple(normalize_symbol(item) for item in value))
        else:
            setattr(target, key, value)


def _resolve_paths(config: HibtConfig) -> None:
    config.signal_path = _path(config.signal_path)
    config.state_path = _path(config.state_path)
    config.log_path = _path(config.log_path)
    config.browser.user_data_dir = _path(config.browser.user_data_dir)


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return LIVE_DIR / path


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_paths(item) for item in value]
    return value


def normalize_symbol(value: str) -> str:
    compact = value.strip().upper().replace("_", "").replace("-", "").replace("/", "")
    if compact in {"BTCUSDT", "BTC"}:
        return "BTC-USDT"
    if compact in {"ETHUSDT", "ETH"}:
        return "ETH-USDT"
    if compact.endswith("USDT") and len(compact) > 4:
        return f"{compact[:-4]}-USDT"
    return value.strip().upper()


def hibt_symbol_path(symbol: str) -> str:
    return normalize_symbol(symbol)
