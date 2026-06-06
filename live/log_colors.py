from __future__ import annotations

import logging
import os
import re


RESET = "\033[0m"
BOLD = "\033[1m"

SYMBOL_HEX = {
    "BTC": "#FFB000",
    "BTCUSDT": "#FFB000",
    "BTC-USDT": "#FFB000",
    "BTC/USDT": "#FFB000",
    "ETH": "#9AA8FF",
    "ETHUSDT": "#9AA8FF",
    "ETH-USDT": "#9AA8FF",
    "ETH/USDT": "#9AA8FF",
}

KEYWORD_HEX = {
    "DEBUG": "#C4C9D4",
    "INFO": "#7DD3FC",
    "WARNING": "#FDE047",
    "ERROR": "#FF5C7A",
    "CRITICAL": "#FF5C7A",
    "DRY RUN": "#4ADE80",
    "dry_run": "#4ADE80",
    "BUY": "#4ADE80",
    "SELL": "#FF5C7A",
    "UP": "#4ADE80",
    "DOWN": "#FF5C7A",
    "HOLD": "#C4C9D4",
    "REST": "#22D3EE",
    "websocket": "#D8B4FE",
    "connect": "#22D3EE",
    "closed": "#4ADE80",
    "backfilled": "#22D3EE",
    "catch-up": "#22D3EE",
    "signal": "#FDE047",
    "callback": "#FDE047",
    "emitted": "#4ADE80",
    "ready": "#4ADE80",
    "Connected": "#4ADE80",
    "Resolved": "#4ADE80",
    "posted": "#4ADE80",
    "Executed": "#4ADE80",
    "submitted": "#4ADE80",
    "needs_manual_confirm": "#FDE047",
    "Posting": "#22D3EE",
    "New signal": "#FDE047",
    "Limit order": "#4ADE80",
    "Ask jumped": "#FDE047",
    "market buy": "#22D3EE",
    "ENTRY": "#FDE047",
    "WATCH": "#FDE047",
    "TAKE": "#4ADE80",
    "POST": "#22D3EE",
    "QUOTE": "#22D3EE",
    "SKIP": "#FDE047",
    "TIMEOUT": "#FDE047",
    "DONE": "#4ADE80",
    "elapsed_ms": "#22D3EE",
    "cancel": "#FDE047",
    "filled": "#4ADE80",
    "partial": "#FDE047",
    "limit": "#22D3EE",
    "market": "#22D3EE",
    "skip": "#FDE047",
    "skipping": "#FDE047",
    "failed": "#FF5C7A",
    "exception": "#FF5C7A",
    "traceback": "#FF5C7A",
    "Cannot": "#FF5C7A",
    "missing": "#FF5C7A",
    "No": "#FF5C7A",
    "disconnected": "#FF5C7A",
    "error": "#FF5C7A",
    "safety halt": "#FF5C7A",
    "stopping": "#FDE047",
    "Shutting down": "#FDE047",
    "stopped": "#FDE047",
    "start": "#22D3EE",
    "Starting": "#22D3EE",
    "run": "#22D3EE",
    "exited": "#FDE047",
}

_KEYWORD_HEX_CASEFOLD = {key.casefold(): value for key, value in KEYWORD_HEX.items()}
_BOLD_KEYWORDS = set(_KEYWORD_HEX_CASEFOLD)


def colors_disabled() -> bool:
    return os.environ.get("LIVE_LOG_NO_COLOR") == "1"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Invalid hex color: {value!r}")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def color_text(text: object, hex_color: str, bold: bool = False) -> str:
    if colors_disabled():
        return str(text)
    red, green, blue = _hex_to_rgb(hex_color)
    codes = ["38;2;%d;%d;%d" % (red, green, blue)]
    if bold:
        codes.insert(0, "1")
    prefix = f"\033[{';'.join(codes)}m"
    return f"{prefix}{text}{RESET}"


def symbol_text(symbol: object) -> str:
    text = str(symbol)
    color = SYMBOL_HEX.get(text.upper().replace("/", "-"))
    if color is None:
        color = SYMBOL_HEX.get(text.upper().replace("-", "/"))
    return color_text(text, color, bold=True) if color else text


def keyword_text(keyword: object) -> str:
    text = str(keyword)
    color = KEYWORD_HEX.get(text)
    if color is None:
        color = _KEYWORD_HEX_CASEFOLD.get(text.casefold())
    return color_text(text, color, bold=text.casefold() in _BOLD_KEYWORDS) if color else text


def _token_pattern(tokens: list[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(token) for token in sorted(tokens, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9_/-])(?:{alternatives})(?![A-Za-z0-9_/-])", re.IGNORECASE)


TOKEN_PATTERN = _token_pattern(list(SYMBOL_HEX) + list(KEYWORD_HEX))


def colorize_line(line: object) -> str:
    text = str(line)
    if colors_disabled():
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        color = SYMBOL_HEX.get(token.upper())
        if color is None:
            color = SYMBOL_HEX.get(token.upper().replace("/", "-"))
        if color is None:
            color = SYMBOL_HEX.get(token.upper().replace("-", "/"))
        if color:
            return color_text(token, color, bold=True)
        return keyword_text(token)

    return TOKEN_PATTERN.sub(replace, text)


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return colorize_line(super().format(record))


def configure_colored_logging(level: int, fmt: str = "%(asctime)s %(levelname)s %(message)s") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(fmt))
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)
    for logger_name in ("httpx", "httpcore", "urllib3", "websockets"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
