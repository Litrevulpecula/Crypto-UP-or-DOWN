#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from playwright.sync_api import Locator, Page, Playwright, sync_playwright

from hibt_config import HibtConfig, normalize_symbol
from hibt_fingerprint import context_options, init_script, launch_args


class SafetyHalt(RuntimeError):
    pass


@dataclass
class PageState:
    url: str
    title: str
    current_symbol: str | None
    active_symbol_card: str | None
    active_time_unit: str | None
    amount_value: str
    available_usdt: float | None
    buy_up_payout_percent: float | None
    buy_down_payout_percent: float | None
    second_confirmation_enabled: bool | None
    logged_in: bool
    open_position_visible: bool
    page_text_sample: str


@dataclass
class OrderResult:
    status: str
    message: str
    state_before: PageState
    state_after: PageState | None = None
    modal_text: str | None = None


class HibtBrowser:
    def __init__(self, config: HibtConfig) -> None:
        self.config = config
        self.playwright: Playwright | None = None
        self.context: Any | None = None
        self.browser: Any | None = None
        self.page: Page | None = None
        self._chrome_proc: subprocess.Popen | None = None
        self._last_mouse_x: float = random.uniform(300, 800)
        self._last_mouse_y: float = random.uniform(200, 500)

    def __enter__(self) -> "HibtBrowser":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self) -> None:
        cfg = self.config.browser
        fp = cfg.fingerprint
        cfg.user_data_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_fingerprint_seed(cfg)
        self.playwright = sync_playwright().start()

        if cfg.cdp_mode:
            self._start_cdp(cfg, fp)
        else:
            self._start_launch(cfg, fp)

    def _start_cdp(self, cfg: Any, fp: Any) -> None:
        """Launch Chrome as a standalone process, then connect via CDP."""
        chrome_bin = self._find_chrome(cfg)
        port = cfg.cdp_port
        args = [
            chrome_bin,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={cfg.user_data_dir}",
            f"--window-size={cfg.viewport_width},{cfg.viewport_height}",
            f"--lang={cfg.locale}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-infobars",
        ]
        if fp.enabled:
            args.append("--disable-blink-features=AutomationControlled")
        if cfg.headless:
            args.append("--headless=new")
        if cfg.proxy_server:
            args.append(f"--proxy-server={cfg.proxy_server}")

        self._chrome_proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        cdp_url = f"http://127.0.0.1:{port}"
        self._wait_for_cdp(cdp_url)
        self.browser = self.playwright.chromium.connect_over_cdp(cdp_url, slow_mo=cfg.slow_mo_ms)
        self.context = self.browser.contexts[0] if self.browser.contexts else self.browser.new_context()
        if fp.enabled:
            self.context.add_init_script(init_script(fp))
        self.context.set_default_timeout(cfg.action_timeout_ms)
        self.context.set_default_navigation_timeout(cfg.navigation_timeout_ms)
        self.context.set_extra_http_headers({"Accept-Language": fp.accept_language})
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()

    def _start_launch(self, cfg: Any, fp: Any) -> None:
        """Fallback: launch via Playwright (original mode)."""
        args = [f"--lang={cfg.locale}", f"--window-size={cfg.viewport_width},{cfg.viewport_height}"]
        if fp.enabled:
            args.extend(arg for arg in launch_args(fp) if arg not in args)
        ctx_opts = context_options(fp) if fp.enabled else {}
        jitter = ctx_opts.pop("_viewport_jitter", 0) if fp.enabled else 0
        vp_w = cfg.viewport_width + jitter
        vp_h = cfg.viewport_height + random.randint(-fp.viewport_jitter_px, fp.viewport_jitter_px)
        launch_kwargs: dict[str, Any] = {
            "user_data_dir": str(cfg.user_data_dir),
            "headless": cfg.headless,
            "slow_mo": cfg.slow_mo_ms,
            "locale": cfg.locale,
            "timezone_id": cfg.timezone_id,
            "viewport": {"width": vp_w, "height": vp_h},
            "args": args,
        }
        if fp.enabled:
            ctx_opts.pop("viewport", None)
            launch_kwargs.update(ctx_opts)
        if cfg.executable_path:
            from pathlib import Path as _P
            if _P(cfg.executable_path).exists():
                launch_kwargs["executable_path"] = cfg.executable_path
            else:
                launch_kwargs["channel"] = "chrome"
        elif cfg.channel:
            launch_kwargs["channel"] = cfg.channel
        if cfg.proxy_server:
            launch_kwargs["proxy"] = {"server": cfg.proxy_server}
        self.context = self.playwright.chromium.launch_persistent_context(**launch_kwargs)
        if fp.enabled:
            self.context.add_init_script(init_script(fp))
        self.context.set_default_timeout(cfg.action_timeout_ms)
        self.context.set_default_navigation_timeout(cfg.navigation_timeout_ms)
        self.context.set_extra_http_headers({"Accept-Language": fp.accept_language})
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()

    def _ensure_fingerprint_seed(self, cfg: Any) -> None:
        fp = cfg.fingerprint
        if not fp.enabled or fp.seed is not None:
            return
        seed_path = cfg.user_data_dir / "fingerprint_seed.txt"
        try:
            if seed_path.exists():
                seed = int(seed_path.read_text(encoding="utf-8").strip())
            else:
                seed = random.randint(1, 2**31)
                seed_path.write_text(f"{seed}\n", encoding="utf-8")
            fp.seed = seed
        except Exception:
            fp.seed = random.randint(1, 2**31)

    def _find_chrome(self, cfg: Any) -> str:
        """Locate Chrome binary."""
        if cfg.executable_path:
            return cfg.executable_path
        for candidate in [
            "/usr/bin/google-chrome-stable",
            "/usr/bin/google-chrome",
            "/opt/google/chrome/chrome",
            "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
        ]:
            if os.path.isfile(candidate):
                return candidate
        found = shutil.which("google-chrome") or shutil.which("chromium-browser")
        if found:
            return found
        raise RuntimeError(
            "Chrome not found. Install with: "
            "wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && "
            "apt install -y ./google-chrome-stable_current_amd64.deb"
        )

    def _wait_for_cdp(self, url: str, timeout: float = 15.0) -> None:
        """Poll until Chrome's CDP endpoint is responsive."""
        import urllib.request
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{url}/json/version", timeout=2)
                return
            except Exception:
                time.sleep(0.3)
        raise RuntimeError(f"Chrome CDP endpoint at {url} did not start within {timeout}s")

    def close(self) -> None:
        if self.browser is not None:
            try:
                self.browser.close()
            except Exception:
                pass
        elif self.context is not None:
            try:
                self.context.close()
            except Exception:
                pass
        if self.playwright is not None:
            self.playwright.stop()
        if self._chrome_proc is not None:
            self._chrome_proc.terminate()
            try:
                self._chrome_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._chrome_proc.kill()
        self.context = None
        self.browser = None
        self.playwright = None
        self.page = None
        self._chrome_proc = None

    def open_symbol(self, symbol: str) -> PageState:
        target = normalize_symbol(symbol)

        if not self._options_page_loaded():
            return self._load_symbol_url(target)

        state = self.read_state()
        if state.current_symbol == target and (state.active_symbol_card is None or state.active_symbol_card == target):
            return state

        self._click_symbol_card(target)
        self._wait_for_page_symbol(target)
        state = self.read_state()
        self._assert_page_symbol(state, target)
        return state

    def open_for_login(self, symbol: str = "BTC-USDT") -> str:
        page = self._page()
        target = normalize_symbol(symbol)
        page.goto(f"{self.config.base_url}/{target}", wait_until="domcontentloaded")
        self._wait_for_cf_challenge(page, timeout_s=60.0)
        self._settle(seconds=2.0)
        return page.url

    def prepare_order(self, symbol: str, duration_label: str | None = None) -> PageState:
        target = normalize_symbol(symbol)
        duration = duration_label or self.config.duration_label
        self.open_symbol(target)
        self.select_duration(duration)
        self.fill_amount(self.config.amount_usdt)
        state = self.read_state()
        self.validate_ready_state(state, target, duration)
        return state

    def select_duration(self, label: str) -> None:
        page = self._page()
        current = self.read_state().active_time_unit
        if current == label:
            return
        locator = page.locator(self.config.selectors.time_units).filter(has_text=label)
        self._require_count(locator, 1, f"time unit {label}")
        self._paced_click(locator)
        self._settle()
        state = self.read_state()
        if state.active_time_unit != label:
            raise SafetyHalt(f"Failed to select duration {label}; active={state.active_time_unit!r}")

    def fill_amount(self, amount: str) -> None:
        page = self._page()
        locator = page.locator(self.config.selectors.amount_input)
        self._require_count(locator, 1, "amount input")
        current = locator.input_value()
        if current.strip():
            try:
                if _decimal_text(current) == _decimal_text(amount):
                    return
            except SafetyHalt:
                pass
        self._pace()
        fp = self.config.browser.fingerprint
        if fp.enabled and fp.human_typing:
            locator.click()
            locator.press("Control+a")
            locator.press("Delete")
            delay = random.uniform(fp.min_type_delay_ms, fp.max_type_delay_ms)
            locator.type(str(amount), delay=delay)
        else:
            locator.fill(amount)
        self._settle()
        state = self.read_state()
        if _decimal_text(state.amount_value) != _decimal_text(amount):
            raise SafetyHalt(f"Amount verification failed: expected {amount}, saw {state.amount_value!r}")

    def execute_order(self, symbol: str, side: str, duration_label: str | None = None) -> OrderResult:
        if side not in {"up", "down"}:
            raise ValueError(f"side must be up/down, got {side!r}")

        state_before = self.prepare_order(symbol, duration_label)
        self.validate_payout_rate(state_before, side)
        if self.config.dry_run:
            return OrderResult("dry_run", "Prepared page only; no trade button was clicked.", state_before)

        self._validate_live_submit_safety(state_before)
        button_selector = self.config.selectors.buy_up_button if side == "up" else self.config.selectors.buy_down_button
        button = self._page().locator(button_selector)
        self._require_count(button, 1, "trade button")
        self._paced_click(button)
        self._settle()

        modal_text = self.visible_modal_text()
        if not modal_text:
            if self.config.risk.allow_direct_submit_without_confirmation:
                state_after = self.read_state()
                return OrderResult(
                    "submitted_without_modal",
                    "Clicked trade button and no confirmation modal appeared.",
                    state_before,
                    state_after,
                )
            raise SafetyHalt(
                "Trade button was clicked but no confirmation modal appeared. "
                "The site may have submitted directly; check the account before continuing."
            )

        if not self.config.click_confirm_order:
            return OrderResult(
                "needs_manual_confirm",
                "Confirmation modal is open; config click_confirm_order=false so the runner stopped before final confirm.",
                state_before,
                self.read_state(),
                modal_text=modal_text,
            )

        self._validate_modal_text(modal_text, state_before, side)
        confirm_button = self._confirm_button()
        self._paced_click(confirm_button)
        self._settle(seconds=self.config.browser.post_submit_settle_seconds)
        state_after = self.read_state()
        return OrderResult("submitted", "Order confirmation clicked.", state_before, state_after, modal_text=modal_text)

    def read_state(self) -> PageState:
        page = self._page()
        selector_payload = self.config.selectors.__dict__
        payload = page.evaluate(
            """
            (selectors) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const textOf = (selector) => norm(document.querySelector(selector)?.innerText || document.querySelector(selector)?.textContent || '');
              const valueOf = (selector) => document.querySelector(selector)?.value || '';
              const bodyText = norm(document.body?.innerText || '');
              const orderText = textOf(selectors.order_panel);
              const available = /可用\\s*([0-9,.]+)\\s*USDT/.exec(orderText);
              const payout = (selector) => {
                const text = textOf(selector);
                const match = /支付率\\s*([0-9,.]+)%/.exec(text);
                return match ? Number(match[1].replace(/,/g, '')) : null;
              };
              const second = document.querySelector('input.el-switch__input');
              const positionText = textOf(selectors.position_container);
              const hasPositionTable = /交易对\\s*方向\\s*开仓数量/.test(positionText);
              const openPositionVisible = hasPositionTable && !/暂无数据/.test(positionText);
              const loginLike = /登录|注册|Sign\\s*in|Log\\s*in/i.test(bodyText);
              const memberLike = !!document.querySelector('a[href*="/member"]') || /资产/.test(bodyText);
              return {
                currentSymbol: textOf(selectors.current_symbol),
                activeSymbolCard: textOf(selectors.active_symbol_card),
                activeTimeUnit: textOf(selectors.active_time_unit),
                amountValue: valueOf(selectors.amount_input),
                availableUsdt: available ? Number(available[1].replace(/,/g, '')) : null,
                buyUpPayoutPercent: payout(selectors.buy_up_button),
                buyDownPayoutPercent: payout(selectors.buy_down_button),
                secondConfirmationEnabled: second ? !!second.checked : null,
                loggedIn: memberLike && !loginLike,
                openPositionVisible,
                pageTextSample: bodyText.slice(0, 1500),
              };
            }
            """,
            selector_payload,
        )
        return PageState(
            url=page.url,
            title=page.title(),
            current_symbol=normalize_symbol(payload.get("currentSymbol", "")) if payload.get("currentSymbol") else None,
            active_symbol_card=normalize_symbol(payload.get("activeSymbolCard", "")) if payload.get("activeSymbolCard") else None,
            active_time_unit=payload.get("activeTimeUnit") or None,
            amount_value=payload.get("amountValue") or "",
            available_usdt=payload.get("availableUsdt"),
            buy_up_payout_percent=payload.get("buyUpPayoutPercent"),
            buy_down_payout_percent=payload.get("buyDownPayoutPercent"),
            second_confirmation_enabled=payload.get("secondConfirmationEnabled"),
            logged_in=bool(payload.get("loggedIn")),
            open_position_visible=bool(payload.get("openPositionVisible")),
            page_text_sample=payload.get("pageTextSample") or "",
        )

    def validate_ready_state(self, state: PageState, symbol: str, duration_label: str | None = None) -> None:
        target = normalize_symbol(symbol)
        duration = duration_label or self.config.duration_label
        self._assert_page_symbol(state, target)
        if not state.logged_in:
            raise SafetyHalt("HiBT page does not look logged in. Log in with the persistent browser profile first.")
        if state.active_time_unit != duration:
            raise SafetyHalt(f"Wrong duration: expected {duration}, saw {state.active_time_unit!r}")
        if _decimal_text(state.amount_value) != _decimal_text(self.config.amount_usdt):
            raise SafetyHalt(f"Wrong amount: expected {self.config.amount_usdt}, saw {state.amount_value!r}")
        if self.config.risk.require_amount_available:
            min_available = max(float(self.config.risk.min_available_usdt), float(self.config.amount_usdt))
            if state.available_usdt is None or state.available_usdt < min_available:
                raise SafetyHalt(f"Available USDT too low: need at least {min_available}, saw {state.available_usdt}")
        if self.config.risk.block_when_open_position_visible and state.open_position_visible:
            raise SafetyHalt("Open position table is not empty; blocked by risk.block_when_open_position_visible.")

    def validate_payout_rate(self, state: PageState, side: str) -> None:
        rate = state.buy_up_payout_percent if side == "up" else state.buy_down_payout_percent
        label = "买涨" if side == "up" else "买跌"
        minimum = self.config.risk.min_payout_rate_percent
        if rate is None:
            raise SafetyHalt(f"Could not read payout rate for {label}; blocked before order.")
        if rate < minimum:
            raise SafetyHalt(f"Payout rate too low for {label}: {rate:.2f}% < {minimum:.2f}%.")

    def visible_modal_text(self) -> str:
        page = self._page()
        return page.evaluate(
            """
            () => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const visible = (el) => {
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
              };
              const nodes = Array.from(document.querySelectorAll('.el-overlay,.el-message-box,.el-dialog'))
                .filter(visible)
                .map(el => norm(el.innerText || el.textContent))
                .filter(Boolean);
              return nodes.join('\\n').slice(0, 3000);
            }
            """
        )

    def _validate_live_submit_safety(self, state: PageState) -> None:
        risk = self.config.risk
        if risk.require_second_confirmation_enabled and not state.second_confirmation_enabled:
            raise SafetyHalt(
                "HiBT second confirmation appears disabled. Enable 下单二次确认 on the site, "
                "or set allow_direct_submit_without_confirmation=true only if you accept direct-submit risk."
            )

    def _validate_modal_text(self, modal_text: str, state: PageState, side: str) -> None:
        expected_side = "买涨" if side == "up" else "买跌"
        checks = [
            (state.current_symbol or "").replace("-", ""),
            expected_side,
            self.config.amount_usdt,
        ]
        missing = [item for item in checks if item and item not in modal_text.replace("-", "")]
        if missing:
            raise SafetyHalt(f"Confirmation modal did not contain expected fields {missing}; text={modal_text!r}")

    def _confirm_button(self) -> Locator:
        page = self._page()
        candidates = page.locator(self.config.selectors.visible_overlay_buttons).filter(
            has_text=re.compile(r"确认|确定|提交|下单")
        )
        count = candidates.count()
        if count != 1:
            raise SafetyHalt(f"Expected exactly one confirmation button, found {count}.")
        return candidates

    def _assert_page_symbol(self, state: PageState, target: str) -> None:
        if state.current_symbol != target:
            raise SafetyHalt(f"Wrong page symbol: expected {target}, saw {state.current_symbol!r} at {state.url}")
        if state.active_symbol_card and state.active_symbol_card != target:
            raise SafetyHalt(f"Wrong active symbol card: expected {target}, saw {state.active_symbol_card!r}")

    def _load_symbol_url(self, target: str) -> PageState:
        page = self._page()
        page.goto(f"{self.config.base_url}/{target}", wait_until="domcontentloaded")
        self._wait_for_cf_challenge(page)
        self._settle()
        page.wait_for_selector(self.config.selectors.order_panel, state="visible")
        state = self.read_state()
        self._assert_page_symbol(state, target)
        return state

    def _options_page_loaded(self) -> bool:
        page = self._page()
        if page.url == "about:blank":
            return False
        try:
            return page.locator(self.config.selectors.order_panel).count() > 0
        except Exception:
            return False

    def _click_symbol_card(self, target: str) -> None:
        page = self._page()
        cards = page.locator(self.config.selectors.symbol_cards)
        count = cards.count()
        if count == 0:
            raise SafetyHalt(f"No symbol cards found on the loaded options page; refused to reload for {target}.")

        index = page.evaluate(
            """
            ({ selector, target }) => {
              const compact = (value) => (value || '').toUpperCase().replace(/[\\s_\\/-]/g, '');
              const targetCompact = compact(target);
              const baseCompact = compact(target.split('-')[0]);
              const cards = Array.from(document.querySelectorAll(selector));
              return cards.findIndex((card) => {
                const label = card.querySelector('.txt-symbol')?.innerText || card.innerText || card.textContent || '';
                const cardCompact = compact(label);
                return cardCompact === targetCompact
                  || cardCompact === baseCompact
                  || cardCompact.startsWith(targetCompact)
                  || cardCompact.startsWith(baseCompact);
              });
            }
            """,
            {"selector": self.config.selectors.symbol_cards, "target": target},
        )
        if not isinstance(index, int) or index < 0:
            raise SafetyHalt(f"Could not find symbol card for {target}; refused to reload the page.")
        self._paced_click(cards.nth(index))

    def _wait_for_page_symbol(self, target: str) -> None:
        page = self._page()
        page.wait_for_function(
            """
            ({ currentSelector, activeSelector, target }) => {
              const textOf = (selector) => {
                const node = document.querySelector(selector);
                return (node?.innerText || node?.textContent || '').replace(/\\s+/g, ' ').trim();
              };
              const normalize = (value) => {
                const compact = (value || '').toUpperCase().replace(/[\\s_\\/-]/g, '');
                if (compact === 'BTC' || compact === 'BTCUSDT') return 'BTC-USDT';
                if (compact === 'ETH' || compact === 'ETHUSDT') return 'ETH-USDT';
                if (compact.endsWith('USDT') && compact.length > 4) {
                  return `${compact.slice(0, -4)}-USDT`;
                }
                return compact;
              };
              const current = normalize(textOf(currentSelector));
              const active = normalize(textOf(activeSelector));
              return current === target && (!active || active === target);
            }
            """,
            {
                "currentSelector": self.config.selectors.current_symbol,
                "activeSelector": self.config.selectors.active_symbol_card,
                "target": target,
            },
        )

    def _paced_click(self, locator: Locator) -> None:
        self._pace()
        locator.scroll_into_view_if_needed()
        self._move_mouse_to(locator)
        self._pace()
        fp = self.config.browser.fingerprint
        if fp.enabled and fp.human_typing:
            cfg = self.config.browser
            delay_ms = random.uniform(cfg.min_click_delay_ms, cfg.max_click_delay_ms)
            locator.click(delay=delay_ms)
        else:
            locator.click()

    def _move_mouse_to(self, locator: Locator) -> None:
        fp = self.config.browser.fingerprint
        if not fp.enabled or not fp.human_typing:
            return
        try:
            box = locator.bounding_box()
        except Exception:
            box = None
        if not box:
            return
        page = self._page()
        target_x = box["x"] + box["width"] * random.uniform(0.25, 0.75)
        target_y = box["y"] + box["height"] * random.uniform(0.25, 0.75)
        cfg = self.config.browser
        points = _bezier_path(
            self._last_mouse_x,
            self._last_mouse_y,
            target_x,
            target_y,
            cfg.min_mouse_path_points,
            cfg.max_mouse_path_points,
        )
        for px, py in points:
            try:
                page.mouse.move(px, py)
                time.sleep(random.uniform(cfg.min_mouse_step_delay_ms, cfg.max_mouse_step_delay_ms) / 1000.0)
            except Exception:
                break
        self._last_mouse_x = target_x
        self._last_mouse_y = target_y

    def _pace(self) -> None:
        cfg = self.config.browser
        time.sleep(random.uniform(cfg.min_action_delay_seconds, cfg.max_action_delay_seconds))

    def _settle(self, seconds: float | None = None) -> None:
        cfg = self.config.browser
        time.sleep(
            seconds
            if seconds is not None
            else random.uniform(cfg.min_settle_delay_seconds, cfg.max_settle_delay_seconds)
        )

    def _wait_for_cf_challenge(self, page: Page, timeout_s: float = 30.0) -> None:
        """Detect and wait for Cloudflare Turnstile challenge to resolve."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            is_cf = page.evaluate("""() => {
              const body = document.body ? document.body.innerText : '';
              const hasTurnstile = !!document.querySelector('iframe[src*="challenges.cloudflare.com"]');
              const hasCfText = /验证您不是自动程序|Verify you are human|Just a moment/i.test(body);
              return hasTurnstile || hasCfText;
            }""")
            if not is_cf:
                return
            time.sleep(1.5)
        raise SafetyHalt(
            "Cloudflare challenge page did not resolve within timeout. "
            "Check: 1) use real Chrome (channel='chrome'), "
            "2) avoid datacenter proxy IPs, "
            "3) try residential proxy or direct connection."
        )

    def _require_count(self, locator: Locator, expected: int, name: str) -> None:
        count = locator.count()
        if count != expected:
            raise SafetyHalt(f"Expected {expected} element(s) for {name}, found {count}.")

    def _page(self) -> Page:
        if self.page is None:
            raise RuntimeError("Browser is not started.")
        return self.page


def _bezier_path(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    min_points: int,
    max_points: int,
) -> list[tuple[float, float]]:
    """Generate a human-like cubic Bezier curve between two points."""
    dist = math.hypot(x1 - x0, y1 - y0)
    low = max(2, int(min_points))
    high = max(low, int(max_points))
    num_points = max(low, min(high, int(dist / 30)))
    spread = dist * 0.3
    cp1x = x0 + (x1 - x0) * random.uniform(0.2, 0.5) + random.uniform(-spread, spread) * 0.4
    cp1y = y0 + (y1 - y0) * random.uniform(0.1, 0.4) + random.uniform(-spread, spread) * 0.4
    cp2x = x0 + (x1 - x0) * random.uniform(0.5, 0.8) + random.uniform(-spread, spread) * 0.3
    cp2y = y0 + (y1 - y0) * random.uniform(0.6, 0.9) + random.uniform(-spread, spread) * 0.3
    points: list[tuple[float, float]] = []
    for i in range(num_points + 1):
        t = i / num_points
        t += random.uniform(-0.01, 0.01)
        t = max(0.0, min(1.0, t))
        u = 1 - t
        px = u**3 * x0 + 3 * u**2 * t * cp1x + 3 * u * t**2 * cp2x + t**3 * x1
        py = u**3 * y0 + 3 * u**2 * t * cp1y + 3 * u * t**2 * cp2y + t**3 * y1
        points.append((px, py))
    return points


def _decimal_text(value: str) -> Decimal:
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError) as exc:
        raise SafetyHalt(f"Could not parse decimal value: {value!r}") from exc
