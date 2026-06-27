#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT.parent / "aligned_data_oos"
HIBT_PAYOUT_RATE = 0.80
DEFAULT_BANKROLL = 200.0
MIN_BANKROLL = 2.0
KELLY_FRACTIONS = [
    ("BTC", "3m", 0.02207),
    ("ETH", "3m", 0.01297),
    ("BTC", "5m", 0.01218),
    ("ETH", "5m", 0.01264),
    ("BTC", "15m", 0.02069),
    ("ETH", "15m", 0.01839),
]


def venue_paths(venue: str) -> dict[str, Path | str]:
    if venue == "turboflow":
        return {
            "name": "TurboFlow",
            "control_file": ROOT / "turboflow" / "runtime" / "turboflow_control.json",
            "log_file": ROOT / "turboflow" / "runtime" / "turboflow_api_orders.jsonl",
        }
    if venue == "hibt":
        return {
            "name": "HiBT",
            "control_file": ROOT / "hibt" / "runtime" / "hibt_control.json",
            "log_file": ROOT / "hibt" / "runtime" / "hibt_api_orders.jsonl",
        }
    raise ValueError(f"unknown venue: {venue}")

INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Execution Panel</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1014;
      --panel: #17191f;
      --panel2: #1d2028;
      --line: #2b303a;
      --text: #f4f7fb;
      --muted: #8f98a8;
      --accent: #00d4ff;
      --accent2: #66f0c8;
      --bad: #ff6b6b;
      --warn: #ffc266;
      font-family: IBS-R, "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    .shell { min-height: 100vh; display: grid; grid-template-columns: 260px minmax(0, 1fr); }
    aside { border-right: 1px solid var(--line); background: #111319; padding: 18px; }
    main { padding: 18px; display: grid; gap: 12px; align-content: start; }
    h1 { margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0; }
    h2 { margin: 0 0 12px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .muted { color: var(--muted); font-size: 12px; }
    .brand { display: grid; gap: 4px; margin-bottom: 18px; }
    .panel { min-width: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .controls { display: grid; gap: 12px; }
    label.field { display: grid; gap: 7px; color: var(--muted); font-size: 12px; }
    input[type="number"] { width: 100%; border: 1px solid var(--line); border-radius: 8px; padding: 10px 11px; background: #0d0f14; color: var(--text); font: inherit; outline: none; }
    input[type="number"]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(0,212,255,.14); }
    .toggle { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 2px 0; font-size: 14px; }
    .switch { position: relative; width: 42px; height: 24px; flex: 0 0 auto; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider { position: absolute; inset: 0; cursor: pointer; border-radius: 8px; background: #2d3340; transition: .16s ease; }
    .slider:before { content: ""; position: absolute; width: 18px; height: 18px; left: 3px; top: 3px; border-radius: 6px; background: #d7dde8; transition: .16s ease; }
    .switch input:checked + .slider { background: var(--accent); }
    .switch input:checked + .slider:before { transform: translateX(18px); background: #061017; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
    button { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: #10131a; color: var(--text); font: inherit; cursor: pointer; }
    button.primary { background: var(--accent); border-color: var(--accent); color: #061017; font-weight: 700; }
    button:hover { border-color: var(--accent); }
    .topbar { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
    .status { display: inline-flex; align-items: center; gap: 7px; padding: 7px 10px; border: 1px solid var(--line); border-radius: 8px; background: #111319; color: var(--muted); font-size: 12px; }
    .dot { width: 7px; height: 7px; border-radius: 999px; background: var(--accent2); }
    .stats { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 10px; }
    .stat { min-height: 82px; display: grid; align-content: space-between; background: var(--panel2); }
    .stat b { display: block; margin-top: 8px; font-size: 26px; font-weight: 700; letter-spacing: 0; }
    .content-grid { display: grid; grid-template-columns: minmax(360px, .72fr) minmax(0, 1.28fr); gap: 12px; align-items: stretch; }
    .stack { min-width: 0; display: grid; grid-template-rows: minmax(300px, 1fr) auto; gap: 12px; }
    .chart-panel { display: grid; grid-template-rows: auto minmax(270px, 1fr); }
    .chart { height: 100%; min-height: 270px; }
    .table-scroll { max-height: calc(100vh - 232px); min-height: 432px; overflow: auto; }
    canvas { width: 100%; height: 100%; display: block; }
    table { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 13px; }
    th, td { height: 38px; padding: 8px 8px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    th { position: sticky; top: 0; z-index: 1; background: var(--panel); color: var(--muted); font-size: 12px; font-weight: 700; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    tbody tr:hover { background: rgba(0,212,255,.06); }
    .pill, .tag { display: inline-flex; align-items: center; justify-content: center; min-width: 38px; border: 1px solid var(--line); border-radius: 8px; padding: 4px 8px; background: #10131a; }
    .tag.up { border-color: rgba(102,240,200,.42); color: var(--accent2); }
    .tag.down { border-color: rgba(255,107,107,.42); color: var(--bad); }
    .ok { color: var(--accent2); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .stats, .content-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">
        <h1 id="venue_title">Execution</h1>
        <span class="muted">execution panel</span>
      </div>
      <div class="controls" id="controls"></div>
    </aside>
    <main>
      <div class="topbar">
        <div>
          <h1>Live Execution</h1>
          <div class="muted" id="venue_note"></div>
        </div>
        <div class="status"><span class="dot"></span><span id="status">loading</span></div>
      </div>
      <section class="stats" id="stats"></section>
      <section class="content-grid">
        <div class="stack">
          <section class="panel chart-panel">
            <h2>Profit Curve</h2>
            <div class="chart"><canvas id="equity"></canvas></div>
          </section>
          <section class="panel">
            <h2>Timeframes</h2>
            <table>
              <colgroup><col style="width:12%"><col style="width:10%"><col style="width:20%"><col style="width:14%"><col style="width:14%"><col style="width:14%"><col style="width:16%"></colgroup>
              <thead><tr><th>周期</th><th class="num">次数</th><th class="num">胜率</th><th class="num">回报</th><th class="num">信号</th><th class="num">执行</th><th class="num">盈亏</th></tr></thead>
              <tbody id="timeframes"></tbody>
            </table>
          </section>
        </div>
        <section class="panel">
          <h2>Recent Orders</h2>
          <div class="table-scroll">
            <table>
              <colgroup><col style="width:12%"><col style="width:14%"><col style="width:9%"><col style="width:9%"><col style="width:11%"><col style="width:11%"><col style="width:11%"><col style="width:10%"><col style="width:13%"></colgroup>
              <thead><tr><th>时间</th><th>市场</th><th>方向</th><th class="num">金额</th><th class="num">总耗时</th><th class="num">信号</th><th class="num">执行</th><th>结果</th><th class="num">盈亏</th></tr></thead>
              <tbody id="recent"></tbody>
            </table>
          </div>
        </section>
      </section>
    </main>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    let kellyFractions = [];
    let currentVenue = "turboflow";
    let dirty = false;
    let loadedOnce = false;

    function pct(value) { return value === null || value === undefined ? "N/A" : `${(Number(value) * 100).toFixed(1)}%`; }
    function money(value) { return value === null || value === undefined ? "N/A" : Number(value).toFixed(2); }
    function secs(value) {
      if (value === null || value === undefined) return "N/A";
      const number = Number(value);
      if (number >= 3600) return `${(number / 3600).toFixed(1)}h`;
      if (number >= 60) return `${(number / 60).toFixed(1)}m`;
      return `${number.toFixed(1)}s`;
    }
    function pnlClass(value) { return value === null || value === undefined ? "warn" : Number(value) >= 0 ? "ok" : "bad"; }
    function cellClass(value) { return value === true || value === "win" ? "ok" : value === false || value === "loss" ? "bad" : "warn"; }
    function marketText(row) { return `${(row.symbol || "").replace("-USDT", "").replace("USDT", "")} ${row.timeframe || ""}`.trim(); }
    function sideText(value) { return String(value || "").toUpperCase() === "SELL" ? "DOWN" : "UP"; }
    function shortTime(value) {
      const text = value || "";
      const match = text.match(/T(\d{2}:\d{2})/);
      return match ? text.slice(5, 10) + " " + match[1] : text;
    }
    function timeLabel(value) {
      const date = new Date(value);
      const pad = (number) => String(number).padStart(2, "0");
      return `${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}`;
    }
    function winText(row) { return row.settled ? `${pct(row.win_rate)} (${row.wins}/${row.settled})` : "N/A"; }
    function kellyAmount(bankroll, fraction) { return Math.max(2, Number(bankroll || 200) * fraction); }
    function renderControls(control, kellyBankroll, bankrollSource) {
      const isTurbo = currentVenue === "turboflow";
      $("controls").innerHTML = `
        <section class="panel">
          <h2>Execution</h2>
          <label class="toggle">${isTurbo ? "TurboFlow" : "HiBT"} <span class="switch"><input id="enabled" type="checkbox"><span class="slider"></span></span></label>
        </section>
        <section class="panel">
          <h2>${isTurbo ? "Dry-run" : "Order"}</h2>
          ${isTurbo
            ? `<label class="field">Dry-run bankroll<input id="bankroll" type="number" min="2" step="0.01"></label>`
            : `<label class="field">下单金额<input id="order_amount" type="number" min="0.01" step="0.01"></label>`}
          <div class="actions">
            <button id="refresh">刷新</button>
            <button class="primary" id="save">保存</button>
          </div>
        </section>
        ${isTurbo ? `<section class="panel">
          <h2>Kelly Amount</h2>
          <div class="muted" id="kelly_source"></div>
          <table>
            <colgroup><col style="width:32%"><col style="width:28%"><col style="width:40%"></colgroup>
            <thead><tr><th>币种</th><th>周期</th><th class="num">金额</th></tr></thead>
            <tbody id="kelly"></tbody>
          </table>
        </section>` : ""}
      `;
      $("refresh").addEventListener("click", load);
      $("save").addEventListener("click", () => save().catch((err) => $("status").textContent = err.message));
      document.querySelectorAll("#controls input").forEach((input) => {
        input.addEventListener("input", () => {
          if (loadedOnce) dirty = true;
          if (currentVenue === "turboflow") renderKelly($("bankroll").value, "dry-run input");
          $("status").textContent = "未保存";
        });
      });
      $("enabled").checked = Boolean(control.enabled);
      if (isTurbo) {
        $("bankroll").value = control.bankroll;
        renderKelly(kellyBankroll, bankrollSource);
      } else {
        $("order_amount").value = control.order_amount;
      }
    }
    function renderKelly(bankroll, source) {
      const table = $("kelly");
      if (!table) return;
      const sourceEl = $("kelly_source");
      if (sourceEl) sourceEl.textContent = `bankroll ${money(bankroll)} · ${source || "dry-run"}`;
      table.innerHTML = kellyFractions.map(([symbol, timeframe, fraction]) => `
        <tr><td>${symbol}</td><td><span class="pill">${timeframe}</span></td><td class="num">${money(kellyAmount(bankroll, fraction))}</td></tr>
      `).join("");
    }
    function setForm(control, kellyBankroll, bankrollSource) {
      if (dirty) return;
      renderControls(control, kellyBankroll, bankrollSource);
    }
    function collectForm() {
      if (currentVenue === "turboflow") return { enabled: $("enabled").checked, bankroll: Number($("bankroll").value) };
      return { enabled: $("enabled").checked, order_amount: Number($("order_amount").value) };
    }
    function renderStats(stats) {
      const wins = stats.wins ?? 0;
      const winLabel = stats.settled ? `胜率 · ${wins}/${stats.settled}` : "胜率";
      $("stats").innerHTML = [
        ["执行次数", stats.count],
        ["成功率", pct(stats.success_rate)],
        ["平均回报", pct(stats.avg_payout_rate)],
        ["总耗时", secs(stats.avg_order_delay_seconds)],
        ["信号延迟", secs(stats.avg_signal_delay_seconds)],
        [winLabel, pct(stats.win_rate)],
        ["总盈亏", money(stats.pnl)],
      ].map(([label, value]) => `<div class="panel stat"><span class="muted">${label}</span><b>${value}</b></div>`).join("");
      $("timeframes").innerHTML = Object.entries(stats.timeframes).map(([name, row]) => `
        <tr>
          <td><span class="pill">${name}</span></td>
          <td class="num">${row.count}</td>
          <td class="num">${winText(row)}</td>
          <td class="num">${pct(row.avg_payout_rate)}</td>
          <td class="num">${secs(row.avg_signal_delay_seconds)}</td>
          <td class="num">${secs(row.avg_trader_delay_seconds)}</td>
          <td class="num ${pnlClass(row.pnl)}">${money(row.pnl)}</td>
        </tr>
      `).join("");
    }
    function renderCurve(points) {
      const canvas = $("equity");
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      const padLeft = 52, padRight = 18, padTop = 28, padBottom = 38;
      const width = rect.width - padLeft - padRight;
      const height = rect.height - padTop - padBottom;
      const axisColor = "rgba(143,152,168,.65)";
      const gridColor = "rgba(143,152,168,.18)";
      const labelColor = "#8f98a8";
      const drawAxes = () => {
        ctx.strokeStyle = axisColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padLeft, padTop);
        ctx.lineTo(padLeft, padTop + height);
        ctx.lineTo(padLeft + width, padTop + height);
        ctx.stroke();
      };
      if (!points || points.length === 0) {
        drawAxes();
        ctx.fillStyle = labelColor;
        ctx.font = "12px system-ui";
        ctx.textAlign = "center";
        ctx.fillText("暂无已结算盈亏", padLeft + width / 2, padTop + height / 2);
        return;
      }
      const times = points.map((point) => Date.parse(point.logged_at));
      const firstTime = Math.min(...times);
      const lastTime = Math.max(...times);
      const span = Math.max(1, lastTime - firstTime);
      const yValues = points.map((point) => Number(point.pnl));
      const minValue = Math.min(0, ...yValues);
      const maxValue = Math.max(0, ...yValues);
      const rawStep = Math.max(1, (maxValue - minValue) / 4);
      const power = Math.pow(10, Math.floor(Math.log10(rawStep)));
      const ratio = rawStep / power;
      const step = (ratio <= 2 ? 2 : ratio <= 5 ? 5 : 10) * power;
      const min = Math.floor(minValue / step) * step;
      const max = Math.ceil(maxValue / step) * step;
      const xFor = (point) => padLeft + ((Date.parse(point.logged_at) - firstTime) / span) * width;
      const yFor = (value) => padTop + height - ((value - min) / Math.max(step, max - min)) * height;
      ctx.strokeStyle = gridColor;
      ctx.fillStyle = labelColor;
      ctx.font = "12px system-ui";
      ctx.textAlign = "right";
      for (let value = min; value <= max + step / 2; value += step) {
        const y = yFor(value);
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(padLeft + width, y);
        ctx.stroke();
        ctx.fillText(money(value), padLeft - 8, y + 4);
      }
      drawAxes();
      ctx.fillStyle = labelColor;
      ctx.textAlign = "left";
      ctx.fillText(timeLabel(firstTime), padLeft, padTop + height + 24);
      ctx.textAlign = "center";
      ctx.fillText(timeLabel(firstTime + span / 2), padLeft + width / 2, padTop + height + 24);
      ctx.textAlign = "right";
      ctx.fillText(timeLabel(lastTime), padLeft + width, padTop + height + 24);
      ctx.strokeStyle = "#00d4ff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      points.forEach((point, index) => {
        const x = xFor(point);
        const y = yFor(Number(point.pnl));
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    function renderRecent(rows) {
      $("recent").innerHTML = rows.map((row) => {
        const side = sideText(row.side);
        return `
          <tr>
            <td title="${row.logged_at || ""}">${shortTime(row.logged_at)}</td>
            <td>${marketText(row)}</td>
            <td><span class="tag ${side === "UP" ? "up" : "down"}">${side}</span></td>
            <td class="num">${money(row.amount)}</td>
            <td class="num">${secs(row.order_delay_seconds)}</td>
            <td class="num">${secs(row.signal_delay_seconds)}</td>
            <td class="num">${secs(row.trader_delay_seconds)}</td>
            <td class="${cellClass(row.outcome)}">${row.outcome || "pending"}</td>
            <td class="num ${pnlClass(row.pnl)}">${money(row.pnl)}</td>
          </tr>
        `;
      }).join("");
    }
    async function load() {
      const res = await fetch("/api/state");
      const data = await res.json();
      currentVenue = data.venue;
      kellyFractions = data.kelly_fractions || [];
      $("venue_title").textContent = data.venue_name || "Execution";
      $("venue_note").textContent = currentVenue === "turboflow"
        ? "Live orders use the TurboFlow USDT available balance; the input only changes dry-run sizing."
        : "金额和开关写入控制文件，trader 下一次信号处理生效。";
      const kellyBankroll = data.stats.kelly_bankroll || data.control.bankroll;
      const bankrollSource = data.stats.latest_bankroll ? "latest live order" : "dry-run equity";
      setForm(data.control, kellyBankroll, bankrollSource);
      renderStats(data.stats);
      renderCurve(data.stats.equity_curve);
      renderRecent(data.recent);
      $("status").textContent = dirty ? "未保存" : `更新 ${new Date().toLocaleTimeString()}`;
      loadedOnce = true;
    }
    async function save() {
      $("status").textContent = "保存中";
      const res = await fetch("/api/control", {
        method: "POST",
        headers: {"content-type": "application/json"},
        body: JSON.stringify(collectForm()),
      });
      if (!res.ok) throw new Error(await res.text());
      dirty = false;
      await load();
    }
    load();
    setInterval(load, 5000);
  </script>
</body>
</html>
"""


def default_control(venue: str) -> dict[str, Any]:
    if venue == "turboflow":
        return {"enabled": True, "bankroll": DEFAULT_BANKROLL}
    return {"enabled": True, "order_amount": 3.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local web control panel for live execution.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--venue", choices=("turboflow", "hibt"), default="turboflow")
    parser.add_argument("--control-file", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tail", type=int, default=0, help="Rows to read from the order log. 0 reads full history.")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_control(path: Path, venue: str) -> dict[str, Any]:
    if not path.exists():
        return default_control(venue)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_control(venue)
    return sanitize_control(data if isinstance(data, dict) else {}, venue)


def sanitize_control(data: dict[str, Any], venue: str) -> dict[str, Any]:
    if venue == "turboflow":
        return {
            "enabled": bool(data.get("enabled", True)),
            "bankroll": max(MIN_BANKROLL, positive_float(data.get("bankroll"), DEFAULT_BANKROLL)),
        }
    return {
        "enabled": bool(data.get("enabled", True)),
        "order_amount": positive_float(data.get("order_amount"), 3.0),
    }


def positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] | list[dict[str, Any]]
    rows = [] if limit <= 0 else deque(maxlen=limit)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return list(rows)


def build_state(control_file: Path, log_file: Path, data_root: Path, tail: int, venue: str, venue_name: str) -> dict[str, Any]:
    records = read_jsonl_tail(log_file, tail)
    closes = read_closes_for(records, data_root)
    control = read_control(control_file, venue)
    stat_rows = stats(records, closes)
    if venue == "turboflow":
        stat_rows["kelly_bankroll"] = stat_rows["latest_bankroll"] or max(MIN_BANKROLL, control["bankroll"] + stat_rows["pnl"])
    return {
        "venue": venue,
        "venue_name": venue_name,
        "kelly_fractions": KELLY_FRACTIONS if venue == "turboflow" else [],
        "control": control,
        "stats": stat_rows,
        "recent": recent(records, closes),
        "files": {
            "control_file": str(control_file),
            "log_file": str(log_file),
            "data_root": str(data_root),
        },
    }


def stats(records: list[dict[str, Any]], closes: dict[str, dict[int, float]]) -> dict[str, Any]:
    items = [(row, settle(row, closes)) for row in records]
    return {
        **summarize(items),
        "records": len(records),
        "timeframes": {
            name: summarize([item for item in items if timeframe_key(item[0]) == name], total=len(items))
            for name in sorted({timeframe_key(row) for row, _outcome in items})
        },
    }


def summarize(items: list[tuple[dict[str, Any], dict[str, Any] | None]], total: int | None = None) -> dict[str, Any]:
    count = len(items)
    successes = sum(1 for row, _outcome in items if row.get("success"))
    latency_rows = [
        (total_delay, signal_delay, trader_delay)
        for row, _outcome in items
        if (total_delay := order_delay_seconds(row)) is not None
        and (signal_delay := signal_delay_seconds(row)) is not None
        and (trader_delay := trader_delay_seconds(row)) is not None
    ]
    outcomes = [outcome for row, outcome in items if row.get("success") and outcome is not None]
    payout_rates = [payout_rate_for(row) for row, _outcome in items if row.get("success")]
    wins = sum(1 for outcome in outcomes if outcome["win"])
    pnl = sum(pnl_for(row, outcome) for row, outcome in items if row.get("success") and outcome is not None)
    return {
        "count": count,
        "share": None if total in (None, 0) else count / total,
        "success": successes,
        "success_rate": None if count == 0 else successes / count,
        "avg_order_delay_seconds": None if not latency_rows else sum(row[0] for row in latency_rows) / len(latency_rows),
        "avg_signal_delay_seconds": None if not latency_rows else sum(row[1] for row in latency_rows) / len(latency_rows),
        "avg_trader_delay_seconds": None if not latency_rows else sum(row[2] for row in latency_rows) / len(latency_rows),
        "avg_payout_rate": None if not payout_rates else sum(payout_rates) / len(payout_rates),
        "settled": len(outcomes),
        "wins": wins,
        "win_rate": None if not outcomes else wins / len(outcomes),
        "pnl": pnl,
        "equity_curve": equity_curve(items),
        "latest_bankroll": latest_bankroll([row for row, _outcome in items]),
    }


def latest_bankroll(records: list[dict[str, Any]]) -> float | None:
    for row in reversed(records):
        if row.get("bankroll_source") != "turboflow":
            continue
        value = parse_amount(row.get("bankroll"))
        if value is not None:
            return value
    return None


def equity_curve(items: list[tuple[dict[str, Any], dict[str, Any] | None]]) -> list[dict[str, Any]]:
    pnl = 0.0
    points = []
    for row, outcome in sorted(items, key=lambda item: str(item[0].get("logged_at", ""))):
        if not row.get("success") or outcome is None:
            continue
        pnl += pnl_for(row, outcome)
        points.append({"logged_at": row.get("logged_at", ""), "pnl": pnl})
    return points


def recent(records: list[dict[str, Any]], closes: dict[str, dict[int, float]]) -> list[dict[str, Any]]:
    rows = []
    for row in reversed(records[-50:]):
        signal = row.get("signal") if isinstance(row.get("signal"), dict) else {}
        order = row.get("order") if isinstance(row.get("order"), dict) else {}
        outcome = settle(row, closes)
        rows.append(
            {
                "logged_at": row.get("logged_at", ""),
                "signal_id": signal.get("signal_id", order.get("signal_id", "")),
                "symbol": signal.get("symbol", ""),
                "timeframe": signal.get("timeframe", ""),
                "side": signal.get("side", ""),
                "amount": parse_amount(order.get("amount")),
                "order_delay_seconds": order_delay_seconds(row),
                "signal_delay_seconds": signal_delay_seconds(row),
                "trader_delay_seconds": trader_delay_seconds(row),
                "success": bool(row.get("success")),
                "outcome": None if outcome is None else ("win" if outcome["win"] else "loss"),
                "pnl": None if outcome is None else pnl_for(row, outcome),
            }
        )
    return rows


def timeframe_key(row: dict[str, Any]) -> str:
    signal = row.get("signal") if isinstance(row.get("signal"), dict) else {}
    return str(signal.get("timeframe") or "unknown").strip().lower() or "unknown"


def order_delay_seconds(row: dict[str, Any]) -> float | None:
    signal = row.get("signal") if isinstance(row.get("signal"), dict) else {}
    try:
        event_start = parse_dt(signal.get("decision_time"))
        order_start = parse_dt(row.get("order_started_at"))
    except (TypeError, ValueError):
        return None
    return (order_start - event_start).total_seconds()


def signal_delay_seconds(row: dict[str, Any]) -> float | None:
    signal = row.get("signal") if isinstance(row.get("signal"), dict) else {}
    try:
        event_start = parse_dt(signal.get("decision_time"))
        generated_at = parse_dt(signal.get("signal_generated_at"))
    except (TypeError, ValueError):
        return None
    return (generated_at - event_start).total_seconds()


def trader_delay_seconds(row: dict[str, Any]) -> float | None:
    signal = row.get("signal") if isinstance(row.get("signal"), dict) else {}
    try:
        generated_at = parse_dt(signal.get("signal_generated_at"))
        order_start = parse_dt(row.get("order_started_at"))
    except (TypeError, ValueError):
        return None
    return (order_start - generated_at).total_seconds()


def pnl_for(row: dict[str, Any], outcome: dict[str, Any]) -> float:
    order = row.get("order") if isinstance(row.get("order"), dict) else {}
    amount = parse_amount(order.get("amount")) or 0.0
    return amount * payout_rate_for(row) if outcome["win"] else -amount


def payout_rate_for(row: dict[str, Any]) -> float:
    return parse_amount(row.get("payout_rate")) or HIBT_PAYOUT_RATE


def parse_amount(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_closes_for(records: list[dict[str, Any]], data_root: Path) -> dict[str, dict[int, float]]:
    symbols = {
        normalize_symbol((row.get("signal") or {}).get("symbol", ""))
        for row in records
        if isinstance(row.get("signal"), dict)
    }
    return {symbol: read_live_closes(data_root, symbol) for symbol in symbols if symbol}


def read_live_closes(data_root: Path, symbol: str) -> dict[int, float]:
    path = data_root / "binance_spot_klines" / symbol / "1m_live.csv"
    closes: dict[int, float] = {}
    if not path.exists():
        return closes
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                closes[int(row["open_time"])] = float(row["close"])
            except (KeyError, TypeError, ValueError):
                continue
    return closes


def settle(row: dict[str, Any], closes: dict[str, dict[int, float]]) -> dict[str, Any] | None:
    signal = row.get("signal") if isinstance(row.get("signal"), dict) else {}
    try:
        minutes = timeframe_minutes(signal.get("timeframe"))
        decision_time = parse_dt(signal.get("decision_time"))
        start_time = parse_dt(signal.get("last_kline_time"))
        end_time = decision_time + timedelta(minutes=minutes - 1)
        direction = signal_side(signal.get("side"))
    except (TypeError, ValueError):
        return None
    table = closes.get(normalize_symbol(signal.get("symbol", "")), {})
    start_close = table.get(to_ms(start_time))
    end_close = table.get(to_ms(end_time))
    if start_close is None or end_close is None or start_close == end_close:
        return None
    return {
        "win": end_close > start_close if direction == "up" else end_close < start_close,
        "start_close": start_close,
        "end_close": end_close,
    }


def parse_dt(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty datetime")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def timeframe_minutes(value: Any) -> int:
    text = str(value or "").strip().lower().replace("min", "m")
    if text.isdigit():
        return int(text)
    if text.endswith("m") and text[:-1].isdigit():
        return int(text[:-1])
    raise ValueError(f"bad timeframe: {value!r}")


def signal_side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"BUY", "UP", "LONG", "CALL", "BULL", "RISE", "1"}:
        return "up"
    if text in {"SELL", "DOWN", "SHORT", "PUT", "BEAR", "FALL", "-1"}:
        return "down"
    raise ValueError(f"bad side: {value!r}")


def normalize_symbol(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "").replace("/", "").replace("_", "")
    if text in {"BTC", "ETH", "SOL"}:
        text += "USDT"
    return text


def make_handler(control_file: Path, log_file: Path, data_root: Path, tail: int, venue: str, venue_name: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            if urlparse(self.path).path == "/":
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.end_headers()
                return
            self.send_error(404)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self.send_html(INDEX_HTML)
                return
            if path == "/api/state":
                self.send_json(build_state(control_file, log_file, data_root, tail, venue, venue_name))
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/control":
                self.send_error(404)
                return
            length = int(self.headers.get("content-length", "0"))
            if length > 100_000:
                self.send_error(413)
                return
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self.send_error(400, "bad json")
                return
            if not isinstance(payload, dict):
                self.send_error(400, "json object required")
                return
            control = sanitize_control(payload, venue)
            write_json(control_file, control)
            self.send_json({"ok": True, "control": control})

        def send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, data: Any) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def self_test() -> None:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        data_root = root / "aligned_data_oos"
        live = data_root / "binance_spot_klines" / "BTCUSDT" / "1m_live.csv"
        live.parent.mkdir(parents=True)
        decision = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
        live.write_text(
            "open_time,close\n"
            f"{to_ms(decision - timedelta(minutes=1))},100\n"
            f"{to_ms(decision + timedelta(minutes=4))},101\n",
            encoding="utf-8",
        )
        log = root / "hibt_api_orders.jsonl"
        row = {
            "logged_at": decision.isoformat(),
            "signal": {
                "signal_id": "x",
                "symbol": "BTCUSDT",
                "timeframe": "5m",
                "side": "BUY",
                "timestamp": decision.isoformat(),
                "decision_time": decision.isoformat(),
                "last_kline_time": (decision - timedelta(minutes=1)).isoformat(),
                "signal_generated_at": (decision + timedelta(seconds=1)).isoformat(),
            },
            "order": {"amount": "3", "signal_id": "x"},
            "payout_rate": 0.8,
            "order_started_at": (decision + timedelta(seconds=2)).isoformat(),
            "success": True,
            "bankroll": 250.0,
            "bankroll_source": "turboflow",
        }
        log.write_text(json.dumps({**row, "success": False}) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        state = build_state(root / "control.json", log, data_root, 100, "hibt", "HiBT")
        full_state = build_state(root / "control.json", log, data_root, 0, "hibt", "HiBT")
        assert state["stats"]["count"] == 2
        assert full_state["stats"]["count"] == 2
        assert build_state(root / "control.json", log, data_root, 1, "hibt", "HiBT")["stats"]["count"] == 1
        assert state["stats"]["win_rate"] == 1.0
        assert abs(state["stats"]["pnl"] - 2.4) < 1e-9
        assert state["stats"]["avg_payout_rate"] == HIBT_PAYOUT_RATE
        assert abs(state["stats"]["equity_curve"][-1]["pnl"] - 2.4) < 1e-9
        assert state["stats"]["timeframes"]["5m"]["avg_order_delay_seconds"] == 2.0
        tf_state = build_state(root / "tf_control.json", log, data_root, 1, "turboflow", "TurboFlow")
        assert tf_state["control"]["bankroll"] == DEFAULT_BANKROLL
        assert tf_state["stats"]["latest_bankroll"] == 250.0
        assert tf_state["stats"]["kelly_bankroll"] == 250.0
        dry_log = root / "dry.jsonl"
        dry_log.write_text(json.dumps({**row, "bankroll": 200.0, "bankroll_source": "dry_run"}) + "\n", encoding="utf-8")
        assert abs(build_state(root / "tf_control.json", dry_log, data_root, 1, "turboflow", "TurboFlow")["stats"]["kelly_bankroll"] - 202.4) < 1e-9


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    paths = venue_paths(args.venue)
    control_file = args.control_file or Path(paths["control_file"])
    log_file = args.log_file or Path(paths["log_file"])
    handler = make_handler(control_file, log_file, args.data_root, args.tail, args.venue, str(paths["name"]))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"control panel http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
