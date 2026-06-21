#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightgbm_5m_direction_btc_eth import SYMBOL_COLORS, sharpe_ratio


ROOT = Path("results")
PERIODS = (3, 5, 10, 15, 30)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def merge_period(minutes: int) -> None:
    btc_dir = ROOT / f"btc_eth_external_{minutes}m"
    eth_dir = ROOT / f"eth_btc_external_{minutes}m"
    out_dir = ROOT / f"baseline_{minutes}m"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    btc_summary = read_json(btc_dir / "summary.json")
    eth_summary = read_json(eth_dir / "summary.json")
    summary = dict(btc_summary)
    summary["symbols"] = ["BTCUSDT", "ETHUSDT"]
    summary["feature_set"] = "v1_sessions_price_position_phase_peer_external"
    summary["feature_set_description"] = (
        "new baseline: BTC uses ETH peer external features; ETH uses BTC peer external features"
    )
    summary["samples"] = min(int(btc_summary["samples"]), int(eth_summary["samples"]))
    summary["ETH"] = eth_summary["ETH"]
    write_json(out_dir / "summary.json", summary)

    btc_pred = pd.read_csv(btc_dir / "predictions.csv", parse_dates=["decision_time"])
    eth_pred = pd.read_csv(eth_dir / "predictions.csv", parse_dates=["decision_time"])
    predictions = btc_pred.merge(eth_pred, on="decision_time", how="outer").sort_values("decision_time")
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    equity = pd.DataFrame({"decision_time": predictions["decision_time"]})
    equity["BTC"] = predictions["BTC_pnl"].fillna(0.0).cumsum()
    equity["ETH"] = predictions["ETH_pnl"].fillna(0.0).cumsum()
    equity.to_csv(out_dir / "equity_curve.csv", index=False)
    plot_equity(out_dir, equity, minutes)
    write_json(out_dir / "computed_metrics.json", computed_metrics(out_dir, predictions, minutes, summary))

    btc_folds = pd.read_csv(btc_dir / "folds.csv")
    eth_folds = pd.read_csv(eth_dir / "folds.csv")
    folds = btc_folds.merge(eth_folds, on=["fold", "test_start", "test_end"], how="outer").sort_values("fold")
    folds.to_csv(out_dir / "folds.csv", index=False)

    features = {"features_by_symbol": {}}
    features["features_by_symbol"].update(read_json(btc_dir / "feature_columns.json")["features_by_symbol"])
    features["features_by_symbol"].update(read_json(eth_dir / "feature_columns.json")["features_by_symbol"])
    write_json(out_dir / "feature_columns.json", features)

    params = {}
    params.update(read_json(btc_dir / "optuna_best_params.json"))
    params.update(read_json(eth_dir / "optuna_best_params.json"))
    write_json(out_dir / "optuna_best_params.json", params)

    dataset_info = read_json(btc_dir / "dataset_info.json")
    dataset_info["symbols"] = ["BTCUSDT", "ETHUSDT"]
    dataset_info["target_symbols"] = ["BTCUSDT", "ETHUSDT"]
    dataset_info["feature_set"] = "v1_sessions_price_position_phase_peer_external"
    write_json(out_dir / "dataset_info.json", dataset_info)


def max_drawdown(pnl: pd.Series) -> float:
    equity = pnl.fillna(0.0).cumsum()
    return float((equity - equity.cummax()).min())


def symbol_metrics(predictions: pd.DataFrame, prefix: str, minutes: int) -> dict:
    trades = predictions[f"{prefix}_is_trade"].fillna(False).astype(bool)
    wins = predictions[f"{prefix}_is_win"].fillna(False).astype(bool)
    ties = predictions.get(f"{prefix}_is_tie", pd.Series(False, index=predictions.index)).fillna(False).astype(bool)
    strict_trades = trades & ~ties
    pnl = predictions[f"{prefix}_pnl"].fillna(0.0).astype(float)
    return {
        "trades": int(trades.sum()),
        "tie_trades": int((trades & ties).sum()),
        "wins": int(wins[strict_trades].sum()),
        "losses": int((strict_trades & ~wins).sum()),
        "win_rate_ex_ties": float(wins[strict_trades].mean()) if strict_trades.any() else float("nan"),
        "pnl": float(pnl.sum()),
        "annualized_sharpe": sharpe_ratio(pnl.to_numpy(), minutes),
        "max_drawdown": max_drawdown(pnl),
    }


def computed_metrics(out_dir: Path, predictions: pd.DataFrame, minutes: int, summary: dict) -> dict:
    btc = symbol_metrics(predictions, "BTC", minutes)
    eth = symbol_metrics(predictions, "ETH", minutes)
    combo_pnl = predictions["BTC_pnl"].fillna(0.0).astype(float) + predictions["ETH_pnl"].fillna(0.0).astype(float)
    combo_trades = predictions["BTC_is_trade"].fillna(False).astype(bool) | predictions["ETH_is_trade"].fillna(False).astype(bool)
    return {
        "baseline_alias": str(out_dir),
        "current_baseline": True,
        "period": f"{minutes}m",
        "source": "merged peer external BTC/ETH walk-forward results",
        "feature_set": summary["feature_set"],
        "symbols": {"BTC": btc, "ETH": eth},
        "combo": {
            "samples": int(len(predictions)),
            "trades": int(combo_trades.sum()),
            "pnl": float(combo_pnl.sum()),
            "annualized_sharpe": sharpe_ratio(combo_pnl.to_numpy(), minutes),
            "max_drawdown": max_drawdown(combo_pnl),
        },
    }


def plot_equity(out_dir: Path, equity: pd.DataFrame, minutes: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for prefix in ("BTC", "ETH"):
        ax.plot(pd.to_datetime(equity["decision_time"]), equity[prefix], label=prefix, color=SYMBOL_COLORS.get(prefix))
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title(f"LightGBM {minutes}m Direction Walk-Forward Equity")
    ax.set_xlabel("Decision time")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curve.png", dpi=160)
    plt.close(fig)


def main() -> int:
    for minutes in PERIODS:
        merge_period(minutes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
