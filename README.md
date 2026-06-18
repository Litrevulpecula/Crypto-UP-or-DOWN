# Crypto UP or DOWN

用于 Binance K 线对齐、LightGBM 步进式（walk-forward）研究，以及实盘信号生成的 Python 工作流。

## 当前研究基线

当前的基线结果目录为：

- `results/baseline_3m`
- `results/baseline_5m`
- `results/baseline_10m`
- `results/baseline_15m`
- `results/baseline_30m`

汇总基线指标位于 `results/baseline_summary.json`。这些基线使用 v1 LightGBM 步进式流程，包含 UTC 交易时段指标和现货收盘位置特征。

## 实盘模型刷新

实盘模型按月重训。训练截止点必须是带有完整目标标签的最近一个完整月份：

- 5m 截止点：当前 UTC 月初减 5 分钟，例如在 2026 年 6 月运行时为 `2026-05-31T23:55:00Z`。
- 15m 截止点：当前 UTC 月初减 15 分钟，例如在 2026 年 6 月运行时为 `2026-05-31T23:45:00Z`。
- 不要为训练刷新拉取当前分钟的数据。
- 不要在训练中包含 `1m_live.csv` 叠加数据。
- 不要在验证集行上拟合部署模型。验证集仅用于早停和阈值选择。
- 除非有独立的步进式回测予以验证，否则不要运行新的实盘 Optuna 搜索。

生产环境的 15m 实盘模型使用经步进式验证的 Optuna 参数，存放于：

```bash
config/lgbm_15m_walk_forward_optuna_params.json
```

该文件从 `results/baseline_15m/optuna_best_params.json` 复制而来，与当前 15m 基线步进式运行一致：

- 目标周期：15 分钟
- 训练/验证/测试采样：15 分钟
- 特征集：`v1`，含 UTC 交易时段指标和现货收盘位置特征
- 平局策略（tie policy）：`expected`
- 研究得出的阈值 EMA alpha：`0.7`

实盘刷新可以在每月截止点重新拟合最新的 BTC/ETH 模型并重新选择验证阈值，但绝不能运行全新的 Optuna 搜索。已搜索出的超参数是经步进式验证的研究产物。仅基于实盘的新搜索会产生未经回测的参数，不应用于生产环境。

刷新最新的 5m 实盘模型：

```bash
.venv/bin/python live/train_live_model.py \
  --out-dir live/models_5m \
  --target-horizon-minutes 5 \
  --train-sample-minutes 5 \
  --validation-sample-minutes 5 \
  --threshold-update-weight 0.2 \
  --fixed-params-json results/baseline_5m/optuna_best_params.json \
  --feature-set v1
```

训练或刷新最新的 15m 实盘模型：

```bash
.venv/bin/python live/train_live_model.py
```

默认值为：

- `--out-dir live/models_15m`
- `--fixed-params-json config/lgbm_15m_walk_forward_optuna_params.json`
- `--target-horizon-minutes 15`
- `--train-sample-minutes 15`
- `--validation-sample-minutes 15`
- `--feature-set v1`

`live/train_live_model.py` 始终从 `--fixed-params-json` 读取经步进式验证的固定 Optuna 参数，仅在最新月度截止点重新拟合模型并重选验证阈值。它不会运行实盘 Optuna 搜索；如需重新搜索超参数，请回到独立的步进式研究流程。

## 实盘数据与信号管线

`live/update_live_klines.py` 是实盘 Binance 数据进程。它使用现货/期货 websocket K 线流，仅将已收盘的 1 分钟 K 线写入 `aligned_data_oos` 下按符号划分的 `1m_live.csv` 叠加文件。当传入 `--signal-file` 时，更新器会在写入新的或变更的已收盘 1 分钟行后立即运行 LightGBM 信号回调。`live/write_lightgbm_signals.py` 是单次信号工具加共享回调辅助函数；它不是实盘计时循环。不要把 `scripts/fetch_oos_klines.py` 当作实盘循环运行；它仅用于 REST 回补。

信号生成由数据回调驱动。回调将每根已收盘 1 分钟行映射到 `decision_time = open_time + 1 minute`；如果该决策时间是已配置的事件起点，且所有必需的 BTC/ETH 现货/期货叠加行都齐全，它就会将到期模型运行一次，并为该确切事件窗口写入信号。同一进程内的回调不会对同一周期/决策时间写入两次。

`live/write_lightgbm_signals.py` 支持多个 `--model-dir timeframe=path` 条目用于研究/调试。HiBT 执行端在 `live/hibt/`，前端控制面板只控制 HiBT 的启用开关和统一下单金额。

启动实盘信号生成：

```bash
.venv/bin/python live/run_signal_stack.py --signal-file live/signals.json
```

启动 HiBT 信号生成、执行和前端：

```bash
.venv/bin/python live/hibt/run_hibt_signal_stack.py
.venv/bin/python live/hibt/run_hibt_api_trader.py --timeframes 3m,5m,15m
.venv/bin/python live/hibt/control_panel.py --host 127.0.0.1 --port 8765
```
