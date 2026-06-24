# HiBT API Runner

HiBT 实盘路径只保留 API 执行：本地模型写 `signals.json`，API trader 读取信号并调用 HiBT web order endpoint。

## 当前链路

1. 本地 Python 进程订阅 Binance 1m K 线。
2. 到决策点后，用配置的 `live/models_*` 生成 BTC/ETH 信号。
3. 信号写入 `live/hibt/signals.json`。
4. `run_hibt_api_trader.py` 读取信号，通过 HiBT web API 下单。

API trader 只读取 `lightgbm_live_signal_writer` 写出的 `signals` 列表，不兼容旧格式，不做每日订单上限、cooldown、仓位管理或额外策略过滤。唯一持久状态是尝试提交后的 `signal_id` 幂等，避免同一信号重复提交。

## 目录

```text
live/hibt/
├── run_hibt_signal_stack.py    # 本地 kline + LightGBM 信号生成
├── run_hibt_api_trader.py      # 读取 signals.json 并通过 HiBT web API 下单
├── hibt_api_client.py          # HiBT web API order client
├── signals.json                # 实盘信号输出，启动时生成
└── signals.sample.json         # 信号格式示例
```

## 启动信号生成

从仓库根目录运行：

```bash
.venv/bin/python live/hibt/run_hibt_signal_stack.py
```

默认跑 3m/5m/15m 三套模型：

```text
--signal-model-dir 3m=live/models_3m
--signal-model-dir 5m=live/models_5m
--signal-model-dir 15m=live/models_15m
```

指定周期示例：

```bash
.venv/bin/python live/hibt/run_hibt_signal_stack.py \
  --signal-model-dir 15m=live/models_15m
```

启动时会清空旧 `live/hibt/signals.json`，避免读到上一轮残留信号。

## API 执行

API trader 默认 dry-run，不会下单：

```bash
.venv/bin/python live/hibt/run_hibt_api_trader.py --timeframes 3m,5m,15m
```

前端控制面板：

```bash
.venv/bin/python live/control_panel.py --venue hibt --host 127.0.0.1 --port 8765
```

前端写入 `live/hibt/runtime/hibt_control.json`，只包含 HiBT 启用开关和统一下单金额。

实盘前从浏览器网络请求里取 HiBT 的 `v`、`Authorization`、`x-auth-token`，只放到环境变量：

```bash
export HIBT_API_V='...'
export HIBT_AUTHORIZATION='...'
export HIBT_X_AUTH_TOKEN='...'
```

实盘执行：

```bash
.venv/bin/python live/hibt/run_hibt_api_trader.py \
  --live \
  --timeframes 3m,5m,15m
```

`direction` 默认使用 `up=1`、`down=-1`，对应当前 HiBT 事件合约页面的请求参数。如果实测 HiBT 映射不同，用：

```bash
--direction-up 1 --direction-down -1
```
