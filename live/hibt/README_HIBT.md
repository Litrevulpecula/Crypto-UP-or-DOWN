# HiBT Web Runner

监听 `signals.json` 信号文件，自动打开 HiBT 事件合约页面，选择 BTC 或 ETH，匹配信号对应的 3 分钟或 5 分钟时间框，填入 3 USDT，根据信号方向执行买涨或买跌。每个信号仅执行一次。

## 目录结构

```
live/
├── run_hibt_live.py           # 主入口
├── hibt_browser.py            # 浏览器控制 (CDP/Launch 双模式)
├── hibt_fingerprint.py        # 反检测指纹注入 (16层)
├── hibt_config.py             # 配置数据类
├── hibt_signal_reader.py      # 信号文件读取
├── hibt_trader.py             # 交易逻辑 + 风控
├── write_lightgbm_signals.py  # 模型信号生成
├── hibt_config.example.json   # 本地配置示例
├── hibt_config.vps.json       # VPS 配置
├── deploy_to_vps.sh           # 一键部署到 VPS
├── sync_profile_to_vps.sh     # 同步登录态到 VPS
├── setup_chrome_vps.sh        # VPS 环境安装脚本
└── runtime/                   # 运行时数据 (git-ignored)
    ├── hibt-chrome-profile/   # Chrome 用户配置/cookie
    ├── hibt_state.json        # 交易状态
    └── hibt_orders.jsonl      # 交易日志
```

## 本地安装

```bash
cd ~/Crypto_UP_or_DOWN/live
pip install -r requirements_hibt.txt
```

不需要 `playwright install` — 程序使用 CDP 模式直接连接系统 Chrome，不依赖 Playwright 自带的 Chromium。

## 首次登录

```bash
python3 run_hibt_live.py --config hibt_config.example.json --login
```

会启动系统 Chrome 打开 hibt.com。手动完成登录后 Ctrl+C 退出。登录态保存在 `runtime/hibt-chrome-profile/` 中。

## 信号文件格式

由 `write_lightgbm_signals.py` 生成：

```bash
python3 write_lightgbm_signals.py --once   # 单次
python3 write_lightgbm_signals.py          # 持续更新
```

`signals.json` 示例：

```json
{
  "signal_id": "btc-20260605-123000",
  "symbol": "BTC-USDT",
  "timeframe": "5m",
  "signal": "BUY",
  "confidence": 0.73,
  "timestamp": "2026-06-05T12:30:00+08:00"
}
```

| 方向 | 字段取值 |
|------|---------|
| 看涨 | `BUY`, `UP`, `LONG`, `CALL` |
| 看跌 | `SELL`, `DOWN`, `SHORT`, `PUT` |
| 不交易 | `HOLD`, `WAIT`, `NONE` |

## 运行模式

```bash
# 模拟运行 (dry run) — 验证页面但不点击交易
python3 run_hibt_live.py --config hibt_config.example.json --once

# 实盘模式
python3 run_hibt_live.py --config hibt_config.example.json --live --confirm-order
```

| 参数 | 作用 |
|------|------|
| `--config` | 指定配置文件 |
| `--once` | 处理完当前信号后退出 |
| `--live` | 启用实盘下单（默认 dry_run） |
| `--confirm-order` | 自动确认二次弹窗 |
| `--login` | 仅登录，保持浏览器开启 |

## VPS 部署

### 第一步：本地登录

```bash
cd ~/Crypto_UP_or_DOWN/live
python3 run_hibt_live.py --config hibt_config.example.json --login
# Chrome 弹出后手动登录 hibt.com，完成后 Ctrl+C
```

### 第二步：首次部署（安装 Chrome + 环境）

```bash
./deploy_to_vps.sh --setup
```

会在 VPS 上安装 Google Chrome Stable、Xvfb、Python venv，并配置 systemd 服务。

### 第三步：同步登录态

```bash
./sync_profile_to_vps.sh
```

把本地 Chrome profile（含 cookies/登录会话）推送到 VPS。

### 第四步：启动服务

```bash
ssh root@47.79.32.65 'systemctl start hibt-trader'
ssh root@47.79.32.65 'journalctl -u hibt-trader -f'
```

### 日常更新

```bash
./deploy_to_vps.sh                                        # 推送代码
ssh root@47.79.32.65 'systemctl restart hibt-trader'      # 重启服务
```

### Session 过期

如果 VPS 上登录态失效（报 "not logged in"），在本地重新登录再同步：

```bash
python3 run_hibt_live.py --config hibt_config.example.json --login
# 重新登录后 Ctrl+C
./sync_profile_to_vps.sh
ssh root@47.79.32.65 'systemctl restart hibt-trader'
```

## 反检测机制

程序采用 CDP connect 模式 + 16 层指纹伪装，对抗常见反自动化检测：

### 架构层

| 对抗点 | 方案 |
|--------|------|
| Cloudflare Turnstile | CDP 模式：Chrome 作为独立进程启动，不经过 Playwright launch 管道，无 `--remote-debugging-pipe` 等自动化特征 |
| TLS 指纹 (JA3/JA4) | 使用系统安装的 Google Chrome（非 Playwright Chromium），TLS 握手与普通用户完全一致 |
| Playwright 痕迹 | 清除 `window.__playwright`、`__pw_manual`、`__PW_inspect` 全局变量 |

### JS 注入层 (init_script, 每个页面加载前执行)

| # | 防护项 | 说明 |
|---|--------|------|
| 1 | navigator.webdriver | delete + redefine → undefined |
| 2 | navigator 属性 | languages/platform/cores/memory/vendor/appVersion/maxTouchPoints |
| 3 | window.chrome | 补全 runtime/app/csi/loadTimes 对象 |
| 4 | Screen 属性 | width/height/colorDepth/availWidth/outerWidth 全一致 |
| 5 | Permissions.query | 修复 notifications 与 Notification.permission 不一致破绽 |
| 6 | WebGL | 伪装 vendor/renderer 为真实 GPU |
| 7 | Canvas 噪声 | toDataURL/toBlob/getImageData 加 seeded 微量像素扰动 |
| 8 | AudioContext 噪声 | getFloatFrequencyData/copyFromChannel 微量扰动 |
| 9 | WebRTC | 清空 iceServers + Chromium flag 双重阻止 IP 泄露 |
| 10 | navigator.connection | 补充 Network Information API |
| 11 | Plugins | 5 个真实 Chrome PDF 插件 |
| 12 | Iframe 递归 | createElement 拦截 |
| 13 | Function.toString | 所有被改写函数返回 `function name() { [native code] }` |
| 14 | Playwright 全局变量 | 删除 `__playwright` 等标识 |
| 15 | performance.now() | 0.05ms 级随机抖动，对抗时序指纹 |
| 16 | Notification.permission | 强制 'default'，防不一致检测 |

### 行为层

| 对抗点 | 方案 |
|--------|------|
| 鼠标轨迹 | 三阶贝塞尔曲线 (12-35个采样点)，非直线移动 |
| 键盘输入 | 逐字打字，55-165ms 随机延迟 |
| 点击位置 | 元素区域 25%-75% 随机取点，非中心 |
| 点击时长 | mousedown→mouseup 间 40-120ms 随机延迟 |
| 视口尺寸 | 每次启动 ±3px 随机抖动 |
| 操作间隔 | 0.25-1.1s 随机延迟 |

### 已知局限

| 风险 | 说明 | 缓解措施 |
|------|------|----------|
| Session 过期 | Chrome profile 的 cookie 会过期 | 本地重新 --login 后 sync_profile_to_vps.sh |
| IP 信誉突变 | 如果 VPS IP 被 Cloudflare 降级 | 换住宅代理出口 (配置 proxy_server) |
| Chrome 更新 | 大版本更新可能改变某些行为 | fingerprint.user_agent 手动更新版本号 |
| 长时间无交互 | 页面可能弹出"重新连接"提示 | trader 每次下单都重新 open_symbol，不保持长连接 |
