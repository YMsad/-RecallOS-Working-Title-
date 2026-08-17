# RecallOS V1.0 — API Key 安全方案（Cloudflare Workers）

零预算、自动化的 API Key 分发：Workers 动态密钥 + 用量监控告警 + 一键撤销。

**策略**：Worker 优先，手动 Key 兜底。

```text
用户电脑                          Cloudflare Workers（免费）          告警
┌──────────────┐   /get-key     ┌─────────────────────┐      ┌──────────┐
│ RecallOS     │ ──────────────▶│ ・设备指纹校验        │      │ Telegram │
│ ・设备指纹     │                │ ・KV 每日限额(20/天)│      │ 用量监控  │
│ ・缓存Key 24h │                │ ・返回临时Key+过期  │      │ 一键撤销  │
└──────────────┘                └─────────────────────┘      └──────────┘
```

## 文件清单

| 文件 | 用途 |
|------|------|
| `workers/get-key.js` | Cloudflare Worker：`/get-key`、`/revoke`、`/unrevoke`、`/health` |
| `wrangler.toml` | Workers 部署配置（KV 命名空间） |
| `core/device_fingerprint.py` | 设备指纹（跨平台，降级不抛错） |
| `core/key_manager.py` | 取临时 Key + 24h 缓存 + 用量日志 + 撤销 |
| `core/client.py` | 集成：Worker 优先，失败回退手动 Key |
| `scripts/monitor_usage.py` | 每日用量统计 + Telegram 告警 + 一键撤销 |

## 部署

```bash
# 1. 注册 https://dash.cloudflare.com/sign-up（免费）
npm install -g wrangler
wrangler login

# 2. 创建 KV 并回填 wrangler.toml
wrangler kv:namespace create "RECALLOS_KV"

# 3. 修改 workers/get-key.js 里的配置（至少 2 个真实 DeepSeek Key + 随机 ADMIN_SECRET）

# 4. 部署
wrangler deploy

# 5. 客户端 .env 增加
RECALLOS_WORKER_URL=https://<你的子域>.workers.dev
```

## 监控部署（每日一次）

```bash
# Windows 任务计划程序
schtasks /create /tn "RecallOS_Monitor" /tr "python C:\path\to\RecallOS\scripts\monitor_usage.py" /sc daily /st 20:00

# Linux / macOS cron
0 20 * * * cd /path/to/RecallOS && python scripts/monitor_usage.py >> /var/log/recallos_monitor.log 2>&1
```

环境变量：`TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`、`REVOKE_SECRET`、`RECALLOS_WORKER_URL`（阈值可选：`DAILY_TOKEN_THRESHOLD`/`DAILY_REQUEST_THRESHOLD`/`REQUEST_TOKEN_ESTIMATE`）。

## 验收清单

- [ ] `/health` 返回 `{"status":"ok"}`
- [ ] 客户端首次启动能从 Worker 取到 Key
- [ ] 同一设备 24h 内第二次启动走缓存，不重复请求
- [ ] 同一设备每日超 20 次返回 429
- [ ] 不同设备拿到不同 Key（都能用）
- [ ] `/revoke?secret=...` 后 `/get-key` 一律 403
- [ ] `/unrevoke?secret=...` 后恢复
- [ ] 监控脚本能统计今日用量，超标发 Telegram 告警
- [ ] 全程免费

## 应急手册

| 情况 | 操作 |
|------|------|
| 收到用量异常告警 | 1. 立即访问 `https://<worker>/revoke?secret=xxx` 2. 到 DeepSeek 控制台确认用量停止 3. 更换 Worker 里 Key 列表 4. `wrangler deploy` 重新部署 5. 确认后 `/unrevoke` 恢复服务 |
| 客户端离线 | 用缓存 Key（最长 24h）；缓存失效提示「无法连接 Key 服务器」，用户在设置页填自己的 Key 即可 |
| Key 被提取 | Key 绑定设备指纹，换电脑无效；24h 后过期 |
| 用户换电脑 | 新指纹 → 新 Key，旧 Key 24h 过期 |
| 克隆硬盘 | 两台机器共享同一指纹，共享每日 20 次限额（设计如此） |

## 常见问题

- **Workers 免费额度用完**：10 万请求/天 ≈ DAU 5000+，届时应已能覆盖付费升级。
- **为什么是「Worker 优先 + 手动兜底」**：兼顾零配置体验与离线/自备 Key 的用户。