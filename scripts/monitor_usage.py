"""用量监控 —— 每日检查 API 用量，超标则发 Telegram 告警，支持一键撤销（V1.0）。

数据来源：本机 ~/.recallos/usage.log（由 core.key_manager.KeyManager 在每次
成功取 Key 时追加一行 JSON）。正常一台设备每天只会取 1 次 Key，所以
「请求数」近似于「活跃设备取 Key 次数」，Token 数是估算值。

部署方式（每天定时跑一次）：
  Windows:  schtasks /create /tn "RecallOS_Monitor" /tr "python <本文件>" /sc daily /st 20:00
  macOS:    crontab 加一行  0 20 * * * python3 <本文件>
  Linux:    crontab 加一行  0 20 * * * cd <项目> && python scripts/monitor_usage.py

环境变量：
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID   Telegram 告警（@BotFather 申请）
  REVOKE_SECRET                           管理员撤销密钥
  RECALLOS_WORKER_URL                     Worker 地址（撤销要用）
  DAILY_TOKEN_THRESHOLD                   每日 Token 阈值（默认 500000）
  DAILY_REQUEST_THRESHOLD                 每日请求次数阈值（默认 50）
  REQUEST_TOKEN_ESTIMATE                  每次请求估算 Token（默认 5000）
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_USAGE_LOG = Path.home() / ".recallos" / "usage.log"

DAILY_TOKEN_THRESHOLD = int(os.environ.get("DAILY_TOKEN_THRESHOLD", "500000"))
DAILY_REQUEST_THRESHOLD = int(os.environ.get("DAILY_REQUEST_THRESHOLD", "50"))
REQUEST_TOKEN_ESTIMATE = int(os.environ.get("REQUEST_TOKEN_ESTIMATE", "5000"))

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
REVOKE_SECRET = os.environ.get("REVOKE_SECRET", "")
WORKER_URL = os.environ.get("RECALLOS_WORKER_URL", "")


def check_usage(
    log_file: Path = DEFAULT_USAGE_LOG,
    *,
    today: str | None = None,
    token_threshold: int = DAILY_TOKEN_THRESHOLD,
    request_threshold: int = DAILY_REQUEST_THRESHOLD,
    token_per_request: int = REQUEST_TOKEN_ESTIMATE,
) -> dict[str, Any]:
    """统计今日用量，返回是否超标的判定结果。"""
    today = today or date.today().isoformat()
    if not log_file.exists():
        return {
            "status": "no_log",
            "date": today,
            "total_tokens": 0,
            "total_requests": 0,
            "unique_devices": 0,
            "is_over_threshold": False,
        }

    total_requests = 0
    devices: set[str] = set()
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_date = str(entry.get("timestamp", ""))[:10]
                if entry_date != today:
                    continue
                total_requests += 1
                if "device_id" in entry:
                    devices.add(str(entry["device_id"]))
    except OSError as exc:
        return {"status": "error", "message": str(exc)}

    total_tokens = total_requests * token_per_request
    return {
        "status": "ok",
        "date": today,
        "total_tokens": total_tokens,
        "total_requests": total_requests,
        "unique_devices": len(devices),
        "is_over_threshold": (
            total_tokens > token_threshold or total_requests > request_threshold
        ),
    }


def send_telegram_alert(
    message: str,
    *,
    bot_token: str | None = None,
    chat_id: str | None = None,
    timeout: float = 10.0,
) -> bool:
    """发送 Telegram 告警；未配置或不成功返回 False。"""
    bot_token = bot_token if bot_token is not None else TELEGRAM_BOT_TOKEN
    chat_id = chat_id if chat_id is not None else TELEGRAM_CHAT_ID
    if not bot_token or not chat_id:
        print("⚠️ Telegram 未配置，跳过告警")
        return False
    try:
        response = httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": f"🚨 *RecallOS 用量告警*\n\n{message}",
                "parse_mode": "Markdown",
            },
            timeout=timeout,
        )
        return response.status_code == 200
    except Exception as exc:  # noqa: BLE001 —— 告警失败不应让脚本崩溃
        print(f"❌ 发送 Telegram 告警失败：{exc}")
        return False


def auto_revoke(
    *,
    secret: str | None = None,
    worker_url: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """一键撤销：让 Worker 拒绝一切新的 Key 请求。"""
    secret = secret if secret is not None else REVOKE_SECRET
    worker_url = worker_url if worker_url is not None else WORKER_URL
    if not secret or not worker_url:
        return {"error": "未配置 REVOKE_SECRET / RECALLOS_WORKER_URL"}
    try:
        response = httpx.get(
            f"{worker_url.rstrip('/')}/revoke",
            params={"secret": secret},
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"撤销失败（HTTP {response.status_code}）"}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main() -> int:
    print(f"[{datetime.now()}] 开始检查用量…")
    usage = check_usage()

    if usage.get("status") == "error":
        print(f"❌ 检查失败：{usage.get('message')}")
        send_telegram_alert(f"❌ 用量检查失败\n```\n{usage.get('message')}\n```")
        return 1

    print(
        f"📊 今日用量：{usage['total_tokens']:,} token，"
        f"{usage['total_requests']} 次请求，活跃设备 {usage['unique_devices']} 台"
    )

    if not usage.get("is_over_threshold", False):
        print("✅ 用量正常")
        return 0

    message = (
        "📊 *今日用量异常*\n\n"
        f"📅 日期：{usage['date']}\n"
        f"🔢 Token：{usage['total_tokens']:,}（阈值：{DAILY_TOKEN_THRESHOLD:,}）\n"
        f"📞 请求数：{usage['total_requests']}（阈值：{DAILY_REQUEST_THRESHOLD}）\n"
        f"📱 活跃设备：{usage['unique_devices']} 台\n\n"
        "⚠️ 超过阈值，建议立即撤销 Key：\n"
        f"`{WORKER_URL}/revoke?secret=REVOKE_SECRET`"
    )
    print("🚨 超标！发送告警…")
    send_telegram_alert(message)

    # 需要自动撤销时取消下面注释（慎用，会中断所有用户的 Key 分发）
    # result = auto_revoke()
    # print("🔒 自动撤销结果：", result)
    # if result.get("success"):
    #     send_telegram_alert("🔒 已自动撤销所有 Key")
    return 0


if __name__ == "__main__":
    sys.exit(main())
