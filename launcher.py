"""RecallOS 桌面版启动器 —— PyInstaller 打包入口。

应用本体仍是 Streamlit Web app；打包成单个 exe 后，本文件负责：
1. 通过 ``streamlit.web.bootstrap.run`` 以 Server 模式启动 app.py；
2. 自动打开浏览器访问本地页面；
3. windowed（无控制台）模式下把 stdout/stderr 重定向到日志文件，避免崩溃。
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import streamlit.web.bootstrap as bootstrap


def _bundled_root() -> Path:
    """打包后数据/资源所在目录（onefile 为解压的临时目录 _MEIPASS）。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _redirect_std_streams() -> None:
    """windowed 模式下 sys.stdout/stderr 为 None，重定向到 ~/.recallos/recallos.log。"""
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_dir = Path.home() / ".recallos"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "recallos.log"
    stream = log_file.open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def _free_port() -> int:
    """拿一个空闲端口，避免与已有 Streamlit 实例冲突。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    _redirect_std_streams()
    app_path = _bundled_root() / "app.py"

    flag_options = {
        # 打包后 streamlit 不在 site-packages，developmentMode 会被判定为 True，
        # 导致 server.port 冲突且前端地址指向 dev server；显式关闭。
        "global.developmentMode": False,
        # 本地桌面应用：仅监听本机，自动打开浏览器（headless=False）
        "server.address": "127.0.0.1",
        "server.port": _free_port(),
        "server.headless": False,
        "browser.gatherUsageStats": False,
        # 与 .streamlit/config.toml 保持一致的主题
        "theme.base": "light",
        "theme.primaryColor": "#D4875E",
        "theme.backgroundColor": "#FDFBF7",
        "theme.secondaryBackgroundColor": "#EFEBE4",
        "theme.textColor": "#1A1A1A",
    }
    # bootstrap.run 本身不应用 flag_options，需先加载进 config（同 streamlit run CLI 的行为）。
    bootstrap.load_config_options(flag_options=flag_options)

    bootstrap.run(
        str(app_path),
        is_hello=False,
        args=["streamlit", "run", str(app_path)],
        flag_options=flag_options,
    )


if __name__ == "__main__":
    main()