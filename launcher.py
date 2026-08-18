"""RecallOS desktop launcher - PyInstaller entry point.

The app itself is still a Streamlit web app; when packaged into a single EXE
this file is responsible for:
1. Starting app.py via ``streamlit.web.bootstrap.run`` (server mode);
2. Auto-opening the browser on the local page;
3. windowed (no console) mode: redirecting stdout/stderr to a log file;
4. Logging every startup step and any fatal error to
   ``~/.recallos/recallos.log`` / ``~/.recallos/launcher_error.log`` so a
   packaged EXE never dies silently.
"""

from __future__ import annotations

import socket
import sys
import traceback
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path.home() / ".recallos"
_LOG_FILE = _LOG_DIR / "recallos.log"
_ERROR_LOG_FILE = _LOG_DIR / "launcher_error.log"


def _log(message: str) -> None:
    """Write one startup/progress line to the log file and the console.

    Never raises: logging must not be able to crash the launcher.
    """
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    try:
        print(line, flush=True)
    except Exception:  # noqa: BLE001 - stream may be None in windowed mode
        pass


def _write_error_log(exc: BaseException) -> None:
    """Dump the full traceback to ~/.recallos/launcher_error.log."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _ERROR_LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write("=" * 30 + f" {datetime.now().isoformat(timespec='seconds')} " + "=" * 30 + "\n")
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=fh)
            fh.write("\n")
    except OSError:
        pass


def _redirect_std_streams() -> None:
    """windowed mode: sys.stdout/stderr are None -> redirect to the log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    stream = _LOG_FILE.open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def _bundled_root() -> Path:
    """Directory of bundled data/resources (the extracted temp dir in onefile)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _free_port() -> int:
    """Grab a free port to avoid clashing with an already-running instance."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run() -> None:
    _redirect_std_streams()
    root = _bundled_root()
    app_path = root / "app.py"
    _log(f"launcher start: frozen={getattr(sys, 'frozen', False)} root={root}")
    if not app_path.exists():
        raise FileNotFoundError(
            f"app.py is missing in the bundle: {app_path}. "
            "The EXE was built without app.py - check that the spec file's "
            "`datas` includes ('app.py', '.')."
        )

    # Import streamlit late so that any failure (e.g. missing dist-info
    # metadata in the bundle) is caught by main()'s handler and logged,
    # instead of killing the process before our logging is installed.
    import streamlit.web.bootstrap as bootstrap  # noqa: PLC0415

    port = _free_port()
    _log(f"streamlit imported OK; starting server on 127.0.0.1:{port} app={app_path}")

    flag_options = {
        # After packaging, streamlit is not in site-packages, so
        # developmentMode would be detected as True -> port conflicts and a
        # frontend pointing at the dev server; disable it explicitly.
        "global.developmentMode": False,
        # Local desktop app: listen on localhost only, auto-open the browser.
        "server.address": "127.0.0.1",
        "server.port": port,
        "server.headless": False,
        "browser.gatherUsageStats": False,
        # Same theme as .streamlit/config.toml.
        "theme.base": "light",
        "theme.primaryColor": "#D4875E",
        "theme.backgroundColor": "#FDFBF7",
        "theme.secondaryBackgroundColor": "#EFEBE4",
        "theme.textColor": "#1A1A1A",
    }
    # bootstrap.run does not apply flag_options by itself; load them into the
    # config first (same behavior as the `streamlit run` CLI).
    bootstrap.load_config_options(flag_options=flag_options)

    bootstrap.run(
        str(app_path),
        is_hello=False,
        args=["streamlit", "run", str(app_path)],
        flag_options=flag_options,
    )


def main() -> None:
    try:
        _run()
    except SystemExit:
        # streamlit/bootstrap exit the process normally; keep that behavior.
        raise
    except BaseException as exc:  # noqa: BLE001 - capture everything
        _log(f"launcher crashed: {type(exc).__name__}: {exc}")
        _write_error_log(exc)
        traceback.print_exc()
        # In console mode, keep the window open so the user can read the error;
        # in windowed mode stdin is None and we just exit with a non-zero code.
        try:
            if sys.stdin is not None:
                input("\n[RecallOS] Startup failed. Check ~/.recallos/launcher_error.log. Press Enter to close...")
        except Exception:  # noqa: BLE001 - never hang on a broken stdin
            pass
        raise


if __name__ == "__main__":
    main()
