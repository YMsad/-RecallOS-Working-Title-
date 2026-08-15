# -*- mode: python ; coding: utf-8 -*-
"""RecallOS 桌面版 PyInstaller 打包配置（Windows）。

构建：
    venv\\Scripts\\python.exe -m PyInstaller packaging\\RecallOS.spec

产物：dist\\RecallOS.exe（onefile、windowed）。
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # 仓库根目录（packaging/ 的上一级）

# Streamlit 没有内置 PyInstaller hook，必须手动收集其全部资源（前端 static 等）。
streamlit_datas, streamlit_binaries, streamlit_hidden = collect_all("streamlit")

# app.py 是作为数据文件打包的（由 launcher 在运行时启动），它的依赖不会被
# 静态分析到，因此显式收集 core 及第三方依赖。
core_hidden = collect_submodules("core")

hiddenimports = (
    list(streamlit_hidden)
    + list(core_hidden)
    + [
        "pandas",
        "httpx",
        "pydantic",
        "pydantic_settings",
        "dotenv",
    ]
)

datas = [(str(ROOT / "app.py"), ".")] + list(streamlit_datas)
binaries = list(streamlit_binaries)

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["IPython", "matplotlib", "notebook", "jupyter", "pytest"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="RecallOS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
