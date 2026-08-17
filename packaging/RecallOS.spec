# -*- mode: python ; coding: utf-8 -*-
"""RecallOS 桌面版 PyInstaller 打包配置（Windows，--onefile）。

构建（在项目根目录运行）：
    venv\\Scripts\\python.exe -m PyInstaller packaging\\RecallOS.spec --clean --noconfirm

产物：dist\\RecallOS.exe（onefile + windowed）。

说明：
- 与根目录 launcher.spec 逻辑完全一致，仅产物名不同（RecallOS.exe vs launcher.exe）。
- 本 spec 的结构（EXE 直接带上 a.binaries / a.datas，且无 COLLECT）即
  PyInstaller 的 --onefile 模式，构建时无需再加 --onefile。
- 修复过的两个经典坑：
  1) importlib.metadata.PackageNotFoundError: No package metadata was
     found for streamlit —— streamlit 启动时会用 importlib.metadata 查自己
     的版本，必须把 *.dist-info 元数据一并打进去（copy_metadata）。
  2) app.py 是运行时才被 launcher 启动的数据文件，core 等模块不会被
     静态分析到，需 collect_submodules("core") 显式收集。
"""
from pathlib import Path

from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import (
    collect_all,
    collect_submodules,
    copy_metadata,
)

ROOT = Path(SPECPATH).resolve().parent  # 仓库根目录（packaging/ 的上一级）

# ---------------------------------------------------------------------------
# 1) Streamlit：官方没有内置 PyInstaller hook，必须手动收集。
#    collect_all 一次性拿到它的数据（前端 static/模板）、二进制与纯模块，
#    否则打包出来的 exe 缺前端资源，Streamlit 服务起不来。
# ---------------------------------------------------------------------------
streamlit_datas, streamlit_binaries, streamlit_hidden = collect_all("streamlit")

# 关键修复：把运行时需要查版本的 dist-info 元数据一起打进包。
# 只有 streamlit 会触发上面那个报错；其余包一并带上以防同类问题。
_METADATA_PACKAGES = [
    "streamlit",
    "pydantic",
    "pydantic-settings",  # 发行名（distribution name）带连字符
    "python-dotenv",
    "httpx",
    "pandas",
    "numpy",
    "tornado",
    "click",
    "jinja2",
    "blinker",
    "altair",
]
metadata_datas = []
for _pkg in _METADATA_PACKAGES:
    try:
        metadata_datas += copy_metadata(_pkg)
    except Exception as _exc:  # noqa: BLE001 —— 缺某个 dist-info 不应中断构建
        print(f"[RecallOS.spec] 未找到元数据，跳过：{_pkg}（{_exc}）")

# ---------------------------------------------------------------------------
# 2) core：launcher.py 不直接 import 它，是运行时的 app.py 才 import，
#    静态分析收集不到；必须显式收集所有子模块。
# ---------------------------------------------------------------------------
core_hidden = collect_submodules("core")

hiddenimports = (
    list(streamlit_hidden)
    + list(core_hidden)
    + [
        "pandas",
        "numpy",
        "httpx",
        "pydantic",
        "pydantic_settings",
        "dotenv",
    ]
)

# ---------------------------------------------------------------------------
# 3) 数据文件：app.py（运行时入口脚本）、core 文件夹、.env
# ---------------------------------------------------------------------------
datas = [
    (str(ROOT / "app.py"), "."),
    (str(ROOT / ".env"), "."),
]
datas += metadata_datas
datas += list(streamlit_datas)
# core 整文件夹随包分发（排除 __pycache__）；模块导入仍走上面收集的子模块。
# Tree 产出 TOC 三元组，需转回 (源, 目标) 二元组再并入 datas。
_core_tree = Tree(str(ROOT / "core"), prefix="core", excludes=["__pycache__"])
datas += [(src, dest) for dest, src, _typecode in _core_tree]

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
    console=False,  # windowed：launcher.py 已把 stdout/stderr 重定向到日志
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
