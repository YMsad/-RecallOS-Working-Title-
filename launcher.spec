# -*- mode: python ; coding: utf-8 -*-
r"""RecallOS launcher.exe PyInstaller spec (Windows, --onefile).

Build (run in the repo root):
    venv\Scripts\python.exe -m PyInstaller launcher.spec --clean --noconfirm

Output: dist\launcher.exe (onefile + console).

Why this spec is "complete" (all the classic pitfalls fixed):
1. importlib.metadata.PackageNotFoundError: No package metadata was found
   for streamlit — streamlit reads its own version via importlib.metadata at
   import time (streamlit/version.py). The dist-info metadata must be bundled
   with copy_metadata or the launcher crashes before it even starts.
2. Streamlit has no official PyInstaller hook: collect_all("streamlit") is
   required to ship its frontend static/ assets and binary wheels, otherwise
   the server starts but serves a blank/broken UI.
3. app.py is a *data* file launched at runtime by launcher.py — static
   analysis never sees it, so it must be listed in `datas` explicitly, or
   launcher.py raises FileNotFoundError at startup.
4. core/ is imported at runtime by app.py, not by launcher.py — collect
   submodules explicitly AND ship the folder as data so `import core.*`
   works from the extracted bundle.
5. .env is bundled next to app.py; core/config.py resolves it relative to
   its own file, so it is found inside the onefile temp dir. If .env is
   missing at build time, fall back to .env.example (with a warning).
"""
from pathlib import Path

from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import (
    collect_all,
    collect_submodules,
    copy_metadata,
)

ROOT = Path(SPECPATH).resolve()  # repo root (launcher.spec lives in it)

# ---------------------------------------------------------------------------
# 1) Streamlit: no official PyInstaller hook - collect data/bins/modules.
# ---------------------------------------------------------------------------
streamlit_datas, streamlit_binaries, streamlit_hidden = collect_all("streamlit")

# Runtime version-lookup metadata (the critical PackageNotFoundError fix).
_METADATA_PACKAGES = [
    "streamlit",
    "pydantic",
    "pydantic-settings",  # distribution name contains a hyphen
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
    except Exception as _exc:  # noqa: BLE001 - a missing dist-info must not abort the build
        print(f"[launcher.spec] metadata not found, skipping: {_pkg} ({_exc})")

# ---------------------------------------------------------------------------
# 2) core: imported at runtime by app.py - collect submodules explicitly.
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
# 3) Data files: app.py (runtime entry), .env, core/ folder.
# ---------------------------------------------------------------------------
datas = [
    (str(ROOT / "app.py"), "."),
]
_env_src = ROOT / ".env"
if _env_src.exists():
    datas.append((str(_env_src), "."))
else:
    print("[launcher.spec] WARNING: .env not found - bundling .env.example instead")
    datas.append((str(ROOT / ".env.example"), ".env"))
datas += metadata_datas
datas += list(streamlit_datas)
# Ship the core/ folder as data too (imports also work through the collected
# submodules; the folder covers any non-.py resource files).
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
    name='launcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # launcher.py prints startup progress + error prompts to the console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
