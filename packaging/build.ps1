# RecallOS 桌面版打包脚本（Windows）
# 用法：  .\packaging\build.ps1    （在仓库根目录运行）
# 产物：  dist\launcher.exe（onefile、windowed）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "venv\Scripts\python.exe"
$spec = Join-Path $root "launcher.spec"   # 根目录 launcher.spec 为唯一构建入口

if (-not (Test-Path $python)) {
    Write-Error "未找到 venv：$python，请先创建虚拟环境并安装 requirements.txt + pyinstaller"
}

if (-not (Test-Path (Join-Path $root "venv\Lib\site-packages\PyInstaller"))) {
    Write-Host "[RecallOS] 未安装 PyInstaller，正在安装 ..."
    & $python -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[RecallOS] 使用 PyInstaller 构建 ..."
& $python -m PyInstaller $spec --clean --noconfirm
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[RecallOS] 构建完成：dist\launcher.exe"