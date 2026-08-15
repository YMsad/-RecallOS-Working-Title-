# RecallOS 桌面版打包脚本（Windows）
# 用法：  .\packaging\build.ps1    （在仓库根目录运行）
# 产物：  dist\RecallOS.exe（onefile、windowed）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "venv\Scripts\python.exe"
$spec = Join-Path $PSScriptRoot "RecallOS.spec"

if (-not (Test-Path $python)) {
    Write-Error "未找到 venv：$python，请先创建虚拟环境并安装 requirements.txt"
}

Write-Host "[RecallOS] 使用 PyInstaller 构建 ..."
& $python -m PyInstaller $spec --clean --noconfirm
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[RecallOS] 构建完成：dist\RecallOS.exe"