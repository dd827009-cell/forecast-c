# install_windows.ps1 — 新 Windows 電腦第一步：裝 WSL2 + Ubuntu。
# 用「系統管理員」開 PowerShell，執行：
#   powershell -ExecutionPolicy Bypass -File pipeline\setup\install_windows.ps1
# 裝完會要求「重開機一次」，重開後首次啟動 Ubuntu 會請你設一組 Linux 帳號/密碼，
# 接著跑第二支腳本 pipeline/setup/setup_wsl.sh（在 Ubuntu 裡）。

$ErrorActionPreference = "Stop"

# 1) 必須系統管理員
$admin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host "✗ 請以『系統管理員』身分重開 PowerShell 再執行本腳本。" -ForegroundColor Red
    exit 1
}

# 2) Windows 版本檢查（wsl --install 需 Win10 2004/組建19041+ 或 Win11）
$build = [int](Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuildNumber
Write-Host "Windows 組建: $build"
if ($build -lt 19041) {
    Write-Host "✗ 這台 Windows 太舊（需組建 19041+）。請先 Windows Update。" -ForegroundColor Red
    exit 1
}

# 3) 若已裝過 WSL 發行版就不重裝
$existing = ""
try { $existing = (wsl.exe -l -q) 2>$null } catch {}
if ($existing -match "Ubuntu") {
    Write-Host "✓ 偵測到已安裝 Ubuntu，跳過安裝。" -ForegroundColor Green
    Write-Host "  直接開 Ubuntu 跑第二步： bash <repo>/pipeline/setup/setup_wsl.sh"
    exit 0
}

# 4) 安裝 WSL2 + Ubuntu（會自動開啟所需 Windows 功能 + 下載 Linux 核心 + 預設 Ubuntu）
Write-Host "→ 安裝 WSL2 + Ubuntu（可能要幾分鐘、會下載）..." -ForegroundColor Cyan
wsl.exe --install -d Ubuntu
wsl.exe --set-default-version 2 2>$null

Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host " ✓ WSL2 + Ubuntu 安裝指令已送出。" -ForegroundColor Green
Write-Host " 接下來："
Write-Host "  1) 【重新開機】這台電腦。"
Write-Host "  2) 重開後從『開始』開 Ubuntu → 首次啟動請設一組 Linux 使用者名稱/密碼。"
Write-Host "  3) 在 Ubuntu 視窗把專案抓下來（或複製過來）："
Write-Host "       git clone https://github.com/dd827009-cell/forecast-c.git"
Write-Host "     （兩份 Excel 不在 git 裡，記得另外複製到這台。）"
Write-Host "  4) 跑第二支腳本裝工具 + 設密碼："
Write-Host "       bash forecast-c/pipeline/setup/setup_wsl.sh"
Write-Host "============================================================" -ForegroundColor Yellow
