# weekly_lotto.ps1
# Weekly pipeline: collect -> analyze -> train -> generate -> buy
#
# Pipeline:
#   [1] collect   (inside run)   fetch latest draw results
#   [2] analyze                  update number statistics
#   [3] build-features           build ML feature matrix
#   [4] train-model              retrain model
#   [5] run                      generate games (purchased combos excluded)
#   [6] buy-lotto                auto-fill + purchase
#
# Register Task Scheduler (admin PowerShell):
#   schtasks /create /tn "MyLotto Weekly" `
#     /tr "powershell -NonInteractive -ExecutionPolicy Bypass -File D:\Projects\agent-pjt\my-agents\mylotto-agent\weekly_lotto.ps1" `
#     /sc weekly /d SAT /st 09:00 /f

# UTF-8 console output
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding            = [System.Text.Encoding]::UTF8

# Force Python UTF-8 output
$env:PYTHONUTF8       = "1"
$env:PYTHONIOENCODING = "utf-8"

$ProjectDir = "D:\Projects\agent-pjt\my-agents\mylotto-agent"
$PythonExe  = "$ProjectDir\.venv\Scripts\python.exe"
$LogDir     = "$ProjectDir\logs"
$LogFile    = "$LogDir\weekly_lotto.log"

Set-Location $ProjectDir

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line -ForegroundColor DarkGray
    Add-Content $LogFile $line -Encoding UTF8
}

function Step($n, $total, $label) {
    Write-Host "`n[$n/$total] $label" -ForegroundColor Cyan
}

function Die($msg) {
    Write-Host "`n[ERROR] $msg" -ForegroundColor Red
    Log "FAIL: $msg"
    exit 1
}

Log "=== Weekly lotto pipeline started ==="

# ─────────────────────────────────────────────────────────────
# Step 1: Analyze (update number statistics)
# ─────────────────────────────────────────────────────────────
Step 1 5 "Analyze number statistics"
& $PythonExe main.py analyze
if ($LASTEXITCODE -ne 0) { Die "analyze failed" }
Log "analyze done"

# ─────────────────────────────────────────────────────────────
# Step 2: Build ML features
# ─────────────────────────────────────────────────────────────
Step 2 5 "Build ML features (build-features)"
& $PythonExe main.py build-features
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] build-features failed -- continuing without ML strategy." -ForegroundColor Yellow
    Log "WARN: build-features failed (skipped)"
}
else {
    Log "build-features done"

    # ─────────────────────────────────────────────────────────
    # Step 3: Retrain model
    # ─────────────────────────────────────────────────────────
    Step 3 5 "Retrain model (train-model)"
    & $PythonExe main.py train-model
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] train-model failed -- continuing without ML strategy." -ForegroundColor Yellow
        Log "WARN: train-model failed (skipped)"
    }
    else {
        Log "train-model done"
    }
}

# ─────────────────────────────────────────────────────────────
# Step 4: Generate 5 games (balanced_v2 x3 + random x2)
#   collect runs automatically inside 'run'
#   purchased combos are automatically excluded
# ─────────────────────────────────────────────────────────────
Step 4 5 "Generate 5 games"
& $PythonExe main.py run --balanced-v2-games 3 --random-games 2 --balanced-games 0
if ($LASTEXITCODE -ne 0) { Die "game generation failed" }
Log "game generation done (5 games)"

# ─────────────────────────────────────────────────────────────
# Step 5: Auto-purchase (5 games)
# ─────────────────────────────────────────────────────────────
Step 5 5 "Auto-purchase (5 games)"
& $PythonExe main.py buy-lotto --max-games 5
Log "buy-lotto done"

Log "=== Weekly lotto pipeline finished ==="
Write-Host "`n[DONE] Pipeline complete" -ForegroundColor Green
