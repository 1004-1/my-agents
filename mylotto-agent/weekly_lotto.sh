#!/bin/bash
# weekly_lotto.sh
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
# Setup:
#   chmod +x weekly_lotto.sh
#   pip install playwright && playwright install chromium
#   playwright install-deps chromium      # install system dependencies
#
# .env 설정 (Linux headless 서버):
#   HEADLESS=true                         # 디스플레이 없는 서버
#   HEADLESS=false                        # X11/Wayland 데스크탑
#
# Register as cron job (every Saturday 09:00):
#   crontab -e
#   0 9 * * 6 /bin/bash /path/to/mylotto-agent/weekly_lotto.sh >> /path/to/mylotto-agent/logs/cron.log 2>&1

set -uo pipefail

# ── 경로 설정 ─────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/weekly_lotto.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# ── 환경변수 ──────────────────────────────────────────────────
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# .env 파일 로드 (HEADLESS 등 설정 반영)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$PROJECT_DIR/.env"
    set +a
fi

# HEADLESS 미설정 시 Linux 기본값은 true (디스플레이 없는 환경 대비)
export HEADLESS="${HEADLESS:-true}"

# ── 유틸 함수 ─────────────────────────────────────────────────
log() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $1" | tee -a "$LOG_FILE"
}

step() {
    echo ""
    echo "[$1/$2] $3"
}

die() {
    echo ""
    echo "[ERROR] $1" >&2
    log "FAIL: $1"
    exit 1
}

# ── Python 및 venv 확인 ───────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    die "Python not found: $PYTHON  (run: python -m venv .venv && .venv/bin/pip install -r requirements.txt)"
fi

log "=== Weekly lotto pipeline started ==="

# ─────────────────────────────────────────────────────────────
# Step 1: Analyze (update number statistics)
# ─────────────────────────────────────────────────────────────
step 1 5 "Analyze number statistics"
"$PYTHON" main.py analyze || die "analyze failed"
log "analyze done"

# ─────────────────────────────────────────────────────────────
# Step 2: Build ML features
# ─────────────────────────────────────────────────────────────
step 2 5 "Build ML features (build-features)"
if ! "$PYTHON" main.py build-features; then
    echo "  [WARN] build-features failed -- continuing without ML strategy."
    log "WARN: build-features failed (skipped)"
else
    log "build-features done"

    # ─────────────────────────────────────────────────────────
    # Step 3: Retrain model
    # ─────────────────────────────────────────────────────────
    step 3 5 "Retrain model (train-model)"
    if ! "$PYTHON" main.py train-model; then
        echo "  [WARN] train-model failed -- continuing without ML strategy."
        log "WARN: train-model failed (skipped)"
    else
        log "train-model done"
    fi
fi

# ─────────────────────────────────────────────────────────────
# Step 4: Generate 5 games (balanced_v2 x3 + random x2)
#   collect runs automatically inside 'run'
#   purchased combos are automatically excluded
# ─────────────────────────────────────────────────────────────
step 4 5 "Generate 5 games"
"$PYTHON" main.py run --balanced-v2-games 3 --random-games 2 --balanced-games 0 || die "game generation failed"
log "game generation done (5 games)"

# ─────────────────────────────────────────────────────────────
# Step 5: Auto-purchase (1 game at a time, 5 rounds)
# ─────────────────────────────────────────────────────────────
step 5 5 "Auto-purchase (1 game x 5 rounds)"
"$PYTHON" main.py buy-lotto --max-games 5
log "buy-lotto done"

log "=== Weekly lotto pipeline finished ==="
echo ""
echo "[DONE] Pipeline complete"
