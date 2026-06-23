#!/bin/bash
# weekly_lotto.sh
# Weekly pipeline: collect -> analyze -> train -> generate -> buy -> check-results
#
# Pipeline:
#   [1] analyze          update number statistics  (collect is inside 'run')
#   [2] build-features   build ML feature matrix
#   [3] train-model      retrain model
#   [4] run              collect + generate games (purchased combos excluded)
#   [5] buy-lotto        auto-purchase (1 game x 5 rounds)
#   [6] check-results    compare vs actual draw results (accumulated feedback)
#
# ── macOS (Mac Mini) 최초 설정 ──────────────────────────────────
#   python3.12 -m venv .venv
#   .venv/bin/pip install -r requirements.txt
#   .venv/bin/playwright install chromium
#   cp .env.example .env   # 그 후 .env에 DH_LOGIN_ID/PW 입력
#
# ── Amazon Linux 2023 최초 설정 ─────────────────────────────────
#   bash scripts/setup_amazon_linux.sh
#
# ── .env 필수 설정 ──────────────────────────────────────────────
#   DH_LOGIN_ID=<동행복권 아이디>        # 자동 로그인에 필수
#   DH_LOGIN_PW=<동행복권 비밀번호>      # 자동 로그인에 필수
#   HEADLESS=false                       # macOS: 브라우저 창 표시 (기본)
#   HEADLESS=true                        # 서버/무인 환경
#
# ── macOS cron 등록 (매주 토요일 09:00 KST) ──────────────────
#   crontab -e
#   0 9 * * 6 /bin/bash /Users/soft1003/MyAgents/agents-in-github/mylotto-agent/weekly_lotto.sh >> /Users/soft1003/MyAgents/agents-in-github/mylotto-agent/logs/cron.log 2>&1
#
# ── (선택) macOS launchd 등록 ──────────────────────────────────
#   scripts/setup_macos_launchd.sh

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

# .env 파일 로드 (HEADLESS, DH_LOGIN_ID, DH_LOGIN_PW 등 반영)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$PROJECT_DIR/.env"
    set +a
fi

# HEADLESS 기본값: macOS는 false(브라우저 창 표시), 그 외는 true
if [ "$(uname)" = "Darwin" ]; then
    export HEADLESS="${HEADLESS:-false}"
else
    export HEADLESS="${HEADLESS:-true}"
fi

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
    die "Python not found: $PYTHON
  최초 설정이 필요합니다:
    bash scripts/setup_amazon_linux.sh     # Amazon Linux 2023
    python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt  # 기타"
fi

# ── 헤드리스 서버에서 자동 로그인 설정 경고 ──────────────────
if [ "${HEADLESS}" = "true" ]; then
    if [ -z "${DH_LOGIN_ID:-}" ] || [ -z "${DH_LOGIN_PW:-}" ]; then
        echo ""
        echo "[WARN] HEADLESS=true 이지만 DH_LOGIN_ID / DH_LOGIN_PW 가 설정되지 않았습니다."
        echo "       buy-lotto 단계에서 로그인 대기 타임아웃이 발생합니다."
        echo "       .env 파일에 DH_LOGIN_ID, DH_LOGIN_PW 를 추가하세요."
        log "WARN: DH_LOGIN_ID or DH_LOGIN_PW not set (headless mode)"
    fi
fi

log "=== Weekly lotto pipeline started (HEADLESS=${HEADLESS}) ==="

# ─────────────────────────────────────────────────────────────
# Step 1: Analyze (update number statistics)
#   당첨번호 수집은 Step 4의 'run' 내부에서 자동 실행됨
# ─────────────────────────────────────────────────────────────
step 1 6 "Analyze number statistics"
"$PYTHON" main.py analyze || die "analyze failed"
log "analyze done"

# ─────────────────────────────────────────────────────────────
# Step 2: Build ML features
# ─────────────────────────────────────────────────────────────
step 2 6 "Build ML features (build-features)"
if ! "$PYTHON" main.py build-features; then
    echo "  [WARN] build-features failed -- continuing without ML strategy."
    log "WARN: build-features failed (skipped)"
else
    log "build-features done"

    # ─────────────────────────────────────────────────────────
    # Step 3: Retrain model
    # ─────────────────────────────────────────────────────────
    step 3 6 "Retrain model (train-model)"
    if ! "$PYTHON" main.py train-model; then
        echo "  [WARN] train-model failed -- continuing without ML strategy."
        log "WARN: train-model failed (skipped)"
    else
        log "train-model done"
    fi
fi

# ─────────────────────────────────────────────────────────────
# Step 4: Collect + Generate 5 games (balanced_v2 x3 + random x2)
#   collect: 최신 당첨번호 증분 수집
#   generate: 구매 이력 번호 자동 제외
# ─────────────────────────────────────────────────────────────
step 4 6 "Collect + Generate 5 games"
"$PYTHON" main.py run --balanced-v2-games 3 --random-games 2 --balanced-games 0 || die "game generation failed"
log "game generation done (5 games)"

# ─────────────────────────────────────────────────────────────
# Step 5: Auto-purchase (1 game at a time, 5 rounds)
#   실제 금전 거래 발생 — DH_LOGIN_ID/DH_LOGIN_PW 필수
# ─────────────────────────────────────────────────────────────
step 5 6 "Auto-purchase (1 game x 5 rounds)"
"$PYTHON" main.py buy-lotto --max-games 5
BUY_EXIT=$?
if [ $BUY_EXIT -ne 0 ]; then
    echo "  [WARN] buy-lotto exited with code $BUY_EXIT -- continuing to check-results"
    log "WARN: buy-lotto exit=$BUY_EXIT"
else
    log "buy-lotto done"
fi

# ─────────────────────────────────────────────────────────────
# Step 6: Check results (compare generated numbers vs actual draws)
#   이전 회차 번호의 적중 결과를 prediction_results.csv 에 누적
#   이번 주 구매분은 당첨 발표 전이므로 자동으로 스킵됨
# ─────────────────────────────────────────────────────────────
step 6 6 "Check results (feedback loop)"
if ! "$PYTHON" main.py check-results; then
    echo "  [WARN] check-results failed -- non-critical, skipping"
    log "WARN: check-results failed (skipped)"
else
    log "check-results done"
fi

log "=== Weekly lotto pipeline finished ==="
echo ""
echo "[DONE] Pipeline complete"
