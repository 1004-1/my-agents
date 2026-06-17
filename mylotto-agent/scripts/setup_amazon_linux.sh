#!/bin/bash
# setup_amazon_linux.sh
# Amazon Linux 2023 기준 mylotto-agent 최초 환경 구성 스크립트
#
# 실행 방법:
#   chmod +x scripts/setup_amazon_linux.sh
#   bash scripts/setup_amazon_linux.sh
#
# 전제 조건:
#   - Amazon Linux 2023 (AL2023) — AL2는 지원 미보장
#   - sudo 권한
#   - .env 파일에 DH_LOGIN_ID, DH_LOGIN_PW 설정 완료
#     (서버에는 브라우저 화면이 없으므로 자동 로그인 필수)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=""
VENV_DIR="$PROJECT_DIR/.venv"

# ── 색상 출력 ──────────────────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[0;33m"; RED="\033[0;31m"; NC="\033[0m"
ok()   { echo -e "${GREEN}[OK]${NC}  $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
die()  { echo -e "${RED}[ERROR]${NC} $1" >&2; exit 1; }

echo "========================================"
echo " mylotto-agent  Amazon Linux 2023 Setup"
echo "========================================"
echo ""

# ── 1. Amazon Linux 2023 확인 ──────────────────────────────────
if ! grep -q "Amazon Linux" /etc/os-release 2>/dev/null; then
    warn "Amazon Linux 2023이 아닙니다. 계속 진행하지만 오류가 발생할 수 있습니다."
fi
if grep -q "VERSION_ID=\"2\"" /etc/os-release 2>/dev/null; then
    die "Amazon Linux 2는 지원하지 않습니다. Amazon Linux 2023을 사용하세요."
fi
ok "Amazon Linux 2023 확인"

# ── 2. 시스템 패키지 업데이트 ─────────────────────────────────
echo ""
echo "[1/6] 시스템 패키지 업데이트..."
sudo dnf update -y --quiet
ok "dnf update 완료"

# ── 3. Python 3.11 설치 ────────────────────────────────────────
echo ""
echo "[2/6] Python 3.11 설치..."

# AL2023은 python3.11이 기본 포함됨
if command -v python3.11 &>/dev/null; then
    PYTHON_BIN="$(command -v python3.11)"
    ok "Python 3.11 이미 설치됨: $PYTHON_BIN"
else
    sudo dnf install -y python3.11 python3.11-pip
    PYTHON_BIN="$(command -v python3.11)"
    ok "Python 3.11 설치 완료: $PYTHON_BIN"
fi

PYTHON_VER=$("$PYTHON_BIN" --version 2>&1)
echo "  사용 버전: $PYTHON_VER"

# ── 4. Chromium 시스템 의존성 설치 ────────────────────────────
echo ""
echo "[3/6] Chromium 헤드리스 시스템 의존성 설치..."

# Playwright 1.60 기준 AL2023에서 필요한 패키지
CHROMIUM_DEPS=(
    # 기본 그래픽/UI 라이브러리
    alsa-lib
    atk
    at-spi2-atk
    at-spi2-core
    cups-libs
    gtk3
    libdrm
    libgbm
    libX11
    libXcomposite
    libXcursor
    libXdamage
    libXext
    libXfixes
    libXi
    libXrandr
    libXScrnSaver
    libXtst
    libxkbcommon
    mesa-libgbm
    # 네트워크/보안
    nss
    nspr
    # 폰트
    xorg-x11-fonts-Type1
    google-noto-sans-cjk-fonts
    # 기타
    pango
    gdk-pixbuf2
    cairo
    glib2
)

sudo dnf install -y "${CHROMIUM_DEPS[@]}" --quiet
ok "Chromium 의존성 설치 완료"

# ── 5. 가상환경 생성 및 Python 패키지 설치 ────────────────────
echo ""
echo "[4/6] Python 가상환경 생성..."

if [ -d "$VENV_DIR" ]; then
    warn "가상환경이 이미 존재합니다: $VENV_DIR (건너뜀)"
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    ok "가상환경 생성: $VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

echo "  pip 업그레이드..."
"$VENV_PY" -m pip install --upgrade pip --quiet
ok "pip 업그레이드 완료"

echo "  requirements.txt 설치 (시간이 걸릴 수 있습니다)..."
"$VENV_PIP" install -r "$PROJECT_DIR/requirements.txt" --quiet
ok "Python 패키지 설치 완료"

# ── 6. Playwright Chromium 설치 ───────────────────────────────
echo ""
echo "[5/6] Playwright Chromium 설치..."

PLAYWRIGHT="$VENV_DIR/bin/playwright"

# install-deps 먼저 시도 (AL2023은 dnf 감지로 성공 가능)
echo "  playwright install-deps 시도..."
if "$PLAYWRIGHT" install-deps chromium 2>/dev/null; then
    ok "playwright install-deps 완료"
else
    warn "playwright install-deps 실패 — 위에서 수동으로 설치한 패키지로 대체 시도"
fi

"$PLAYWRIGHT" install chromium
ok "Playwright Chromium 설치 완료"

# ── 7. .env 파일 확인 ─────────────────────────────────────────
echo ""
echo "[6/6] .env 설정 확인..."

ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    warn ".env 파일이 없습니다. 아래를 참고해 생성하세요:"
    cat <<'EOF'

  cat > .env << 'ENVEOF'
  DH_LOGIN_ID=<동행복권 아이디>
  DH_LOGIN_PW=<동행복권 비밀번호>
  HEADLESS=true
  ENVEOF

EOF
else
    # HEADLESS=true 확인
    if grep -q "HEADLESS=false" "$ENV_FILE"; then
        warn ".env에 HEADLESS=false가 설정되어 있습니다."
        warn "서버(EC2)에서는 HEADLESS=true 로 변경하세요:"
        echo "  sed -i 's/HEADLESS=false/HEADLESS=true/' $ENV_FILE"
    fi
    # DH_LOGIN_ID 확인
    if ! grep -q "DH_LOGIN_ID=" "$ENV_FILE" || grep -q "DH_LOGIN_ID=$" "$ENV_FILE"; then
        warn "DH_LOGIN_ID가 설정되지 않았습니다."
        warn "헤드리스 서버에서는 자동 로그인을 위해 반드시 설정해야 합니다."
    else
        ok "DH_LOGIN_ID 설정됨"
    fi
    if ! grep -q "DH_LOGIN_PW=" "$ENV_FILE" || grep -q "DH_LOGIN_PW=$" "$ENV_FILE"; then
        warn "DH_LOGIN_PW가 설정되지 않았습니다."
    else
        ok "DH_LOGIN_PW 설정됨"
    fi
fi

# ── 8. data 디렉터리 초기화 ───────────────────────────────────
mkdir -p "$PROJECT_DIR/data/models" "$PROJECT_DIR/logs" "$PROJECT_DIR/screenshots"
ok "data/, logs/, screenshots/ 디렉터리 확인"

# ── 9. 당첨번호 초기 수집 ─────────────────────────────────────
echo ""
echo "[선택] 당첨번호 초기 수집을 지금 실행하시겠습니까? (y/N)"
read -r -t 30 answer || answer="n"
if [[ "$answer" =~ ^[Yy]$ ]]; then
    cd "$PROJECT_DIR"
    "$VENV_PY" main.py collect
    ok "당첨번호 초기 수집 완료"
else
    echo "  건너뜀 — 나중에 'python main.py collect' 로 실행하세요."
fi

# ── 10. 완료 메시지 ────────────────────────────────────────────
echo ""
echo "========================================"
echo " 설치 완료"
echo "========================================"
echo ""
echo "다음 단계:"
echo ""
echo "  1) .env 파일 확인 (DH_LOGIN_ID, DH_LOGIN_PW, HEADLESS=true)"
echo "     vi $ENV_FILE"
echo ""
echo "  2) 파이프라인 테스트 실행:"
echo "     cd $PROJECT_DIR"
echo "     .venv/bin/python main.py run --balanced-v2-games 3 --random-games 2 --balanced-games 0"
echo "     .venv/bin/python main.py buy-lotto --max-games 1   # 1게임만 먼저 테스트"
echo ""
echo "  3) cron 등록 (매주 토요일 09:00):"
echo "     crontab -e"
echo "     0 9 * * 6 /bin/bash $PROJECT_DIR/weekly_lotto.sh >> $PROJECT_DIR/logs/cron.log 2>&1"
echo ""
