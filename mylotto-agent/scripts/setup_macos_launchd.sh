#!/bin/bash
# setup_macos_launchd.sh
# macOS launchd를 이용해 weekly_lotto.sh를 매주 토요일 09:00에 자동 실행
#
# 사용법:
#   bash scripts/setup_macos_launchd.sh          # 등록
#   bash scripts/setup_macos_launchd.sh unload   # 해제

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.mylotto.weekly"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ "${1:-}" = "unload" ]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "launchd job '$LABEL' 해제 완료"
    exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$PROJECT_DIR/logs"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$PROJECT_DIR/weekly_lotto.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>

    <!-- 매주 토요일 09:00 KST -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>6</integer>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/logs/launchd_err.log</string>

    <!-- Mac이 슬립 상태였다가 깨어났을 때 스킵된 실행을 바로 실행 -->
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "✓ launchd 등록 완료: $LABEL"
echo "  plist: $PLIST"
echo "  로그:  $PROJECT_DIR/logs/launchd.log"
echo ""
echo "  확인:   launchctl list | grep mylotto"
echo "  해제:   bash scripts/setup_macos_launchd.sh unload"
