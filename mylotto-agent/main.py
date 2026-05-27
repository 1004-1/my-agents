"""엔트리포인트 — `python main.py <command>`."""

import io
import sys

# Windows 터미널 인코딩을 UTF-8로 강제 설정 (cp949 이모지 오류 방지)
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.lotto.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
