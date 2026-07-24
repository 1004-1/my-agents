import re

from .config import LANG_FOREIGN, LANG_KOREAN


def clean_text(text: str) -> str:
    skip_keywords = [
        "unsubscribe",
        "view this post",
        "view in browser",
        "privacy policy",
        "manage preferences",
        "이 메일이 잘 안보이시나요",
        "웹으로 보기",
        "잘림 없이 읽기",
        "구독하기",
        "수신거부",
    ]

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower()
        if any(keyword in lower for keyword in skip_keywords):
            continue

        lines.append(line)

    return "\n".join(lines)


def detect_language(subject: str, body: str) -> str:
    sample = f"{subject}\n{body[:3000]}"
    korean_count = len(re.findall(r"[가-힣]", sample))
    latin_count = len(re.findall(r"[A-Za-z]", sample))

    # 한국어 문자가 충분히 많고, 영어 문자 대비 일정 비율 이상이면 한국어 메일로 본다.
    if korean_count >= 20 and korean_count >= latin_count * 0.15:
        return LANG_KOREAN

    return LANG_FOREIGN
