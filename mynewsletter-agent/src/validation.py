import re

from .config import LANG_FOREIGN
from .merge import parse_korean_chunk_output

PLACEHOLDER_PHRASES = ["(핵심 내용 없음)", "(해당 없음)", "(내용 없음)", "정보 없음"]

MIN_BULLETS = 5
MIN_TERMS = 3
MIN_SUMMARY_CHARS = 60
MIN_OUTPUT_RATIO = 0.3

INCOMPLETE_ENDING_WORDS = [
    "직접적인", "위한", "통해", "따라", "때문에", "것으로", "대해", "관련해", "그리고", "하지만",
]


def has_enough_korean(text: str) -> bool:
    korean_count = len(re.findall(r"[가-힣]", text))
    return korean_count >= 30


def contains_long_english_sentence(text: str) -> bool:
    # 기술 용어·고유명사는 허용하되, 영어 문장 전체가 남는 경우를 탐지한다.
    for line in text.splitlines():
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", line)
        if len(words) >= 10:
            return True
    return False


def validate_result(text: str, language: str) -> tuple[bool, str]:
    if not text.strip():
        return False, "empty output"

    if not has_enough_korean(text):
        return False, "insufficient Korean output"

    if language == LANG_FOREIGN and contains_long_english_sentence(text):
        return False, "untranslated English sentence remains"

    return True, "ok"


def _last_meaningful_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def is_incomplete_sentence(last_line: str) -> bool:
    trimmed = last_line.strip().rstrip("\"')」』” ")
    if not trimmed:
        return False

    if re.search(r"[.!?]$", trimmed):
        return False

    if re.search(r"(다|요|죠|임|함)$", trimmed):
        return False

    return any(trimmed.endswith(word) for word in INCOMPLETE_ENDING_WORDS)


def validate_korean_result(text: str, source_length: int) -> tuple[bool, str]:
    if not text.strip():
        return False, "empty_output"

    for phrase in PLACEHOLDER_PHRASES:
        if phrase in text:
            return False, f"placeholder_phrase:{phrase}"

    sections = parse_korean_chunk_output(text)
    summary_text = "\n".join(sections["summary"]).strip()
    bullets = sections["bullets"]
    terms = sections["terms"]

    if not summary_text:
        return False, "missing_summary_section"

    if not has_enough_korean(summary_text) or len(summary_text) < MIN_SUMMARY_CHARS:
        return False, "summary_too_short"

    if len(bullets) < MIN_BULLETS:
        return False, "not_enough_bullets"

    if len(terms) < MIN_TERMS:
        return False, "not_enough_terms"

    if source_length > 0 and len(text) < source_length * MIN_OUTPUT_RATIO:
        return False, "output_too_short_relative_to_source"

    non_header_chars = (
        len(summary_text) + sum(len(b) for b in bullets) + sum(len(t) for t in terms)
    )
    if non_header_chars < MIN_SUMMARY_CHARS * 2:
        return False, "title_and_headers_only"

    if is_incomplete_sentence(_last_meaningful_line(summary_text)):
        return False, "incomplete_sentence"

    return True, "ok"
