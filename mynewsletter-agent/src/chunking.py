import re

from .config import CHUNK_OVERLAP, CHUNK_SIZE

SENTENCE_END_PATTERN = re.compile(r"[.!?다요죠까]\s")


def _find_natural_break(text: str, limit: int, lookback: int) -> int:
    window_start = max(0, limit - lookback)
    window = text[window_start:limit]

    last_sentence_end = -1
    for match in SENTENCE_END_PATTERN.finditer(window):
        last_sentence_end = match.end()

    if last_sentence_end != -1:
        return window_start + last_sentence_end

    last_space = window.rfind(" ")
    if last_space != -1:
        return window_start + last_space + 1

    return limit


def _split_long_paragraph(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    # 단일 문단이 chunk_size를 넘는 드문 경우에만 사용한다.
    # overlap은 텍스트를 중복시키는 용도가 아니라, chunk_size 부근에서 문장/단어가
    # 끊기지 않는 자연스러운 절단 지점을 뒤로 최대 overlap자까지 탐색하는 범위다.
    pieces = []
    remaining = paragraph

    while len(remaining) > chunk_size:
        cut = _find_natural_break(remaining, chunk_size, overlap)
        if cut <= 0:
            cut = chunk_size

        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        pieces.append(remaining)

    return pieces


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """본문을 문단(줄) 단위로 가능한 자연스럽게 chunk_size 이내로 묶어서 분할한다.
    문단 하나가 chunk_size보다 크면 그 문단만 문장/단어 경계 기준으로 재분할한다."""
    paragraphs = [p for p in text.split("\n") if p.strip()]

    if not paragraphs:
        return [text.strip()] if text.strip() else []

    units = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_split_long_paragraph(paragraph, chunk_size, overlap))

    chunks = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit) + 1

        if current and current_len + unit_len > chunk_size:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        current.append(unit)
        current_len += unit_len

    if current:
        chunks.append("\n".join(current))

    return chunks
