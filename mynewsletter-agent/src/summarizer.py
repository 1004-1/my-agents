import time

from .chunking import split_into_chunks
from .config import CHUNK_OVERLAP, CHUNK_SIZE, LANG_KOREAN, NORMAL_FINISH_REASONS
from .llm_client import summarize_chunk_with_gemini, summarize_chunk_with_ollama
from .logutil import log
from .merge import format_foreign_result, format_korean_result, merge_korean_chunk_outputs
from .validation import validate_korean_result, validate_result


def _validate_chunk_content(text: str, language: str, source_length: int) -> tuple[bool, str]:
    if language == LANG_KOREAN:
        return validate_korean_result(text, source_length)
    return validate_result(text, language)


def _attempt_gemini_chunk(
    subject: str,
    sender: str,
    chunk: str,
    language: str,
    chunk_index: int,
    chunk_total: int,
    retry: bool,
) -> tuple[bool, str, str]:
    try:
        text, finish_reason = summarize_chunk_with_gemini(
            subject, sender, chunk, language, chunk_index, chunk_total, retry=retry,
        )
    except Exception as exc:
        return False, "", f"gemini_error:{exc}"

    if finish_reason not in NORMAL_FINISH_REASONS:
        return False, text, f"finish_reason:{finish_reason}"

    valid, reason = _validate_chunk_content(text, language, len(chunk))
    return valid, text, reason


def process_chunk(
    subject: str,
    sender: str,
    chunk: str,
    language: str,
    chunk_index: int,
    chunk_total: int,
) -> tuple[str, str]:
    started_at = time.time()
    prefix = f"[Chunk {chunk_index}/{chunk_total}]"

    valid, text, reason = _attempt_gemini_chunk(
        subject, sender, chunk, language, chunk_index, chunk_total, retry=False,
    )
    log(
        f"{prefix} model=gemini validation={'success' if valid else 'failed'}"
        + ("" if valid else f" reason={reason}")
    )

    if not valid:
        log(f"{prefix} Gemini retry")
        valid, text, reason = _attempt_gemini_chunk(
            subject, sender, chunk, language, chunk_index, chunk_total, retry=True,
        )
        log(
            f"{prefix} model=gemini retry=1 validation={'success' if valid else 'failed'}"
            + ("" if valid else f" reason={reason}")
        )

    if valid:
        used_model = "gemini"
        result = text
    else:
        log(f"{prefix} Gemini 실패 → Ollama fallback. reason={reason}")

        ollama_text = summarize_chunk_with_ollama(
            subject, sender, chunk, language, chunk_index, chunk_total,
        )
        ollama_valid, ollama_reason = _validate_chunk_content(ollama_text, language, len(chunk))
        log(
            f"{prefix} model=ollama validation={'success' if ollama_valid else 'failed'}"
            + ("" if ollama_valid else f" reason={ollama_reason}")
        )

        if not ollama_valid:
            raise RuntimeError(f"모든 모델에서 결과 검증 실패. reason={ollama_reason}")

        used_model = "ollama"
        result = ollama_text

    elapsed = time.time() - started_at
    fallback_note = " (fallback)" if used_model == "ollama" else ""
    log(
        f"{prefix} model={used_model}{fallback_note} / "
        f"elapsed={elapsed:.1f}s"
    )

    return result, used_model


def summarize(subject: str, sender: str, body: str, language: str) -> tuple[str, str]:
    chunks = split_into_chunks(body, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_total = len(chunks)

    if chunk_total == 0:
        raise RuntimeError("본문이 비어 있어 처리할 내용이 없습니다.")

    log(
        f"[Chunk 분할] {chunk_total}개 조각 "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, 원문={len(body)}자)"
    )

    chunk_outputs = []
    gemini_count = 0
    ollama_count = 0
    chunks_started_at = time.time()

    for index, chunk in enumerate(chunks, start=1):
        result, used_model = process_chunk(subject, sender, chunk, language, index, chunk_total)
        chunk_outputs.append(result)

        if used_model == "gemini":
            gemini_count += 1
        else:
            ollama_count += 1

    chunks_elapsed = time.time() - chunks_started_at
    log(
        f"[Chunk 처리 완료] 총 {chunk_total}개 / gemini={gemini_count} / "
        f"ollama={ollama_count} / 총 처리시간={chunks_elapsed:.1f}s"
    )

    if language == LANG_KOREAN:
        summary_body, bullets, terms = merge_korean_chunk_outputs(chunk_outputs)
        final_result = format_korean_result(subject, summary_body, bullets, terms)
    else:
        translation_body = "\n\n".join(output.strip() for output in chunk_outputs)
        final_result = format_foreign_result(subject, translation_body)

    if ollama_count == 0:
        used_model = "gemini"
    elif gemini_count == 0:
        used_model = "ollama"
    else:
        used_model = "mixed"

    return final_result, used_model
