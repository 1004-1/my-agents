import os

import ollama
from google import genai
from google.genai import types

from .config import GEMINI_MODEL, LANG_KOREAN, OLLAMA_MODEL
from .logutil import log
from .prompts import build_chunk_prompt, system_prompt_for

_gemini_client = None


def _get_gemini_client():
    global _gemini_client

    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 .env에 설정되지 않았습니다.")
        _gemini_client = genai.Client(api_key=api_key)

    return _gemini_client


def _extract_gemini_meta(response) -> dict:
    finish_reason = None
    candidates = getattr(response, "candidates", None) or []

    if candidates:
        raw_finish_reason = getattr(candidates[0], "finish_reason", None)
        if raw_finish_reason is not None:
            finish_reason = getattr(raw_finish_reason, "name", None) or str(raw_finish_reason)

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", None) if usage is not None else None
    output_tokens = getattr(usage, "candidates_token_count", None) if usage is not None else None
    total_tokens = getattr(usage, "total_token_count", None) if usage is not None else None

    return {
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def summarize_chunk_with_gemini(
    subject: str,
    sender: str,
    chunk: str,
    language: str,
    chunk_index: int,
    chunk_total: int,
    retry: bool = False,
) -> tuple[str, str]:
    prompt = build_chunk_prompt(language, subject, sender, chunk, chunk_index, chunk_total, retry)
    client = _get_gemini_client()

    max_output_tokens = 8192 if language == LANG_KOREAN else 12288
    temperature = 0.05 if retry else 0.15

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt_for(language),
            temperature=temperature,
            top_p=0.9,
            max_output_tokens=max_output_tokens,
        ),
    )

    meta = _extract_gemini_meta(response)
    prefix = f"[Chunk {chunk_index}/{chunk_total}]"
    log(f"{prefix} [Gemini 응답] finish_reason={meta['finish_reason']}")
    log(
        f"{prefix} [Gemini 토큰] prompt={meta['prompt_tokens']} "
        f"output={meta['output_tokens']} total={meta['total_tokens']}"
    )

    if not response.text:
        raise RuntimeError(f"Gemini가 빈 응답을 반환했습니다. finish_reason={meta['finish_reason']}")

    return response.text.strip(), meta["finish_reason"]


def summarize_chunk_with_ollama(
    subject: str,
    sender: str,
    chunk: str,
    language: str,
    chunk_index: int,
    chunk_total: int,
    retry: bool = False,
) -> str:
    prompt = build_chunk_prompt(language, subject, sender, chunk, chunk_index, chunk_total, retry)

    num_predict = 3000 if language == LANG_KOREAN else 5000

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt_for(language)},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.15,
            "top_p": 0.9,
            "num_predict": num_predict,
        },
    )

    return response["message"]["content"].strip()
