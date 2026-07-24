def parse_korean_chunk_output(text: str) -> dict:
    sections = {"summary": [], "bullets": [], "terms": []}
    current = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("## "):
            heading = stripped[3:].strip()
            if "정리" in heading:
                current = "summary"
            elif "주요" in heading:
                current = "bullets"
            elif "용어" in heading or "표현" in heading:
                current = "terms"
            else:
                current = None
            continue

        if current == "summary" and stripped:
            sections["summary"].append(stripped)
        elif current == "bullets" and stripped.startswith("-"):
            sections["bullets"].append(stripped)
        elif current == "terms" and stripped.startswith("-"):
            sections["terms"].append(stripped)

    if not any(sections.values()):
        sections["summary"] = [text.strip()] if text.strip() else []

    return sections


def merge_korean_chunk_outputs(chunk_outputs: list[str]) -> tuple[str, list[str], list[str]]:
    summaries = []
    bullets = []
    terms = []
    seen_terms = set()

    for output in chunk_outputs:
        sections = parse_korean_chunk_output(output)

        if sections["summary"]:
            summaries.append("\n".join(sections["summary"]))

        bullets.extend(sections["bullets"])

        for term in sections["terms"]:
            if term not in seen_terms:
                seen_terms.add(term)
                terms.append(term)

    return "\n\n".join(summaries), bullets, terms


def format_korean_result(
    subject: str,
    summary_body: str,
    bullets: list[str],
    terms: list[str],
) -> str:
    bullet_block = "\n".join(bullets) if bullets else "- (핵심 내용 없음)"
    term_block = "\n".join(terms) if terms else "- (해당 없음)"

    return f"""## 제목
{subject}

## 전체 정리
{summary_body}

## 주요 내용
{bullet_block}

## 기억할 만한 표현/용어
{term_block}
"""


def format_foreign_result(subject: str, translation_body: str) -> str:
    return f"""## 제목
{subject}

## 전체 번역
{translation_body}
"""
