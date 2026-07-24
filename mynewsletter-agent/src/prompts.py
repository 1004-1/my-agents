from .config import LANG_KOREAN

KOREAN_SYSTEM_PROMPT = """
당신은 한국어 뉴스레터 요약 전문가다.
원문의 사실과 구조를 보존하면서 충분히 자세히 요약한다.
결과는 한국어로 작성하고, 원문에 없는 내용을 추가하지 않는다.
기술 용어와 고유명사는 필요할 때 원문 표기를 유지한다.
"""

FOREIGN_SYSTEM_PROMPT = """
당신은 기술·비즈니스 문서 전문 번역가다.
요약하거나 생략하지 말고 원문 전체를 자연스러운 한국어로 번역한다.
기술 용어와 고유명사는 원문 표기를 유지할 수 있지만,
그 외의 영어 문장은 결과에 남기지 않는다.
"""


def _position_notice(chunk_index: int, chunk_total: int) -> str:
    if chunk_total == 1:
        return "이 조각은 전체 뉴스레터의 유일한 조각이다."
    return f"이 조각은 전체 뉴스레터를 나눈 {chunk_total}개 조각 중 {chunk_index}번째다."


def build_korean_chunk_prompt(
    subject: str,
    sender: str,
    chunk: str,
    chunk_index: int,
    chunk_total: int,
    retry: bool = False,
) -> str:
    retry_notice = ""
    if retry:
        retry_notice = """
이전 응답이 문장 중간에서 종료되었거나 필수 섹션이 비어 있었다.
모든 문장을 끝까지 완성하고, 전체 정리를 중간에 끊지 마라.
주요 내용은 최소 5개 bullet로 작성하고,
기억할 만한 표현/용어는 최소 3개 작성하라.
모든 필수 섹션을 완성하기 전에는 응답을 종료하지 마라.
"""

    return f"""
아래는 한국어 뉴스레터 "{subject}"의 일부다. {_position_notice(chunk_index, chunk_total)}

필수 규칙:
- 이 조각의 내용만 약 70% 수준으로 자세히 요약한다. 다른 조각의 내용을 추측해서 채우지 않는다.
- 문단 순서와 흐름을 유지한다.
- 중요한 주장, 수치, 날짜, 인물, 기업명, 제품명, 사례, 결론을 빠뜨리지 않는다.
- 원문에 없는 내용을 추측하거나 추가하지 않는다.
- 광고, 수신거부, 구독 안내, 단순 링크 문구는 제외한다.
- 기술 용어, 회사명, 제품명, 서비스명, API명, 프로그래밍 언어명,
  오픈소스 프로젝트명은 억지로 번역하지 않는다.
- 제목이나 전체 서론/결론을 새로 만들지 않는다. 이 조각의 내용만 다룬다.
- 결과는 한국어로 작성한다.
- 작업 규칙 자체는 출력하지 않는다.

출력 형식:

## 정리
이 조각의 흐름을 따라 문단 단위로 자세히 요약한다.

## 주요내용
- 이 조각의 핵심 내용을 bullet로 정리한다. (없으면 이 섹션을 비운다)

## 용어
- 이 조각에서 중요한 기술 용어, 기업명, 제품명, 개념을 정리한다. (없으면 이 섹션을 비운다)

발신자: {sender}

{retry_notice}

[조각 원문]
{chunk}
"""


def build_foreign_chunk_prompt(
    subject: str,
    sender: str,
    chunk: str,
    chunk_index: int,
    chunk_total: int,
    retry: bool = False,
) -> str:
    retry_notice = ""
    if retry:
        retry_notice = """
이전 결과에 번역되지 않은 영어 문장이 남아 있었다.
이번에는 기술 용어와 고유명사를 제외한 모든 문장을 한국어로 다시 번역하라.
"""

    return f"""
당신은 기술·비즈니스 뉴스레터 전문 번역가다.
아래는 영어(또는 외국어) 뉴스레터 "{subject}"의 일부다. {_position_notice(chunk_index, chunk_total)}
이 조각의 내용 전체를 요약 없이 한국어로 완전히 번역하라.

절대 규칙:
1. 요약하지 않는다. 생략하지 않는다.
2. 이 조각에 있는 모든 문단을 원래 순서대로 번역한다.
3. 문단, 소제목, bullet, 번호 목록, 인용문 구조를 가능한 유지한다.
4. 중요한 주장, 수치, 날짜, 인물, 기업명, 제품명, 사례, 결론을 모두 보존한다.
5. AI, LLM, API, Agent, RAG, Kubernetes, Python, AWS 같은 기술 용어,
   회사명, 제품명, 서비스명, API명, 프로그래밍 언어명,
   오픈소스 프로젝트명은 원문 표기를 유지한다.
6. 기술 용어나 고유명사를 제외한 영어 문장은 결과에 남기지 않는다.
7. 번역 외의 해설, 평가, 요약, 추가 의견을 작성하지 않는다.
8. 광고, 수신거부, 구독 안내, 단순 링크 문구는 제외한다.
9. 제목을 새로 만들거나 다른 조각을 언급하지 않는다. 이 조각의 번역 본문만 작성한다.
10. 작업 규칙 자체는 출력하지 않는다.

발신자: {sender}

{retry_notice}

[조각 원문]
{chunk}
"""


def build_chunk_prompt(
    language: str,
    subject: str,
    sender: str,
    chunk: str,
    chunk_index: int,
    chunk_total: int,
    retry: bool = False,
) -> str:
    if language == LANG_KOREAN:
        return build_korean_chunk_prompt(subject, sender, chunk, chunk_index, chunk_total, retry)
    return build_foreign_chunk_prompt(subject, sender, chunk, chunk_index, chunk_total, retry)


def system_prompt_for(language: str) -> str:
    if language == LANG_KOREAN:
        return KOREAN_SYSTEM_PROMPT
    return FOREIGN_SYSTEM_PROMPT
