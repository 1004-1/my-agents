import os
import re
import sys
import time
import base64
from datetime import datetime
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText

import ollama
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

LABEL_NAME = "Meco_3631eff2-09a3-465b-8569-8d6e623ef8f3"

GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_MODEL = "llama3.1:8b"
MAX_BODY_CHARS = 12000
MAX_REPORT_CHARS = 90000
TIMEZONE = ZoneInfo("Asia/Seoul")

DRY_RUN_DELETE = False

_gemini_client = None


def log(message):
    now = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def contains_chinese(text):
    return bool(re.search(r"[一-鿿]", text))


def is_korean_email(subject, body):
    text = subject + " " + body[:2000]
    korean_chars = len(re.findall(r"[가-힣ᄀ-ᇿ㄰-㆏]", text))
    return korean_chars >= 20


def clean_text(text):
    lines = []
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

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        lower = line.lower()
        if any(keyword in lower for keyword in skip_keywords):
            continue

        lines.append(line)

    return "\n".join(lines)


def build_summary_prompt(subject, body, retry=False):
    retry_notice = ""

    if retry:
        retry_notice = """
이전 답변에 한국어가 아닌 문장이 포함되었습니다.
이번 답변은 반드시 한국어로만 다시 작성하세요.
"""

    return f"""
아래 뉴스레터를 처리하라. 먼저 원문의 언어를 판별한 뒤, 아래 두 방식 중 해당하는 방식을 그대로 따른다.

공통 규칙:
- 결과는 반드시 한국어로만 작성한다.
- 회사명, 제품명, 서비스명, API명, 프로그래밍 언어명, 오픈소스 프로젝트명은 번역하지 않는다.
- 광고, 수신거부, 구독 안내, 단순 링크 문구는 제외한다.
- 원문에 없는 내용을 추측해서 추가하지 않는다.
- 중요한 숫자, 날짜, 인물, 기업명, 제품명, 사례는 반드시 유지한다.
- 아래 작업 규칙은 출력하지 않는다.

[원문이 한국어인 경우]
- 전체 내용을 약 80% 수준으로 자세히 요약한다.
- 출력 형식:

## 제목
{subject}

## 전체 정리
원문의 흐름을 따라 문단 단위로 자세히 정리한다.

## 주요 내용
- 핵심 내용을 5개이상의 bullet로 정리한다.

## 기억할 만한 표현/용어
- 중요한 기술 용어, 기업명, 제품명, 개념을 정리한다.

[원문이 영어(또는 기타 외국어)인 경우]
- 요약하지 않는다. bullet 요약, "주요 내용" 정리를 만들지 않는다.
- 원문 전체를 빠짐없이 자연스러운 한국어로 번역한다. 생략하지 않는다.
- 원문의 문단 구조와 흐름을 그대로 유지한다.
- 각 문단의 모든 세부 내용, 예시, 수치, 사례를 빠짐없이 옮긴다.
- AI, 데이터, 개발, 클라우드, 투자 관련 전문 용어는 원문을 그대로 유지한다.
- 출력 형식:

## 제목
{subject}

## 번역
원문을 문단 단위로 빠짐없이 번역한다. 요약이나 생략 없이 전체를 옮긴다.

{retry_notice}

[원문]
{body}
"""


SUMMARY_SYSTEM_PROMPT = """당신은 뉴스레터 정리 비서입니다.
규칙:
1. 모든 출력은 반드시 한국어로 작성한다.
2. 중국어, 일본어, 영어 문장을 출력하지 않는다.
3. 광고, 수신거부, 링크 안내 문구는 제외한다.
4. 원문의 의미를 왜곡하지 않는다.
5. 규칙 자체는 출력하지 않는다.
6. 기술 용어를 억지로 한글화하지 않는다.
7. 회사명, 제품명, 서비스명, 기술 용어는 원문을 유지한다."""


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 .env에 설정되지 않았습니다.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def summarize_with_gemini(subject, body):
    body = clean_text(body)
    body = body[:MAX_BODY_CHARS]

    prompt = build_summary_prompt(subject, body, retry=False)
    client = _get_gemini_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SUMMARY_SYSTEM_PROMPT,
            temperature=0.2,
            top_p=0.9,
            max_output_tokens=4096,
        ),
    )

    return response.text.strip()


def summarize(subject, body):
    try:
        summary = summarize_with_gemini(subject, body)
        return summary, "gemini"
    except Exception as e:
        log(f"[Gemini] 실패 → Ollama fallback. error={repr(e)}")
        summary = summarize_with_ollama(subject, body)
        return summary, "ollama"


def summarize_with_ollama(subject, body):
    body = clean_text(body)
    body = body[:MAX_BODY_CHARS]

    system_prompt = SUMMARY_SYSTEM_PROMPT

    prompt = build_summary_prompt(subject, body, retry=False)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 3000,
        },
    )

    summary = response["message"]["content"].strip()

    if contains_chinese(summary):
        retry_prompt = build_summary_prompt(subject, body, retry=True)

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                    + "\n중국어가 포함되면 실패입니다. 반드시 한국어만 출력하세요.",
                },
                {"role": "user", "content": retry_prompt},
            ],
            options={
                "temperature": 0.1,
                "top_p": 0.8,
                "num_predict": 1800,
            },
        )

        summary = response["message"]["content"].strip()

    return summary


def get_service():
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as e:
                log(f"[ERROR] Google OAuth 토큰 갱신 실패: {e}")
                log("[ERROR] 원인: refresh token이 만료되었거나 Google 계정에서 앱 접근이 취소되었습니다.")
                log("[ERROR] 해결 방법: token.json을 삭제하고 재인증이 필요합니다.")
                if os.path.exists(TOKEN_PATH):
                    os.remove(TOKEN_PATH)
                    log("[INFO] token.json을 삭제했습니다.")
                log("[INFO] OAuth 재인증이 필요합니다. python main.py를 수동 실행하세요.")
                sys.exit(1)
        else:
            if not sys.stdin.isatty():
                log("[ERROR] OAuth 인증 토큰이 없습니다.")
                log("[INFO] OAuth 재인증이 필요합니다. python main.py를 수동 실행하세요.")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH,
                SCOPES,
            )
            creds = flow.run_local_server(
                host="localhost",
                port=8080,
                open_browser=False,
            )

        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def get_my_email(service):
    profile = service.users().getProfile(userId="me").execute()
    return profile["emailAddress"]


def find_label_id(service, label_name):
    labels = service.users().labels().list(userId="me").execute()

    for label in labels["labels"]:
        if label["name"] == label_name:
            return label["id"]

    raise Exception(f"'{label_name}' 라벨을 찾을 수 없습니다.")


def decode_body(data):
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")


def extract_body(payload):
    body = payload.get("body", {})

    if body.get("data"):
        content = decode_body(body["data"])
        if payload.get("mimeType") == "text/html":
            return BeautifulSoup(content, "html.parser").get_text("\n")
        return content

    parts = payload.get("parts", [])

    if payload.get("mimeType") == "multipart/alternative":
        # 같은 내용의 서로 다른 표현(plain/html)이므로 하나만 골라 중복을 피한다.
        candidates = []

        for part in parts:
            mime_type = part.get("mimeType", "")
            part_body = part.get("body", {})

            if part_body.get("data") and mime_type == "text/plain":
                candidates.append(decode_body(part_body["data"]))
            elif part_body.get("data") and mime_type == "text/html":
                content = decode_body(part_body["data"])
                candidates.append(BeautifulSoup(content, "html.parser").get_text("\n"))
            elif part.get("parts"):
                nested = extract_body(part)
                if nested:
                    candidates.append(nested)

        if not candidates:
            return ""
        return max(candidates, key=lambda t: len(t.strip()))

    texts = []

    for part in parts:
        mime_type = part.get("mimeType", "")
        part_body = part.get("body", {})

        if part_body.get("data") and mime_type in ["text/plain", "text/html"]:
            content = decode_body(part_body["data"])

            if mime_type == "text/html":
                content = BeautifulSoup(content, "html.parser").get_text("\n")

            texts.append(content)

        if part.get("parts"):
            texts.append(extract_body(part))

    return "\n".join(texts)


def get_header(headers, target):
    for h in headers:
        if h["name"].lower() == target.lower():
            return h["value"]
    return ""


def internal_date_to_datetime(message):
    internal_ms = int(message["internalDate"])
    return datetime.fromtimestamp(internal_ms / 1000, tz=TIMEZONE)


def list_all_label_messages(service, label_id):
    messages = []
    page_token = None

    while True:
        result = (
            service.users()
            .messages()
            .list(
                userId="me",
                labelIds=[label_id],
                maxResults=500,
                pageToken=page_token,
            )
            .execute()
        )

        messages.extend(result.get("messages", []))

        page_token = result.get("nextPageToken")
        if not page_token:
            break

    return messages


def load_full_message(service, message_id):
    return (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )


def send_summary_email(service, to_email, subject, body_text, max_retries=3):
    message = MIMEText(body_text, "plain", "utf-8")
    message["to"] = to_email
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            result = (
                service.users()
                .messages()
                .send(userId="me", body={"raw": raw})
                .execute()
            )
            log(f"Gmail send result: {result}")
            return result

        except Exception as e:
            last_error = e
            log(f"[WARN] Gmail send failed. retry={attempt}/{max_retries}, error={repr(e)}")

            if attempt < max_retries:
                time.sleep(10 * attempt)

    raise last_error


def trash_message(service, message_id):
    service.users().messages().trash(userId="me", id=message_id).execute()


def main():
    job_start_time = time.time()

    service = get_service()
    my_email = get_my_email(service)
    label_id = find_label_id(service, LABEL_NAME)

    log(f"내 Gmail: {my_email}")
    log(f"Meco Label ID: {label_id}")
    log(f"DRY_RUN_DELETE: {DRY_RUN_DELETE}")

    candidates = list_all_label_messages(service, label_id)

    summarize_items = []

    for item in candidates:
        try:
            msg = load_full_message(service, item["id"])
            summarize_items.append(msg)
        except Exception as e:
            log(f"[WARN] 메일 로드 실패. id={item['id']}, error={repr(e)}")

    log(f"\n전체 Meco 메일 수: {len(candidates)}")
    log(f"요약 대상 메일 수: {len(summarize_items)}")

    korean_parts = []
    english_parts = []
    today = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
    run_time = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    for group_parts, label in [(korean_parts, "한글"), (english_parts, "영문")]:
        group_parts.append(f"# 뉴스레터 요약 ({label})")
        group_parts.append("")
        group_parts.append(f"- 실행 시각: {run_time}")

    korean_count = 0
    english_count = 0

    for idx, msg in enumerate(summarize_items, start=1):
        headers = msg["payload"].get("headers", [])

        subject = get_header(headers, "Subject")
        sender = get_header(headers, "From")
        body = extract_body(msg["payload"])

        lang = "한글" if is_korean_email(subject, body) else "영문"

        log("\n" + "=" * 20)
        log(f"[{idx}/{len(summarize_items)}] [{lang}]")
        log(f"제목: {subject}")
        log(f"발신: {sender}")

        start_time = time.time()
        log(f"[START] summarize: {subject}")

        summary, used_model = summarize(subject, body)

        elapsed = time.time() - start_time
        log(f"[DONE] summarize: {subject} / model={used_model} / elapsed={elapsed:.1f}s")

        if contains_chinese(summary):
            log("[경고] 최종 요약에 중국어가 포함되어 있습니다.")

        if lang == "한글":
            korean_count += 1
            parts = korean_parts
            num = korean_count
        else:
            english_count += 1
            parts = english_parts
            num = english_count

        parts.append("=" * 20)
        parts.append(f"## {num}. {subject}")
        parts.append("")
        parts.append(f"- 발신: {sender}")
        parts.append("")
        parts.append(summary)
        parts.append("")

    korean_parts.insert(3, f"- 요약 메일 수: {korean_count}")
    korean_parts.insert(4, "")
    english_parts.insert(3, f"- 요약 메일 수: {english_count}")
    english_parts.insert(4, "")

    def truncate_report(parts):
        body = "\n".join(parts)
        if len(body) > MAX_REPORT_CHARS:
            body = body[:MAX_REPORT_CHARS]
            body += "\n\n[알림] 요약 메일 본문이 너무 길어 일부가 잘렸습니다."
        return body

    send_start_time = time.time()

    if korean_count > 0:
        log("[START] send Korean summary email")
        send_summary_email(
            service=service,
            to_email=my_email,
            subject=f"[뉴스레터 요약 - 한글] {today}",
            body_text=truncate_report(korean_parts),
        )
        log(f"[DONE] 한글 요약 메일 발송 완료: {my_email}")
    else:
        log("한글 뉴스레터 없음 — 발송 생략")

    if english_count > 0:
        log("[START] send English summary email")
        send_summary_email(
            service=service,
            to_email=my_email,
            subject=f"[뉴스레터 요약 - 영문] {today}",
            body_text=truncate_report(english_parts),
        )
        log(f"[DONE] 영문 요약 메일 발송 완료: {my_email}")
    else:
        log("영문 뉴스레터 없음 — 발송 생략")

    send_elapsed = time.time() - send_start_time
    log(f"[DONE] send all summary emails / elapsed={send_elapsed:.1f}s")

    log("\n휴지통 이동 처리 시작")

    for msg in summarize_items:
        headers = msg["payload"].get("headers", [])
        subject = get_header(headers, "Subject")
        message_id = msg["id"]
        received_at = internal_date_to_datetime(msg)

        if DRY_RUN_DELETE:
            log(f"[DRY RUN] 휴지통 이동 예정: {received_at:%Y-%m-%d %H:%M:%S} / {subject}")
        else:
            try:
                trash_message(service, message_id)
                log(f"휴지통 이동 완료: {received_at:%Y-%m-%d %H:%M:%S} / {subject}")
            except Exception as e:
                log(f"[WARN] 휴지통 이동 실패. id={message_id}, subject={subject}, error={repr(e)}")

    total_elapsed = time.time() - job_start_time
    log(f"\n완료 / total_elapsed={total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
