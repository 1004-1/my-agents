import os
import re
import time
import base64
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText

import ollama
from bs4 import BeautifulSoup

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

LABEL_NAME = "Meco_3631eff2-09a3-465b-8569-8d6e623ef8f3"

OLLAMA_MODEL = "llama3.1:8b"
MAX_BODY_CHARS = 12000
MAX_REPORT_CHARS = 90000
TIMEZONE = ZoneInfo("Asia/Seoul")

DRY_RUN_DELETE = False


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


def build_summary_prompt(subject, sender, body, retry=False):
    retry_notice = ""

    if retry:
        retry_notice = """
이전 답변에 한국어가 아닌 문장이 포함되었습니다.
이번 답변은 반드시 한국어로만 다시 작성하세요.
"""

    return f"""
아래 뉴스레터를 정리하라.

작업 규칙:
- 결과는 반드시 한국어로만 작성한다.
- 원문이 한국어이면 전체 내용을 약 70% 수준으로 자세히 요약한다.
- 원문이 영어이면 전체 내용을 한국어로 번역하되, 기술 용어는 억지로 번역하지 않는다.
- 회사명, 제품명, 서비스명, API명, 프로그래밍 언어명, 오픈소스 프로젝트명은 원문을 유지한다.
- 광고, 수신거부, 구독 안내, 단순 링크 문구는 제외한다.
- 원문에 없는 내용을 추측해서 추가하지 않는다.
- 중요한 숫자, 날짜, 인물, 기업명, 제품명, 사례는 가능한 유지한다.
- 아래 작업 규칙은 출력하지 않는다.

{retry_notice}

출력 형식:

## 제목
{subject}

## 전체 정리
원문의 흐름을 따라 문단 단위로 자세히 정리한다.

## 주요 내용
- 핵심 내용을 5~10개 bullet로 정리한다.

## 기억할 만한 표현/용어
- 중요한 기술 용어, 기업명, 제품명, 개념을 정리한다.

{retry_notice}

[원문]
{body}
"""


def summarize_with_ollama(subject, sender, body):
    body = clean_text(body)
    body = body[:MAX_BODY_CHARS]

    system_prompt = """
당신은 뉴스레터 정리 비서입니다.

규칙:
1. 모든 출력은 반드시 한국어로 작성한다.
2. 중국어, 일본어, 영어 문장을 출력하지 않는다.
3. 광고, 수신거부, 링크 안내 문구는 제외한다.
4. 원문의 의미를 왜곡하지 않는다.
5. 규칙 자체는 출력하지 않는다.
6. 기술 용어를 억지로 한글화하지 않는다.
7. 회사명, 제품명, 서비스명, 기술 용어는 원문을 유지한다.
"""

    prompt = build_summary_prompt(subject, sender, body, retry=False)

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
        retry_prompt = build_summary_prompt(subject, sender, body, retry=True)

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

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES,
            )
            creds = flow.run_local_server(
                host="localhost",
                port=8080,
                open_browser=False,
            )

        with open("token.json", "w") as token:
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

    texts = []

    for part in payload.get("parts", []):
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


def get_summary_window():
    now = datetime.now(TIMEZONE)

    today_7 = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if now < today_7:
        today_7 = today_7 - timedelta(days=1)

    yesterday_7 = today_7 - timedelta(days=1)

    summarize_start = yesterday_7
    summarize_end = today_7 - timedelta(seconds=1)

    return summarize_start, summarize_end


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
        msg = load_full_message(service, item["id"])
        summarize_items.append(msg)

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

        summary = summarize_with_ollama(subject, sender, body)

        elapsed = time.time() - start_time
        log(f"[DONE] summarize: {subject} / elapsed={elapsed:.1f}s")

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
            trash_message(service, message_id)
            log(f"휴지통 이동 완료: {received_at:%Y-%m-%d %H:%M:%S} / {subject}")

    total_elapsed = time.time() - job_start_time
    log(f"\n완료 / total_elapsed={total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
