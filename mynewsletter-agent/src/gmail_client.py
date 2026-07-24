import base64
import os
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText

from bs4 import BeautifulSoup
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .config import CREDENTIALS_PATH, SCOPES, TIMEZONE, TOKEN_PATH
from .logutil import log


def get_service():
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as exc:
                log(f"[ERROR] Google OAuth 토큰 갱신 실패: {exc}")
                log("[ERROR] refresh token이 만료되었거나 앱 접근 권한이 취소되었습니다.")

                if os.path.exists(TOKEN_PATH):
                    os.remove(TOKEN_PATH)
                    log("[INFO] token.json을 삭제했습니다.")

                log("[INFO] python main.py를 수동 실행해 OAuth 재인증을 진행하세요.")
                sys.exit(1)
        else:
            if not sys.stdin.isatty():
                log("[ERROR] OAuth 인증 토큰이 없습니다.")
                log("[INFO] python main.py를 수동 실행해 OAuth 재인증을 진행하세요.")
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

        with open(TOKEN_PATH, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def get_my_email(service) -> str:
    profile = service.users().getProfile(userId="me").execute()
    return profile["emailAddress"]


def find_label_id(service, label_name: str) -> str:
    labels = service.users().labels().list(userId="me").execute()

    for label in labels.get("labels", []):
        if label["name"] == label_name:
            return label["id"]

    raise RuntimeError(f"'{label_name}' 라벨을 찾을 수 없습니다.")


def decode_body(data: str) -> str:
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")


def extract_body(payload: dict) -> str:
    body = payload.get("body", {})

    if body.get("data"):
        content = decode_body(body["data"])

        if payload.get("mimeType") == "text/html":
            return BeautifulSoup(content, "html.parser").get_text("\n")

        return content

    parts = payload.get("parts", [])

    if payload.get("mimeType") == "multipart/alternative":
        candidates = []

        for part in parts:
            mime_type = part.get("mimeType", "")
            part_body = part.get("body", {})

            if part_body.get("data") and mime_type == "text/plain":
                candidates.append(decode_body(part_body["data"]))
            elif part_body.get("data") and mime_type == "text/html":
                content = decode_body(part_body["data"])
                candidates.append(
                    BeautifulSoup(content, "html.parser").get_text("\n")
                )
            elif part.get("parts"):
                nested = extract_body(part)
                if nested:
                    candidates.append(nested)

        return max(candidates, key=lambda value: len(value.strip()), default="")

    texts = []

    for part in parts:
        mime_type = part.get("mimeType", "")
        part_body = part.get("body", {})

        if part_body.get("data") and mime_type in {"text/plain", "text/html"}:
            content = decode_body(part_body["data"])

            if mime_type == "text/html":
                content = BeautifulSoup(content, "html.parser").get_text("\n")

            texts.append(content)

        if part.get("parts"):
            nested = extract_body(part)
            if nested:
                texts.append(nested)

    return "\n".join(texts)


def get_header(headers: list[dict], target: str) -> str:
    for header in headers:
        if header.get("name", "").lower() == target.lower():
            return header.get("value", "")

    return ""


def internal_date_to_datetime(message: dict) -> datetime:
    internal_ms = int(message["internalDate"])
    return datetime.fromtimestamp(internal_ms / 1000, tz=TIMEZONE)


def list_all_label_messages(service, label_id: str) -> list[dict]:
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


def load_full_message(service, message_id: str) -> dict:
    return (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )


def send_summary_email(
    service,
    to_email: str,
    subject: str,
    body_text: str,
    max_retries: int = 3,
):
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

        except Exception as exc:
            last_error = exc
            log(
                f"[WARN] Gmail send failed. "
                f"retry={attempt}/{max_retries}, error={repr(exc)}"
            )

            if attempt < max_retries:
                time.sleep(10 * attempt)

    raise last_error


def trash_message(service, message_id: str) -> None:
    service.users().messages().trash(
        userId="me",
        id=message_id,
    ).execute()
