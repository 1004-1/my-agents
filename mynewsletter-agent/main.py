import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.config import DRY_RUN_DELETE, LABEL_NAME, LANG_KOREAN, MAX_REPORT_CHARS, TIMEZONE
from src.gmail_client import (
    extract_body,
    find_label_id,
    get_header,
    get_my_email,
    get_service,
    internal_date_to_datetime,
    list_all_label_messages,
    load_full_message,
    send_summary_email,
    trash_message,
)
from src.logutil import log
from src.summarizer import summarize
from src.text_utils import clean_text, detect_language


def truncate_report(parts: list[str]) -> str:
    body = "\n".join(parts)

    if len(body) > MAX_REPORT_CHARS:
        body = body[:MAX_REPORT_CHARS]
        body += "\n\n[알림] 요약 메일 본문이 너무 길어 일부가 잘렸습니다."

    return body


def main():
    job_start_time = time.time()

    service = get_service()
    my_email = get_my_email(service)
    label_id = find_label_id(service, LABEL_NAME)

    log(f"내 Gmail: {my_email}")
    log(f"Meco Label ID: {label_id}")
    log(f"DRY_RUN_DELETE: {DRY_RUN_DELETE}")

    candidates = list_all_label_messages(service, label_id)
    loaded_messages = []

    for item in candidates:
        try:
            loaded_messages.append(load_full_message(service, item["id"]))
        except Exception as exc:
            log(
                f"[WARN] 메일 로드 실패. "
                f"id={item['id']}, error={repr(exc)}"
            )

    log(f"전체 Meco 메일 수: {len(candidates)}")
    log(f"정상 로드 메일 수: {len(loaded_messages)}")

    korean_parts = []
    foreign_parts = []

    today = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
    run_time = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    for report_parts, label in [
        (korean_parts, "한글"),
        (foreign_parts, "영문"),
    ]:
        report_parts.extend(
            [
                f"# 뉴스레터 요약 ({label})",
                "",
                f"- 실행 시각: {run_time}",
            ]
        )

    korean_count = 0
    foreign_count = 0
    processed_messages = []
    failed_messages = []

    for index, message in enumerate(loaded_messages, start=1):
        headers = message["payload"].get("headers", [])
        subject = get_header(headers, "Subject")
        sender = get_header(headers, "From")
        body = extract_body(message["payload"])
        cleaned_body = clean_text(body)
        language = detect_language(subject, body)
        language_label = "한글" if language == LANG_KOREAN else "영문"

        log("")
        log("=" * 20)
        log(f"[{index}/{len(loaded_messages)}] [{language_label}]")
        log(f"제목: {subject}")
        log(f"발신: {sender}")
        log(f"[입력 원문 시작] ({len(cleaned_body)}자)\n{cleaned_body}\n[입력 원문 끝]")

        started_at = time.time()
        log(f"[START] process: {subject}")

        try:
            result, used_model = summarize(
                subject,
                sender,
                cleaned_body,
                language,
            )
        except Exception as exc:
            elapsed = time.time() - started_at
            log(
                f"[ERROR] 처리 실패: {subject} / "
                f"elapsed={elapsed:.1f}s / error={repr(exc)}"
            )
            failed_messages.append(message)
            continue

        elapsed = time.time() - started_at
        log(
            f"[DONE] process: {subject} / "
            f"model={used_model} / elapsed={elapsed:.1f}s"
        )
        log(f"[처리 결과 시작] ({len(result)}자, model={used_model})\n{result}\n[처리 결과 끝]")

        if language == LANG_KOREAN:
            korean_count += 1
            report_parts = korean_parts
            item_number = korean_count
        else:
            foreign_count += 1
            report_parts = foreign_parts
            item_number = foreign_count

        report_parts.extend(
            [
                "=" * 20,
                f"## {item_number}. {subject}",
                "",
                f"- 발신: {sender}",
                "",
                result,
                "",
            ]
        )

        processed_messages.append(message)

    korean_parts.insert(3, f"- 처리 메일 수: {korean_count}")
    korean_parts.insert(4, "")
    foreign_parts.insert(3, f"- 처리 메일 수: {foreign_count}")
    foreign_parts.insert(4, "")

    sent_any_report = False
    send_started_at = time.time()

    if korean_count > 0:
        log("[START] send Korean summary email")
        send_summary_email(
            service=service,
            to_email=my_email,
            subject=f"[뉴스레터 요약 - 한글] {today}",
            body_text=truncate_report(korean_parts),
        )
        log(f"[DONE] 한글 요약 메일 발송 완료: {my_email}")
        sent_any_report = True
    else:
        log("한글 뉴스레터 없음 — 발송 생략")

    if foreign_count > 0:
        log("[START] send translated newsletter email")
        send_summary_email(
            service=service,
            to_email=my_email,
            subject=f"[뉴스레터 번역 - 영문] {today}",
            body_text=truncate_report(foreign_parts),
        )
        log(f"[DONE] 영문 번역 메일 발송 완료: {my_email}")
        sent_any_report = True
    else:
        log("영문 뉴스레터 없음 — 발송 생략")

    send_elapsed = time.time() - send_started_at
    log(f"[DONE] send all reports / elapsed={send_elapsed:.1f}s")

    if sent_any_report:
        log("휴지통 이동 처리 시작")

        for message in processed_messages:
            headers = message["payload"].get("headers", [])
            subject = get_header(headers, "Subject")
            message_id = message["id"]
            received_at = internal_date_to_datetime(message)

            if DRY_RUN_DELETE:
                log(
                    f"[DRY RUN] 휴지통 이동 예정: "
                    f"{received_at:%Y-%m-%d %H:%M:%S} / {subject}"
                )
                continue

            try:
                trash_message(service, message_id)
                log(
                    f"휴지통 이동 완료: "
                    f"{received_at:%Y-%m-%d %H:%M:%S} / {subject}"
                )
            except Exception as exc:
                log(
                    f"[WARN] 휴지통 이동 실패. "
                    f"id={message_id}, subject={subject}, error={repr(exc)}"
                )
    else:
        log("[WARN] 발송된 보고서가 없어 원본 메일을 이동하지 않습니다.")

    if failed_messages:
        log(
            f"[WARN] 처리 실패 메일 {len(failed_messages)}건은 "
            f"원본 라벨에 남겨두었습니다."
        )

    total_elapsed = time.time() - job_start_time
    log(f"완료 / total_elapsed={total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
