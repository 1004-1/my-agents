"""회차 당첨번호 데이터 유효성 검증."""

from __future__ import annotations

from datetime import datetime

# 로또 6/45 규칙 상수
LOTTO_MIN = 1
LOTTO_MAX = 45
NUM_BALLS = 6
NUM_COLS = [f"num{i}" for i in range(1, NUM_BALLS + 1)]


def validate_draw(row: dict) -> tuple[bool, list[str]]:
    """단일 회차 딕셔너리의 유효성을 검증한다.

    검증 항목:
        1. round_no — 양의 정수
        2. num1~num6 — 각각 1~45 범위, 서로 중복 없음
        3. bonus — 1~45 범위, num1~num6과 중복 없음
        4. date — YYYY-MM-DD 형식

    Args:
        row: {"round_no": int, "date": str, "num1"~"num6": int, "bonus": int}

    Returns:
        (is_valid, errors)
        is_valid  — True이면 모든 검증 통과
        errors    — 실패한 항목 설명 목록 (성공 시 빈 리스트)
    """
    errors: list[str] = []

    # ── 1. round_no ──────────────────────────────────────────────────────
    try:
        rn = int(row["round_no"])
        if rn <= 0:
            errors.append(f"round_no는 양의 정수여야 합니다: {rn}")
    except (KeyError, TypeError, ValueError):
        errors.append(f"round_no가 없거나 숫자가 아닙니다: {row.get('round_no')!r}")
        rn = -1

    # ── 2. num1~num6 ─────────────────────────────────────────────────────
    nums: list[int] = []
    for col in NUM_COLS:
        try:
            n = int(row[col])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{col}가 없거나 숫자가 아닙니다: {row.get(col)!r}")
            continue
        if not (LOTTO_MIN <= n <= LOTTO_MAX):
            errors.append(f"{col}={n} 범위 오류 (1~45)")
        nums.append(n)

    if len(nums) == NUM_BALLS and len(set(nums)) != NUM_BALLS:
        dupes = [n for n in set(nums) if nums.count(n) > 1]
        errors.append(f"당첨번호에 중복이 있습니다: {dupes}")

    # ── 3. bonus ─────────────────────────────────────────────────────────
    try:
        bonus = int(row["bonus"])
    except (KeyError, TypeError, ValueError):
        errors.append(f"bonus가 없거나 숫자가 아닙니다: {row.get('bonus')!r}")
        bonus = -1

    if bonus != -1:
        if not (LOTTO_MIN <= bonus <= LOTTO_MAX):
            errors.append(f"bonus={bonus} 범위 오류 (1~45)")
        elif bonus in nums:
            errors.append(f"bonus={bonus}가 당첨번호와 중복입니다: {nums}")

    # ── 4. date ──────────────────────────────────────────────────────────
    date_str = row.get("date", "")
    try:
        datetime.strptime(str(date_str), "%Y-%m-%d")
    except ValueError:
        errors.append(f"date 형식 오류 (YYYY-MM-DD 필요): {date_str!r}")

    return len(errors) == 0, errors


def validate_dataframe(df) -> tuple[int, list[tuple[int, list[str]]]]:
    """DataFrame 전체를 행별로 검증한다.

    Returns:
        (valid_count, [(round_no, errors), ...])
    """
    import pandas as pd  # noqa: PLC0415

    valid_count = 0
    failures: list[tuple[int, list[str]]] = []

    for _, row in df.iterrows():
        ok, errs = validate_draw(row.to_dict())
        if ok:
            valid_count += 1
        else:
            failures.append((int(row.get("round_no", -1)), errs))

    return valid_count, failures
