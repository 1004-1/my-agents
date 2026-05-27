"""validator 단위 테스트."""

import pytest
import pandas as pd

from src.lotto.collector.validator import validate_draw, validate_dataframe


# ── 유효한 케이스 ─────────────────────────────────────────────────────────

class TestValidateDrawSuccess:
    def test_valid_typical_row(self):
        row = {
            "round_no": 1, "date": "2002-12-07",
            "num1": 10, "num2": 23, "num3": 29,
            "num4": 33, "num5": 37, "num6": 40, "bonus": 16,
        }
        ok, errs = validate_draw(row)
        assert ok is True
        assert errs == []

    def test_valid_boundary_numbers(self):
        """1과 45 경계값 테스트."""
        row = {
            "round_no": 100, "date": "2004-10-09",
            "num1": 1, "num2": 2, "num3": 3,
            "num4": 43, "num5": 44, "num6": 45, "bonus": 42,
        }
        ok, errs = validate_draw(row)
        assert ok is True, errs

    def test_valid_string_numbers_are_coerced(self):
        """문자열 숫자도 int로 변환 가능하면 통과해야 한다."""
        row = {
            "round_no": "5", "date": "2003-01-04",
            "num1": "16", "num2": "24", "num3": "29",
            "num4": "40", "num5": "41", "num6": "42", "bonus": "3",
        }
        ok, errs = validate_draw(row)
        assert ok is True, errs


# ── round_no 오류 ──────────────────────────────────────────────────────────

class TestValidateRoundNo:
    def test_round_no_zero(self):
        row = {
            "round_no": 0, "date": "2002-12-07",
            "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6, "bonus": 7,
        }
        ok, errs = validate_draw(row)
        assert not ok
        assert any("round_no" in e for e in errs)

    def test_round_no_negative(self):
        row = {
            "round_no": -1, "date": "2002-12-07",
            "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6, "bonus": 7,
        }
        ok, errs = validate_draw(row)
        assert not ok

    def test_round_no_missing(self):
        row = {
            "date": "2002-12-07",
            "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6, "bonus": 7,
        }
        ok, errs = validate_draw(row)
        assert not ok


# ── 번호 범위 오류 ────────────────────────────────────────────────────────

class TestValidateNumbers:
    def test_num_out_of_range_zero(self):
        row = {
            "round_no": 1, "date": "2002-12-07",
            "num1": 0, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6, "bonus": 7,
        }
        ok, errs = validate_draw(row)
        assert not ok
        assert any("num1" in e for e in errs)

    def test_num_out_of_range_46(self):
        row = {
            "round_no": 1, "date": "2002-12-07",
            "num1": 46, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6, "bonus": 7,
        }
        ok, errs = validate_draw(row)
        assert not ok

    def test_duplicate_numbers(self):
        row = {
            "round_no": 1, "date": "2002-12-07",
            "num1": 5, "num2": 5, "num3": 3, "num4": 4, "num5": 6, "num6": 7, "bonus": 8,
        }
        ok, errs = validate_draw(row)
        assert not ok
        assert any("중복" in e for e in errs)

    def test_missing_num_column(self):
        row = {
            "round_no": 1, "date": "2002-12-07",
            "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5,
            # num6 누락
            "bonus": 7,
        }
        ok, errs = validate_draw(row)
        assert not ok


# ── 보너스 번호 오류 ──────────────────────────────────────────────────────

class TestValidateBonus:
    def test_bonus_duplicates_main(self):
        row = {
            "round_no": 1, "date": "2002-12-07",
            "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6,
            "bonus": 3,   # num3과 중복
        }
        ok, errs = validate_draw(row)
        assert not ok
        assert any("bonus" in e for e in errs)

    def test_bonus_out_of_range(self):
        row = {
            "round_no": 1, "date": "2002-12-07",
            "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6,
            "bonus": 46,
        }
        ok, errs = validate_draw(row)
        assert not ok

    def test_bonus_missing(self):
        row = {
            "round_no": 1, "date": "2002-12-07",
            "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6,
        }
        ok, errs = validate_draw(row)
        assert not ok


# ── 날짜 형식 오류 ────────────────────────────────────────────────────────

class TestValidateDate:
    def test_wrong_date_format(self):
        row = {
            "round_no": 1, "date": "2002/12/07",   # 잘못된 구분자
            "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6, "bonus": 7,
        }
        ok, errs = validate_draw(row)
        assert not ok
        assert any("date" in e for e in errs)

    def test_invalid_date_value(self):
        row = {
            "round_no": 1, "date": "2002-13-07",   # 13월
            "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6, "bonus": 7,
        }
        ok, errs = validate_draw(row)
        assert not ok


# ── validate_dataframe ───────────────────────────────────────────────────

class TestValidateDataframe:
    def test_all_valid(self, sample_history):
        valid_count, failures = validate_dataframe(sample_history)
        assert valid_count == len(sample_history)
        assert failures == []

    def test_partial_invalid(self, sample_history):
        bad_row = {
            "round_no": 999, "date": "2024-01-01",
            "num1": 0, "num2": 2, "num3": 3,   # num1=0 → 오류
            "num4": 4, "num5": 5, "num6": 6, "bonus": 7,
        }
        df = pd.concat(
            [sample_history, pd.DataFrame([bad_row])],
            ignore_index=True,
        )
        valid_count, failures = validate_dataframe(df)
        assert valid_count == len(sample_history)
        assert len(failures) == 1
        assert failures[0][0] == 999
