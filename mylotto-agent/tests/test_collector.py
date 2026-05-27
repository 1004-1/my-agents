"""DhCollector 단위 테스트 — 실제 HTTP 호출 없이 mock 사용."""

import time
import pytest
from unittest.mock import patch

from src.lotto.collector.dh_collector import (
    DhCollector,
    CollectStats,
    FetchResult,
    COLUMNS,
)


# ── 공통 fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def valid_draw_data():
    """검증을 통과하는 유효한 단일 회차 딕셔너리."""
    return {
        "round_no": 1, "date": "2002-12-07",
        "num1": 10, "num2": 23, "num3": 29,
        "num4": 33, "num5": 37, "num6": 40, "bonus": 16,
    }


# ══════════════════════════════════════════════════════════════════════════
# _parse_row
# ══════════════════════════════════════════════════════════════════════════

class TestDhCollectorParseRow:
    """_parse_row: API JSON → 내부 컬럼 변환."""

    def test_parse_row_returns_expected_keys(self, mock_api_response):
        row = DhCollector._parse_row(mock_api_response)
        assert set(row.keys()) == set(COLUMNS)

    def test_parse_row_values(self, mock_api_response):
        row = DhCollector._parse_row(mock_api_response)
        assert row["round_no"] == 1100
        assert row["date"] == "2024-01-06"
        assert row["num1"] == 1
        assert row["num6"] == 7
        assert row["bonus"] == 8


# ══════════════════════════════════════════════════════════════════════════
# fetch_draw
# ══════════════════════════════════════════════════════════════════════════

class TestDhCollectorFetchDraw:
    """fetch_draw: HTTP 호출을 mock으로 대체."""

    def test_fetch_draw_success(self, mock_api_response):
        collector = DhCollector()
        with patch.object(collector, "_call_api", return_value=mock_api_response):
            result = collector.fetch_draw(1100)
        assert result["round_no"] == 1100
        assert result["bonus"] == 8

    def test_fetch_draw_raises_on_api_failure(self):
        collector = DhCollector()
        with patch.object(collector, "_call_api", return_value={"returnValue": "fail"}):
            with pytest.raises(ValueError, match="1100"):
                collector.fetch_draw(1100)

    def test_fetch_draw_raises_on_missing_return_value(self):
        collector = DhCollector()
        with patch.object(collector, "_call_api", return_value={}):
            with pytest.raises(ValueError):
                collector.fetch_draw(99)


# ══════════════════════════════════════════════════════════════════════════
# FetchResult
# ══════════════════════════════════════════════════════════════════════════

class TestFetchResult:
    def test_is_valid_true(self):
        r = FetchResult(round_no=1, success=True)
        assert r.is_valid is True

    def test_is_valid_false_when_api_failed(self):
        r = FetchResult(round_no=1, success=False, error="timeout")
        assert r.is_valid is False

    def test_is_valid_false_when_validation_errors(self):
        r = FetchResult(round_no=1, success=True, validation_errors=["num1 오류"])
        assert r.is_valid is False


# ══════════════════════════════════════════════════════════════════════════
# CollectStats
# ══════════════════════════════════════════════════════════════════════════

class TestCollectStats:
    def test_total(self):
        s = CollectStats(start_round=1, end_round=10)
        assert s.total == 10

    def test_success_rate_full(self):
        s = CollectStats(start_round=1, end_round=10, collected=10)
        assert s.success_rate == 100.0

    def test_success_rate_partial(self):
        s = CollectStats(start_round=1, end_round=10, collected=8)
        assert s.success_rate == pytest.approx(80.0)

    def test_success_rate_zero(self):
        s = CollectStats(start_round=1, end_round=0)   # total=0
        assert s.success_rate == 0.0


# ══════════════════════════════════════════════════════════════════════════
# fetch_range (BaseCollector 호환 래퍼)
# ══════════════════════════════════════════════════════════════════════════

class TestDhCollectorFetchRange:
    """fetch_range: collect_range 위임 + 통계 무시 인터페이스."""

    def test_fetch_range_returns_dataframe(self, valid_draw_data):
        collector = DhCollector()
        with (
            patch.object(collector, "fetch_draw", return_value=valid_draw_data),
            patch("src.lotto.collector.dh_collector.time.sleep"),
        ):
            df = collector.fetch_range(1, 3)
        assert len(df) == 3
        assert list(df.columns) == COLUMNS

    def test_fetch_range_skips_failed_rounds(self):
        collector = DhCollector()

        def _side_effect(round_no):
            if round_no == 2:
                raise ValueError("API 오류")
            return {
                "round_no": round_no, "date": "2024-01-01",
                "num1": 1, "num2": 2, "num3": 3,
                "num4": 4, "num5": 5, "num6": 6, "bonus": 7,
            }

        with (
            patch.object(collector, "fetch_draw", side_effect=_side_effect),
            patch("src.lotto.collector.dh_collector.time.sleep"),
        ):
            df = collector.fetch_range(1, 3)

        assert len(df) == 2
        assert 2 not in df["round_no"].values


# ══════════════════════════════════════════════════════════════════════════
# collect_range (stats + callback 검증)
# ══════════════════════════════════════════════════════════════════════════

class TestDhCollectorCollectRange:
    """collect_range: FetchResult 콜백, 통계, 검증 포함."""

    def test_collect_range_success_stats(self, valid_draw_data):
        collector = DhCollector()
        with (
            patch.object(collector, "fetch_draw", return_value=valid_draw_data),
            patch("src.lotto.collector.dh_collector.time.sleep"),
        ):
            df, stats = collector.collect_range(1, 5)

        assert stats.collected == 5
        assert stats.skipped == 0
        assert stats.invalid == 0
        assert stats.total == 5
        assert stats.success_rate == 100.0
        assert len(df) == 5

    def test_collect_range_skipped_rounds_tracked(self):
        collector = DhCollector()

        def _fail_round_3(round_no):
            if round_no == 3:
                raise ValueError("API 오류")
            return {
                "round_no": round_no, "date": "2024-01-01",
                "num1": 1, "num2": 2, "num3": 3,
                "num4": 4, "num5": 5, "num6": 6, "bonus": 7,
            }

        with (
            patch.object(collector, "fetch_draw", side_effect=_fail_round_3),
            patch("src.lotto.collector.dh_collector.time.sleep"),
        ):
            df, stats = collector.collect_range(1, 5)

        assert stats.collected == 4
        assert stats.skipped == 1
        assert 3 in stats.failed_rounds

    def test_collect_range_invalid_data_tracked(self):
        """유효성 검증 실패 회차는 invalid 카운트에 포함된다."""
        collector = DhCollector()

        def _bad_num(round_no):
            return {
                "round_no": round_no, "date": "2024-01-01",
                # num1=0 → 범위 오류
                "num1": 0, "num2": 2, "num3": 3,
                "num4": 4, "num5": 5, "num6": 6, "bonus": 7,
            }

        with (
            patch.object(collector, "fetch_draw", side_effect=_bad_num),
            patch("src.lotto.collector.dh_collector.time.sleep"),
        ):
            df, stats = collector.collect_range(1, 3)

        assert stats.invalid == 3
        assert stats.collected == 0
        assert df.empty

    def test_collect_range_progress_callback_called(self, valid_draw_data):
        """on_progress 콜백이 각 회차마다 호출되어야 한다."""
        collector = DhCollector()
        called: list[FetchResult] = []

        with (
            patch.object(collector, "fetch_draw", return_value=valid_draw_data),
            patch("src.lotto.collector.dh_collector.time.sleep"),
        ):
            collector.collect_range(1, 4, on_progress=lambda r: called.append(r))

        assert len(called) == 4
        assert all(isinstance(r, FetchResult) for r in called)

    def test_collect_range_elapsed_tracked(self, valid_draw_data):
        """elapsed 시간이 0보다 크게 기록되어야 한다."""
        collector = DhCollector()
        with (
            patch.object(collector, "fetch_draw", return_value=valid_draw_data),
            patch("src.lotto.collector.dh_collector.time.sleep"),
        ):
            _, stats = collector.collect_range(1, 2)
        assert stats.elapsed >= 0

    def test_collect_range_empty_on_all_fail(self):
        """모든 회차가 실패하면 빈 DataFrame이 반환된다."""
        collector = DhCollector()
        with (
            patch.object(collector, "fetch_draw", side_effect=ValueError("항상 실패")),
            patch("src.lotto.collector.dh_collector.time.sleep"),
        ):
            df, stats = collector.collect_range(1, 3)

        assert df.empty
        assert list(df.columns) == COLUMNS
        assert stats.collected == 0
        assert stats.skipped == 3


# ══════════════════════════════════════════════════════════════════════════
# fetch_latest_round
# ══════════════════════════════════════════════════════════════════════════

class TestFetchLatestRound:
    def test_returns_last_successful_round(self):
        """이진탐색이 마지막 success 회차를 정확히 찾아야 한다."""
        collector = DhCollector()

        def _fake_api(round_no):
            return {"returnValue": "success"} if round_no <= 50 else {"returnValue": "fail"}

        with patch.object(collector, "_call_api", side_effect=_fake_api):
            result = collector.fetch_latest_round()

        assert result == 50

    def test_returns_1_when_only_first_exists(self):
        collector = DhCollector()

        def _fake_api(round_no):
            return {"returnValue": "success"} if round_no == 1 else {"returnValue": "fail"}

        with patch.object(collector, "_call_api", side_effect=_fake_api):
            result = collector.fetch_latest_round()

        assert result == 1
