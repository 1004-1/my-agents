"""ResultChecker 단위 테스트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.lotto.analysis.result_checker import ResultChecker, get_rank, get_reward_estimate


# ──────────────────────────────────────────────────────────────────────────────
# get_rank / get_reward_estimate
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("match,bonus,expected", [
    (6, False, 1),
    (6, True,  1),   # 6개 일치 → 보너스 무관하게 1등
    (5, True,  2),
    (5, False, 3),
    (4, False, 4),
    (4, True,  4),
    (3, False, 5),
    (3, True,  5),
    (2, False, None),
    (1, True,  None),
    (0, False, None),
])
def test_get_rank(match, bonus, expected):
    assert get_rank(match, bonus) == expected


def test_get_reward_estimate():
    assert get_reward_estimate(1) == "jackpot"
    assert get_reward_estimate(2) == "varies"
    assert get_reward_estimate(3) == 1_500_000
    assert get_reward_estimate(4) == 50_000
    assert get_reward_estimate(5) == 5_000
    assert get_reward_estimate(None) == 0


# ──────────────────────────────────────────────────────────────────────────────
# ResultChecker.find_target_round
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def draws_df():
    data = [
        {"round_no": 100, "date": "2026-01-10", "draw_date": pd.Timestamp("2026-01-10").date()},
        {"round_no": 101, "date": "2026-01-17", "draw_date": pd.Timestamp("2026-01-17").date()},
        {"round_no": 102, "date": "2026-01-24", "draw_date": pd.Timestamp("2026-01-24").date()},
    ]
    return pd.DataFrame(data)


def test_find_target_round_normal(draws_df):
    checker = ResultChecker()
    # 1월 11일 생성 → 가장 가까운 미래 추첨 = 101회
    assert checker.find_target_round("2026-01-11", draws_df) == 101


def test_find_target_round_before_all(draws_df):
    checker = ResultChecker()
    # 1월 5일 생성 → 가장 가까운 미래 추첨 = 100회
    assert checker.find_target_round("2026-01-05", draws_df) == 100


def test_find_target_round_no_future(draws_df):
    checker = ResultChecker()
    # 1월 25일 이후 → 미래 추첨 없음
    assert checker.find_target_round("2026-01-25", draws_df) is None


def test_find_target_round_same_day(draws_df):
    checker = ResultChecker()
    # 1월 10일 당일 생성 → 당일은 포함 안 됨 → 101회
    assert checker.find_target_round("2026-01-10", draws_df) == 101


# ──────────────────────────────────────────────────────────────────────────────
# ResultChecker.check_all  (파일 I/O를 tmp_path로 격리)
# ──────────────────────────────────────────────────────────────────────────────

def _write_draws(path: Path) -> None:
    pd.DataFrame([
        {"round_no": 100, "date": "2026-01-10",
         "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6, "bonus": 7},
        {"round_no": 101, "date": "2026-01-17",
         "num1": 10, "num2": 20, "num3": 30, "num4": 40, "num5": 41, "num6": 42, "bonus": 15},
    ]).to_csv(path, index=False)


def _write_games(path: Path) -> None:
    pd.DataFrame([
        # 100회 당첨번호와 3개 일치 (5등)
        {
            "generated_at": "2026-01-05T09:00:00+00:00",
            "strategy": "random", "game_no": 1,
            "num1": 1, "num2": 2, "num3": 3, "num4": 11, "num5": 22, "num6": 33,
            "purchased": False, "purchased_at": "",
        },
        # 101회 당첨번호와 0개 일치
        {
            "generated_at": "2026-01-11T09:00:00+00:00",
            "strategy": "balanced", "game_no": 1,
            "num1": 5, "num2": 6, "num3": 7, "num4": 8, "num5": 9, "num6": 11,
            "purchased": False, "purchased_at": "",
        },
        # 추첨이 없는 미래 날짜 → 스킵
        {
            "generated_at": "2026-02-01T09:00:00+00:00",
            "strategy": "random", "game_no": 1,
            "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6,
            "purchased": False, "purchased_at": "",
        },
    ]).to_csv(path, index=False)


def test_check_all_basic(tmp_path):
    draws  = tmp_path / "draws.csv"
    games  = tmp_path / "games.csv"
    pred   = tmp_path / "pred.csv"
    _write_draws(draws)
    _write_games(games)

    checker = ResultChecker(
        games_path=games,
        results_path=draws,
        prediction_path=pred,
    )
    df, added, skipped = checker.check_all()

    assert added == 2
    assert skipped == 1
    assert len(df) == 2
    assert pred.exists()

    # 첫 번째 게임: 3개 일치 → 5등
    row5 = df[df["strategy_name"] == "random"].iloc[0]
    assert int(row5["match_count"]) == 3
    assert str(row5["rank"]) == "5"
    assert row5["reward_estimate"] == 5000

    # 두 번째 게임: 0개 일치 → 낙첨
    row0 = df[df["strategy_name"] == "balanced"].iloc[0]
    assert int(row0["match_count"]) == 0
    assert str(row0["rank"]) == ""


def test_check_all_no_duplicate(tmp_path):
    """이미 체크된 게임은 재추가되지 않아야 한다."""
    draws  = tmp_path / "draws.csv"
    games  = tmp_path / "games.csv"
    pred   = tmp_path / "pred.csv"
    _write_draws(draws)
    _write_games(games)

    checker = ResultChecker(games_path=games, results_path=draws, prediction_path=pred)
    checker.check_all()
    df2, added2, _ = checker.check_all()

    assert added2 == 0
    assert len(df2) == 2


def test_check_all_no_games(tmp_path):
    draws = tmp_path / "draws.csv"
    games = tmp_path / "games.csv"
    pred  = tmp_path / "pred.csv"
    _write_draws(draws)
    # games 파일 없음

    checker = ResultChecker(games_path=games, results_path=draws, prediction_path=pred)
    df, added, skipped = checker.check_all()

    assert df.empty
    assert added == 0
    assert skipped == 0


# ──────────────────────────────────────────────────────────────────────────────
# ResultChecker.strategy_summary
# ──────────────────────────────────────────────────────────────────────────────

def test_strategy_summary(tmp_path):
    pred = tmp_path / "pred.csv"
    pd.DataFrame([
        {"generated_at": "2026-01-05", "target_round_no": 100,
         "strategy_name": "random", "numbers": "1,2,3,4,5,6",
         "winning_numbers": "1,2,3,7,8,9", "bonus": 10,
         "match_count": 3, "bonus_matched": False, "rank": 5,
         "reward_estimate": 5000, "checked_at": "2026-01-11"},
        {"generated_at": "2026-01-05", "target_round_no": 100,
         "strategy_name": "random", "numbers": "1,2,4,5,6,7",
         "winning_numbers": "1,2,3,7,8,9", "bonus": 10,
         "match_count": 4, "bonus_matched": False, "rank": 4,
         "reward_estimate": 50000, "checked_at": "2026-01-11"},
        {"generated_at": "2026-01-11", "target_round_no": 101,
         "strategy_name": "balanced", "numbers": "10,20,30,40,41,42",
         "winning_numbers": "10,20,30,40,41,42", "bonus": 15,
         "match_count": 6, "bonus_matched": False, "rank": 1,
         "reward_estimate": "jackpot", "checked_at": "2026-01-18"},
    ]).to_csv(pred, index=False)

    checker = ResultChecker(prediction_path=pred)
    summary = checker.strategy_summary()

    assert len(summary) == 2
    balanced_row = summary[summary["strategy"] == "balanced"].iloc[0]
    assert balanced_row["max_match"] == 6
    assert balanced_row["rank_1"] == 1

    random_row = summary[summary["strategy"] == "random"].iloc[0]
    assert random_row["total_games"] == 2
    assert random_row["match_3plus"] == 2
    assert random_row["match_4plus"] == 1
    assert random_row["rank_5"] == 1
    assert random_row["rank_4"] == 1


def test_strategy_summary_empty(tmp_path):
    pred = tmp_path / "pred.csv"
    checker = ResultChecker(prediction_path=pred)
    summary = checker.strategy_summary()
    assert summary.empty
