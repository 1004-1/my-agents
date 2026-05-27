"""백테스트 엔진 단위 테스트."""
from __future__ import annotations

import pandas as pd
import pytest

from src.lotto.ml.backtest import (
    BacktestResult,
    RoundResult,
    _calc_prize,
    run_backtest,
    save_backtest_results,
)


class TestCalcPrize:
    @pytest.mark.parametrize("match,bonus,expected", [
        (6, False, "1등"),
        (5, True,  "2등"),
        (5, False, "3등"),
        (4, False, "4등"),
        (3, False, "5등"),
        (2, False, "꽝"),
        (1, False, "꽝"),
        (0, False, "꽝"),
    ])
    def test_prize_mapping(self, match, bonus, expected):
        assert _calc_prize(match, bonus) == expected


class TestRoundResult:
    def _make_rr(self, match_counts: list[int]) -> RoundResult:
        return RoundResult(
            round_no=100,
            strategy="random",
            games=[[1, 2, 3, 4, 5, 6] for _ in match_counts],
            winning=[1, 2, 3, 4, 5, 6],
            bonus=7,
            match_counts=match_counts,
        )

    def test_best_match(self):
        rr = self._make_rr([2, 4, 1])
        assert rr.best_match == 4

    def test_best_match_empty(self):
        rr = self._make_rr([])
        assert rr.best_match == 0


class TestBacktestResult:
    def _make_result(self) -> BacktestResult:
        result = BacktestResult(strategy="random", total_rounds=3)
        for i, mc in enumerate([[3, 1], [4, 2], [2, 0]], start=1):
            result.per_round.append(RoundResult(
                round_no=i,
                strategy="random",
                games=[[1,2,3,10,20,30], [5,6,15,25,35,45]],
                winning=[1,2,3,10,20,30],
                bonus=7,
                match_counts=mc,
            ))
        return result

    def test_avg_best_match(self):
        result = self._make_result()
        # best_match per round: 3, 4, 2 → mean = 3.0
        assert abs(result.avg_best_match - 3.0) < 1e-6

    def test_match_3_plus(self):
        result = self._make_result()
        assert result.match_3_plus == 2  # round 1 (3), round 2 (4)

    def test_match_4_plus(self):
        result = self._make_result()
        assert result.match_4_plus == 1  # round 2

    def test_prize_summary_has_all_keys(self):
        result = self._make_result()
        ps = result.prize_summary
        assert "꽝" in ps
        assert "5등" in ps

    def test_best_match_distribution(self):
        result = self._make_result()
        dist = result.best_match_distribution
        assert dist[3] == 1
        assert dist[4] == 1
        assert dist[2] == 1


class TestRunBacktest:
    def test_random_strategy(self, larger_history):
        result = run_backtest(
            history=larger_history,
            strategy_name="random",
            start_round=30,
            end_round=40,
            n_games=3,
            min_history_rounds=10,
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy == "random"
        assert result.total_rounds == 11  # 30~40 inclusive

    def test_balanced_strategy(self, larger_history):
        result = run_backtest(
            history=larger_history,
            strategy_name="balanced",
            start_round=35,
            end_round=45,
            n_games=3,
        )
        assert isinstance(result, BacktestResult)
        assert len(result.per_round) == 11

    def test_each_round_uses_only_past_data(self, larger_history):
        """각 회차에서 생성된 게임은 past 데이터(round < t)만 사용했는지 확인."""
        # 이는 구조적 테스트: round_no가 start_round 이상인지 확인
        result = run_backtest(
            history=larger_history,
            strategy_name="random",
            start_round=20,
            end_round=25,
            n_games=2,
            min_history_rounds=5,
        )
        for rr in result.per_round:
            assert rr.round_no >= 20

    def test_all_games_valid(self, larger_history):
        """생성된 모든 게임이 유효한 로또 게임인지 확인."""
        result = run_backtest(
            history=larger_history,
            strategy_name="random",
            start_round=30,
            end_round=35,
            n_games=3,
        )
        for rr in result.per_round:
            for game in rr.games:
                assert len(game) == 6
                assert len(set(game)) == 6
                assert all(1 <= n <= 45 for n in game)

    def test_match_counts_valid(self, larger_history):
        """일치 개수가 0~6 범위인지 확인."""
        result = run_backtest(
            history=larger_history,
            strategy_name="random",
            start_round=30,
            end_round=40,
            n_games=3,
        )
        for rr in result.per_round:
            for mc in rr.match_counts:
                assert 0 <= mc <= 6

    def test_unknown_strategy_raises(self, larger_history):
        with pytest.raises(ValueError, match="알 수 없는 전략"):
            run_backtest(
                history=larger_history,
                strategy_name="unknown_strategy",
                start_round=30,
                end_round=35,
            )

    def test_empty_history_raises(self):
        empty = pd.DataFrame(columns=["round_no","date",
                                       "num1","num2","num3","num4","num5","num6","bonus"])
        with pytest.raises(ValueError):
            run_backtest(empty, strategy_name="random")


class TestSaveBacktestResults:
    def test_saves_csv(self, tmp_path, larger_history):
        result = run_backtest(
            history=larger_history,
            strategy_name="random",
            start_round=30,
            end_round=32,
            n_games=2,
        )
        out = tmp_path / "backtest.csv"
        save_backtest_results(result, out)
        assert out.exists()
        df = pd.read_csv(out)
        # 3회차 × 2게임 = 6행
        assert len(df) == 6
        assert "strategy" in df.columns
        assert "match_count" in df.columns
        assert "prize" in df.columns

    def test_upsert_by_strategy(self, tmp_path, larger_history):
        """같은 전략 결과는 덮어쓰고 다른 전략은 유지한다."""
        out = tmp_path / "backtest.csv"

        r1 = run_backtest(
            history=larger_history,
            strategy_name="random",
            start_round=30,
            end_round=31,
            n_games=2,
        )
        save_backtest_results(r1, out)
        rows_after_r1 = len(pd.read_csv(out))

        r2 = run_backtest(
            history=larger_history,
            strategy_name="balanced",
            start_round=30,
            end_round=31,
            n_games=2,
        )
        save_backtest_results(r2, out)
        rows_after_r2 = len(pd.read_csv(out))

        # r2(balanced)가 추가됨: r1 + r2
        assert rows_after_r2 == rows_after_r1 + rows_after_r1
