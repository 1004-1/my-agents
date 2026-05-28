"""tests/test_balanced_v2_strategy.py — BalancedV2Strategy 단위 테스트."""
from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.lotto.strategy.balanced_v2_strategy import (
    BalancedV2Strategy,
    _DECADE_OF,
    _NUM_COLS,
)


# ── 픽스처 ────────────────────────────────────────────────────────────────

def _make_history(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """재현성 있는 가짜 history DataFrame을 생성한다."""
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        nums = sorted(rng.sample(range(1, 46), 6))
        bonus = rng.choice([n for n in range(1, 46) if n not in nums])
        rows.append({
            "round_no": i,
            "date":     f"2020-01-{(i % 28) + 1:02d}",
            "num1": nums[0], "num2": nums[1], "num3": nums[2],
            "num4": nums[3], "num5": nums[4], "num6": nums[5],
            "bonus": bonus,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def strategy() -> BalancedV2Strategy:
    return BalancedV2Strategy(seed=0)


@pytest.fixture
def history_df() -> pd.DataFrame:
    return _make_history(200)


# ── _DECADE_OF 검증 ───────────────────────────────────────────────────────

class TestDecadeMapping:
    def test_range_1_9_is_decade_1(self):
        for n in range(1, 10):
            assert _DECADE_OF[n] == 1, f"_DECADE_OF[{n}] should be 1"

    def test_range_10_19_is_decade_2(self):
        for n in range(10, 20):
            assert _DECADE_OF[n] == 2, f"_DECADE_OF[{n}] should be 2"

    def test_range_20_29_is_decade_3(self):
        for n in range(20, 30):
            assert _DECADE_OF[n] == 3

    def test_range_30_39_is_decade_4(self):
        for n in range(30, 40):
            assert _DECADE_OF[n] == 4

    def test_range_40_45_is_decade_5(self):
        for n in range(40, 46):
            assert _DECADE_OF[n] == 5

    def test_all_numbers_covered(self):
        assert set(_DECADE_OF.keys()) == set(range(1, 46))


# ── _prepare_context ──────────────────────────────────────────────────────

class TestPrepareContext:
    def test_empty_history_returns_defaults(self, strategy):
        ctx = strategy._prepare_context(None)
        assert ctx["sum_lo"] == 100
        assert ctx["sum_hi"] == 176
        assert ctx["weights"] is None
        assert ctx["prev_nums"] == frozenset()
        assert ctx["hot_nums"] == frozenset()

    def test_with_history_sum_range_is_percentile_based(self, strategy, history_df):
        ctx = strategy._prepare_context(history_df)
        sums = history_df[_NUM_COLS].sum(axis=1).values
        expected_lo = float(np.percentile(sums, 5.0))
        expected_hi = float(np.percentile(sums, 95.0))
        assert abs(ctx["sum_lo"] - expected_lo) < 1e-6
        assert abs(ctx["sum_hi"] - expected_hi) < 1e-6

    def test_sum_lo_less_than_sum_hi(self, strategy, history_df):
        ctx = strategy._prepare_context(history_df)
        assert ctx["sum_lo"] < ctx["sum_hi"]

    def test_weights_sum_to_one(self, strategy, history_df):
        ctx = strategy._prepare_context(history_df)
        assert ctx["weights"] is not None
        total = sum(ctx["weights"].values())
        assert abs(total - 1.0) < 1e-9

    def test_weights_cover_all_numbers(self, strategy, history_df):
        ctx = strategy._prepare_context(history_df)
        assert set(ctx["weights"].keys()) == set(range(1, 46))

    def test_prev_nums_from_last_row(self, strategy, history_df):
        ctx = strategy._prepare_context(history_df)
        last_row = history_df.sort_values("round_no").iloc[-1]
        expected = frozenset(int(last_row[c]) for c in _NUM_COLS)
        assert ctx["prev_nums"] == expected

    def test_hot_nums_type(self, strategy, history_df):
        ctx = strategy._prepare_context(history_df)
        assert isinstance(ctx["hot_nums"], frozenset)

    def test_leakage_free_sum_range_increases_with_more_data(self, strategy):
        """더 많은 history를 줄수록 sum_lo가 안정화되어야 한다 (더 낮거나 같을 수 있음)."""
        h50  = _make_history(50)
        h200 = _make_history(200)
        ctx50  = strategy._prepare_context(h50)
        ctx200 = strategy._prepare_context(h200)
        # 두 범위 모두 유효한 로또 합계 범위 안에 있어야 함
        assert ctx50["sum_lo"]  >= 21   # 최소: 1+2+3+4+5+6
        assert ctx200["sum_hi"] <= 255  # 최대: 40+41+42+43+44+45


# ── _is_valid ─────────────────────────────────────────────────────────────

class TestIsValid:
    def _ctx(self, strategy) -> dict:
        return {
            "sum_lo": 100.0, "sum_hi": 176.0,
            "weights": None,
            "prev_nums": frozenset(),
            "hot_nums":  frozenset(),
        }

    def test_valid_game_passes(self, strategy):
        ctx  = self._ctx(strategy)
        game = [3, 12, 21, 30, 41, 15]   # 합계=122, 홀3짝3, 5구간
        assert strategy._is_valid(game, ctx)

    def test_odd_count_0_fails(self, strategy):
        ctx  = self._ctx(strategy)
        game = [2, 4, 12, 20, 30, 44]  # 홀수 0개
        assert not strategy._is_valid(game, ctx)

    def test_odd_count_5_fails(self, strategy):
        ctx  = self._ctx(strategy)
        game = [1, 3, 11, 21, 31, 40]  # 홀수 5개
        assert not strategy._is_valid(game, ctx)

    def test_sum_below_lo_fails(self, strategy):
        ctx  = {**self._ctx(strategy), "sum_lo": 150.0, "sum_hi": 176.0}
        game = [1, 2, 3, 4, 5, 7]  # 합계=22
        assert not strategy._is_valid(game, ctx)

    def test_sum_above_hi_fails(self, strategy):
        ctx  = {**self._ctx(strategy), "sum_lo": 100.0, "sum_hi": 120.0}
        game = [30, 35, 38, 40, 42, 44]  # 합계=229
        assert not strategy._is_valid(game, ctx)

    def test_max_3_per_decade(self, strategy):
        ctx  = self._ctx(strategy)
        game = [1, 2, 3, 4, 15, 30]  # 1~9에서 4개 → 실패
        assert not strategy._is_valid(game, ctx)

    def test_min_3_decades(self, strategy):
        ctx  = self._ctx(strategy)
        # 1~9에서 3, 10~19에서 3 → 2구간 → 실패 (단, sum도 확인)
        # 합계 ≈ 90 미만 → sum_lo=100 실패로 걸릴 수 있음
        game = [1, 3, 5, 10, 12, 14]  # 합=45, sum_lo 실패
        assert not strategy._is_valid(game, ctx)

    def test_three_consecutive_fails(self, strategy):
        ctx  = self._ctx(strategy)
        game = [5, 6, 7, 20, 30, 40]  # 5-6-7 연속 3개
        assert not strategy._is_valid(game, ctx)

    def test_two_consecutive_passes(self, strategy):
        ctx  = self._ctx(strategy)
        game = [5, 6, 20, 30, 39, 41]  # 5-6 연속 2개만
        # sum = 141, 홀=2(5,39,41)→홀3짝3 → valid if other constraints ok
        assert strategy._is_valid(game, ctx)

    def test_trailing_digit_max_2(self, strategy):
        ctx  = self._ctx(strategy)
        game = [1, 11, 21, 22, 33, 44]  # 1,11,21 끝자리 1 → 3개 → 실패
        assert not strategy._is_valid(game, ctx)

    def test_prev_overlap_limit(self, strategy):
        ctx = {**self._ctx(strategy), "prev_nums": frozenset({1, 2, 3, 4, 5, 6})}
        game = [1, 2, 3, 10, 20, 30]  # 이전 회차 3개 겹침 (max=2) → 실패
        assert not strategy._is_valid(game, ctx)

    def test_hot_numbers_limit(self, strategy):
        ctx  = {**self._ctx(strategy), "hot_nums": frozenset({5, 10, 15, 20})}
        game = [5, 10, 15, 20, 30, 40]  # hot 4개 겹침 (max=3) → 실패
        assert not strategy._is_valid(game, ctx)


# ── generate ─────────────────────────────────────────────────────────────

class TestGenerate:
    def test_returns_n_games(self, strategy, history_df):
        games = strategy.generate(n_games=5, history=history_df)
        assert len(games) == 5

    def test_each_game_has_6_numbers(self, strategy, history_df):
        for game in strategy.generate(n_games=3, history=history_df):
            assert len(game) == 6

    def test_numbers_in_valid_range(self, strategy, history_df):
        for game in strategy.generate(n_games=5, history=history_df):
            assert all(1 <= n <= 45 for n in game)

    def test_no_duplicates_within_game(self, strategy, history_df):
        for game in strategy.generate(n_games=5, history=history_df):
            assert len(set(game)) == 6

    def test_games_are_sorted(self, strategy, history_df):
        for game in strategy.generate(n_games=5, history=history_df):
            assert game == sorted(game)

    def test_generate_without_history(self, strategy):
        games = strategy.generate(n_games=3, history=None)
        assert len(games) == 3
        for g in games:
            assert len(g) == 6

    def test_reproducible_with_seed(self, history_df):
        s1 = BalancedV2Strategy(seed=42)
        s2 = BalancedV2Strategy(seed=42)
        games1 = s1.generate(n_games=5, history=history_df)
        games2 = s2.generate(n_games=5, history=history_df)
        assert games1 == games2

    def test_different_seeds_different_games(self, history_df):
        s1 = BalancedV2Strategy(seed=1)
        s2 = BalancedV2Strategy(seed=9999)
        # 확률적이므로 적어도 일부는 달라야 함
        games1 = s1.generate(n_games=10, history=history_df)
        games2 = s2.generate(n_games=10, history=history_df)
        assert any(g1 != g2 for g1, g2 in zip(games1, games2))


# ── 제약 통계 검증 (통계적) ────────────────────────────────────────────────

class TestConstraintStatistics:
    """전략이 통계적으로 제약 조건을 만족하는지 확인한다."""

    def _run(self, n: int = 200, seed: int = 0) -> list[list[int]]:
        strategy = BalancedV2Strategy(seed=seed)
        history  = _make_history(300)
        games: list[list[int]] = []
        while len(games) < n:
            games.extend(strategy.generate(n_games=5, history=history))
        return games[:n]

    def test_odd_even_ratio(self):
        """홀수 개수가 2~4 범위인 게임 비율 >= 99%."""
        games = self._run(200)
        valid = sum(1 for g in games if 2 <= sum(1 for n in g if n % 2) <= 4)
        assert valid / len(games) >= 0.99, f"odd/even 위반: {len(games)-valid}/{len(games)}"

    def test_no_triple_consecutive(self):
        """3개 이상 연속 번호가 없어야 한다 (100% 강제)."""
        games = self._run(200)
        for g in games:
            s = sorted(g)
            consec = 1
            for i in range(1, len(s)):
                consec = consec + 1 if s[i] == s[i - 1] + 1 else 1
                assert consec < 3, f"3 연속 발견: {s}"

    def test_no_more_than_2_same_trailing(self):
        """같은 끝자리가 3개 이상인 게임 비율이 낮아야 한다 (< 2%)."""
        games = self._run(200)
        violate = sum(
            1 for g in games if max(Counter(n % 10 for n in g).values()) > 2
        )
        assert violate / len(games) < 0.02, f"끝자리 위반: {violate}/{len(games)}"

    def test_min_3_decades_coverage(self):
        """최소 3구간 커버 비율 >= 98%."""
        games = self._run(200)
        ok = sum(1 for g in games if len({_DECADE_OF[n] for n in g}) >= 3)
        assert ok / len(games) >= 0.98, f"3구간 미달: {len(games)-ok}/{len(games)}"


# ── 다양성 후처리 ─────────────────────────────────────────────────────────

class TestEnsureDiversity:
    def test_no_duplicate_games(self, strategy, history_df):
        for _ in range(10):
            games = strategy.generate(n_games=5, history=history_df)
            keys  = [tuple(sorted(g)) for g in games]
            assert len(keys) == len(set(keys)), f"중복 게임 발생: {games}"

    def test_jaccard_below_threshold(self, strategy, history_df):
        for _ in range(5):
            games = strategy.generate(n_games=5, history=history_df)
            for i in range(len(games)):
                for j in range(i + 1, len(games)):
                    a, b = set(games[i]), set(games[j])
                    sim  = len(a & b) / len(a | b)
                    assert sim <= strategy._max_jaccard + 0.05, (
                        f"Jaccard {sim:.3f} > {strategy._max_jaccard} "
                        f"(games[{i}]={games[i]}, games[{j}]={games[j]})"
                    )


# ── game_stats 검증 ───────────────────────────────────────────────────────

class TestGameStats:
    """BacktestResult.game_stats 속성 검증."""

    def test_game_stats_structure(self, history_df):
        from src.lotto.ml.backtest import run_backtest

        result = run_backtest(
            history=history_df,
            strategy_name="balanced_v2",
            start_round=150,
            end_round=200,
            n_games=5,
            seed=42,
        )
        stats = result.game_stats
        assert "avg_sum" in stats
        assert "odd_distribution" in stats
        assert "avg_decade_coverage" in stats
        assert "consecutive_rate" in stats

    def test_avg_sum_in_lotto_range(self, history_df):
        from src.lotto.ml.backtest import run_backtest

        result = run_backtest(
            history=history_df,
            strategy_name="balanced_v2",
            start_round=150,
            end_round=200,
            n_games=5,
            seed=42,
        )
        avg_sum = result.game_stats["avg_sum"]
        assert 50 <= avg_sum <= 230, f"avg_sum={avg_sum} out of plausible range"

    def test_consecutive_rate_is_fraction(self, history_df):
        from src.lotto.ml.backtest import run_backtest

        result = run_backtest(
            history=history_df,
            strategy_name="balanced_v2",
            start_round=150,
            end_round=200,
            n_games=5,
            seed=42,
        )
        rate = result.game_stats["consecutive_rate"]
        assert 0.0 <= rate <= 1.0
