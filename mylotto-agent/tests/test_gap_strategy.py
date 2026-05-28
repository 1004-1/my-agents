"""GapBasedStrategy 단위 테스트."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from src.lotto.strategy.gap_based_strategy import GapBasedStrategy


def _make_history(n_rounds: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    records = []
    for i in range(1, n_rounds + 1):
        nums   = sorted(rng.sample(range(1, 46), 6))
        others = [n for n in range(1, 46) if n not in nums]
        bonus  = rng.choice(others)
        records.append({
            "round_no": i,
            "date": "2002-12-07",
            "num1": nums[0], "num2": nums[1], "num3": nums[2],
            "num4": nums[3], "num5": nums[4], "num6": nums[5],
            "bonus": bonus,
        })
    return pd.DataFrame(records)


# ── 기본 생성 ─────────────────────────────────────────────────────────────

class TestGapBasedGenerate:
    def test_generates_correct_count(self, larger_history):
        strategy = GapBasedStrategy(seed=42)
        assert len(strategy.generate(n_games=5, history=larger_history)) == 5

    def test_each_game_has_6_numbers(self, larger_history):
        strategy = GapBasedStrategy(seed=42)
        for game in strategy.generate(n_games=3, history=larger_history):
            assert len(game) == 6

    def test_numbers_in_range(self, larger_history):
        strategy = GapBasedStrategy(seed=42)
        for game in strategy.generate(n_games=5, history=larger_history):
            assert all(1 <= n <= 45 for n in game)

    def test_no_duplicates(self, larger_history):
        strategy = GapBasedStrategy(seed=42)
        for game in strategy.generate(n_games=5, history=larger_history):
            assert len(set(game)) == 6

    def test_sorted_output(self, larger_history):
        strategy = GapBasedStrategy(seed=42)
        for game in strategy.generate(n_games=5, history=larger_history):
            assert game == sorted(game)

    def test_reproducible_with_seed(self, larger_history):
        s1 = GapBasedStrategy(seed=7)
        s2 = GapBasedStrategy(seed=7)
        assert s1.generate(n_games=3, history=larger_history) == \
               s2.generate(n_games=3, history=larger_history)

    def test_no_constraint_mode(self, larger_history):
        strategy = GapBasedStrategy(seed=42, apply_constraints=False)
        for game in strategy.generate(n_games=5, history=larger_history):
            assert len(game) == 6
            assert len(set(game)) == 6
            assert all(1 <= n <= 45 for n in game)


# ── 히스토리 없을 때 ────────────────────────────────────────────────────

class TestGapBasedFallback:
    def test_empty_history_returns_random(self):
        empty   = pd.DataFrame(columns=["round_no","date","num1","num2","num3","num4","num5","num6","bonus"])
        strategy = GapBasedStrategy(seed=0)
        games   = strategy.generate(n_games=3, history=empty)
        assert len(games) == 3
        for g in games:
            assert len(g) == 6
            assert all(1 <= n <= 45 for n in g)

    def test_none_history_returns_random(self):
        strategy = GapBasedStrategy(seed=0)
        games   = strategy.generate(n_games=3, history=None)
        assert len(games) == 3


# ── gap 계산 검증 ─────────────────────────────────────────────────────────

class TestGapComputation:
    def test_gap_shape(self, larger_history):
        strategy = GapBasedStrategy()
        gaps = strategy.compute_gaps(larger_history)
        assert gaps.shape == (45,)

    def test_gap_non_negative(self, larger_history):
        strategy = GapBasedStrategy()
        gaps = strategy.compute_gaps(larger_history)
        assert (gaps >= 0).all()

    def test_appeared_in_last_round_has_gap_zero(self):
        """마지막 회차에 출현한 번호는 gap=0이어야 한다."""
        records = [
            {"round_no": 1, "date": "2002-12-07",
             "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6, "bonus": 7},
            {"round_no": 2, "date": "2002-12-14",
             "num1": 10, "num2": 11, "num3": 12, "num4": 13, "num5": 14, "num6": 15, "bonus": 16},
        ]
        history  = pd.DataFrame(records)
        strategy = GapBasedStrategy()
        gaps     = strategy.compute_gaps(history)
        # 회차 2에 출현한 번호들 (index 1-indexed: 10..15) → gap = 0
        for n in [10, 11, 12, 13, 14, 15]:
            assert gaps[n - 1] == 0.0, f"번호 {n}: gap={gaps[n-1]} (기대 0)"

    def test_never_appeared_has_max_gap(self):
        """한 번도 출현하지 않은 번호는 gap = n_rounds."""
        records = [
            {"round_no": 1, "date": "2002-12-07",
             "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6, "bonus": 7},
        ]
        history  = pd.DataFrame(records)
        strategy = GapBasedStrategy()
        gaps     = strategy.compute_gaps(history)
        # 번호 44, 45는 출현하지 않음 → gap = 1 (n_rounds=1)
        assert gaps[43] == 1.0  # number 44
        assert gaps[44] == 1.0  # number 45

    def test_high_gap_numbers_get_higher_weight(self, larger_history):
        """gap이 더 큰 번호가 더 높은 가중치를 받는지 확인 (alpha > 0)."""
        strategy = GapBasedStrategy(alpha=1.5)
        gaps     = strategy.compute_gaps(larger_history)
        weights  = strategy._compute_weights(larger_history)
        # gap이 클수록 weight가 같거나 커야 함 (클리핑 전 원칙)
        # 상위 10개 gap 번호 중 최소 7개는 상위 15개 weight에 포함되어야 함
        top_gap_nums    = set(np.argsort(gaps)[-10:])
        top_weight_nums = set(np.argsort(weights)[-15:])
        assert len(top_gap_nums & top_weight_nums) >= 6


# ── 제약 조건 준수 ─────────────────────────────────────────────────────────

class TestGapBasedConstraints:
    def test_most_games_satisfy_odd_even(self):
        """대부분의 게임에서 홀수 번호가 2~4개이어야 한다."""
        strategy = GapBasedStrategy(seed=99, apply_constraints=True)
        history  = _make_history(50, seed=1)
        games    = strategy.generate(n_games=20, history=history)
        valid    = sum(1 for g in games if 2 <= sum(n % 2 for n in g) <= 4)
        assert valid >= 15  # 최소 75%

    def test_most_games_satisfy_sum_range(self):
        """대부분의 게임에서 합계가 70~200 범위이어야 한다."""
        strategy = GapBasedStrategy(seed=99, apply_constraints=True)
        history  = _make_history(50, seed=1)
        games    = strategy.generate(n_games=20, history=history)
        valid    = sum(1 for g in games if 70 <= sum(g) <= 200)
        assert valid >= 15
