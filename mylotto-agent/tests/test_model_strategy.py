"""ModelScoreStrategy 단위 테스트."""
from __future__ import annotations

import random

import pandas as pd
import pytest

from src.lotto.strategy.model_score_strategy import (
    ModelScoreStrategy,
    _check_constraints,
)


def _uniform_scores() -> dict[int, float]:
    """균등 score (1/45) 딕셔너리."""
    return {n: 1.0 / 45 for n in range(1, 46)}


def _random_scores(seed: int = 0) -> dict[int, float]:
    rng = random.Random(seed)
    raw = {n: rng.random() for n in range(1, 46)}
    total = sum(raw.values())
    return {n: v / total for n, v in raw.items()}


class TestFromScores:
    """from_scores() 팩토리 메서드 테스트."""

    def test_generates_correct_count(self):
        strategy = ModelScoreStrategy.from_scores(_uniform_scores(), seed=42)
        games = strategy.generate(n_games=5)
        assert len(games) == 5

    def test_each_game_has_6_numbers(self):
        strategy = ModelScoreStrategy.from_scores(_uniform_scores(), seed=42)
        for game in strategy.generate(n_games=3):
            assert len(game) == 6

    def test_numbers_in_range(self):
        strategy = ModelScoreStrategy.from_scores(_uniform_scores(), seed=42)
        for game in strategy.generate(n_games=10):
            assert all(1 <= n <= 45 for n in game)

    def test_no_duplicates(self):
        strategy = ModelScoreStrategy.from_scores(_uniform_scores(), seed=42)
        for game in strategy.generate(n_games=10):
            assert len(set(game)) == 6

    def test_sorted_output(self):
        strategy = ModelScoreStrategy.from_scores(_uniform_scores(), seed=42)
        for game in strategy.generate(n_games=5):
            assert game == sorted(game)

    def test_reproducible_with_seed(self):
        s1 = ModelScoreStrategy.from_scores(_random_scores(), seed=7)
        s2 = ModelScoreStrategy.from_scores(_random_scores(), seed=7)
        assert s1.generate(n_games=3) == s2.generate(n_games=3)

    def test_different_seeds_differ(self):
        s1 = ModelScoreStrategy.from_scores(_random_scores(), seed=1)
        s2 = ModelScoreStrategy.from_scores(_random_scores(), seed=2)
        games1 = s1.generate(n_games=5)
        games2 = s2.generate(n_games=5)
        assert games1 != games2

    def test_no_constraints_still_valid(self):
        strategy = ModelScoreStrategy.from_scores(
            _uniform_scores(), seed=42, apply_constraints=False
        )
        for game in strategy.generate(n_games=10):
            assert len(game) == 6
            assert len(set(game)) == 6
            assert all(1 <= n <= 45 for n in game)


class TestGenerateWithHistory:
    """모델 없이 history만으로는 폴백 발생 — from_scores로 우회 테스트."""

    def test_generate_with_empty_history_fallback(self):
        # 빈 history → from_scores 아닌 일반 인스턴스는 랜덤 폴백
        # 이 테스트는 from_scores를 통해 scores만 있으면 history 없어도 동작함을 확인
        strategy = ModelScoreStrategy.from_scores(_uniform_scores(), seed=42)
        games = strategy.generate(n_games=3, history=pd.DataFrame())
        assert len(games) == 3


class TestCheckConstraints:
    def test_valid_game_passes(self):
        game = [3, 14, 22, 31, 40, 43]  # 홀짝 4:2, 합계=153, 구간 분산
        assert _check_constraints(game)

    def test_all_odd_fails(self):
        game = [1, 3, 7, 11, 23, 35]  # 홀수 6개
        assert not _check_constraints(game)

    def test_all_even_fails(self):
        game = [2, 4, 8, 12, 24, 36]  # 짝수 6개
        assert not _check_constraints(game)

    def test_3_consecutive_fails(self):
        game = [5, 6, 7, 20, 30, 40]  # 5-6-7 연속
        assert not _check_constraints(game)

    def test_low_sum_fails(self):
        game = [1, 2, 3, 4, 5, 6]  # 합계=21 (< 70)
        assert not _check_constraints(game)

    def test_high_sum_fails(self):
        game = [40, 41, 42, 43, 44, 45]  # 합계=255 (> 200)
        assert not _check_constraints(game)

    def test_band_concentration_fails(self):
        # 구간 1~9에 3개 이상
        game = [1, 3, 5, 20, 30, 40]
        assert not _check_constraints(game)


class TestGenerateWithScores:
    def test_returns_games_and_scores_dict(self):
        strategy = ModelScoreStrategy.from_scores(_random_scores(), seed=0)
        games, scores = strategy.generate_with_scores(n_games=3)
        assert len(games) == 3
        assert len(scores) == 45
        assert all(1 <= n <= 45 for n in scores)
