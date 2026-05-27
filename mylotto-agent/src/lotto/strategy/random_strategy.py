"""순수 랜덤 전략 — 1~45에서 6개를 무작위 추출."""

import random

import pandas as pd

from .base import BaseStrategy, LOTTO_MIN, LOTTO_MAX, NUMBERS_PER_GAME


class RandomStrategy(BaseStrategy):
    """가장 단순한 전략: 완전 무작위 추출.

    과거 데이터와 무관하게 매번 독립적으로 숫자를 고른다.
    """

    name = "random"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def generate(self, n_games: int = 5, history: pd.DataFrame | None = None) -> list[list[int]]:
        """n_games개의 랜덤 게임을 생성한다.

        Args:
            n_games:  생성할 게임 수
            history:  사용하지 않음 (인터페이스 호환용)

        Returns:
            정렬된 숫자 6개 리스트의 리스트
        """
        games = []
        for _ in range(n_games):
            numbers = self._rng.sample(range(LOTTO_MIN, LOTTO_MAX + 1), NUMBERS_PER_GAME)
            games.append(sorted(numbers))
        return games
