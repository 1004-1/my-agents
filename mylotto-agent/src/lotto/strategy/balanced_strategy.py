"""균형 전략 — 구간·홀짝·빈도를 고려한 번호 생성."""

import random
from collections import Counter

import pandas as pd

from .base import BaseStrategy, LOTTO_MIN, LOTTO_MAX, NUMBERS_PER_GAME

# 1~45를 5개 구간으로 분할
BANDS = [
    range(1, 10),   # 1~9
    range(10, 20),  # 10~19
    range(20, 30),  # 20~29
    range(30, 40),  # 30~39
    range(40, 46),  # 40~45
]


def _build_freq_weights(history: pd.DataFrame) -> dict[int, float]:
    """과거 당첨 번호 빈도를 가중치 딕셔너리로 반환한다.

    빈도가 높은 번호일수록 선택 확률이 높아진다.
    """
    num_cols = ["num1", "num2", "num3", "num4", "num5", "num6"]
    all_numbers = history[num_cols].values.flatten()
    counter = Counter(all_numbers)
    total = sum(counter.values())
    return {n: counter.get(n, 0) / total for n in range(LOTTO_MIN, LOTTO_MAX + 1)}


class BalancedStrategy(BaseStrategy):
    """구간 균형 + 홀짝 균형 + 빈도 가중치를 조합한 전략.

    규칙:
    1. 5개 구간(1~9, 10~19, 20~29, 30~39, 40~45) 중 최소 4개 구간을 커버한다.
    2. 홀수:짝수 비율을 3:3 또는 4:2 (2:4) 범위 안에 유지한다.
    3. 과거 데이터가 있으면 빈도 가중치를 반영한다 (없으면 균등).
    4. 연속 3개 이상 연속 번호가 없도록 제한한다.
    """

    name = "balanced"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def generate(self, n_games: int = 5, history: pd.DataFrame | None = None) -> list[list[int]]:
        """n_games개의 균형 잡힌 게임을 생성한다."""
        weights = _build_freq_weights(history) if history is not None and len(history) > 0 else None

        games = []
        for _ in range(n_games):
            game = self._generate_one(weights)
            games.append(game)
        return games

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _generate_one(self, weights: dict[int, float] | None, max_retries: int = 500) -> list[int]:
        """조건을 만족하는 게임 하나를 생성한다."""
        for _ in range(max_retries):
            candidate = self._sample_with_band_balance(weights)
            if self._is_valid(candidate):
                return sorted(candidate)
        # 최대 재시도 초과 시 순수 랜덤 폴백
        return sorted(self._rng.sample(range(LOTTO_MIN, LOTTO_MAX + 1), NUMBERS_PER_GAME))

    def _sample_with_band_balance(self, weights: dict[int, float] | None) -> list[int]:
        """구간 균형을 고려하여 6개 번호를 추출한다."""
        pool = list(range(LOTTO_MIN, LOTTO_MAX + 1))
        if weights:
            w_list = [weights[n] for n in pool]
        else:
            w_list = [1.0 / len(pool)] * len(pool)

        selected: list[int] = []
        # 밴드 4개는 반드시 1개 이상 포함
        chosen_bands = self._rng.sample(range(len(BANDS)), 4)
        for band_idx in chosen_bands:
            band_pool = [n for n in BANDS[band_idx] if n not in selected]
            if not band_pool:
                continue
            band_weights = [weights[n] for n in band_pool] if weights else None
            pick = self._rng.choices(band_pool, weights=band_weights, k=1)[0]
            selected.append(pick)

        # 나머지 2개는 전체 풀에서 가중치 반영해 선택
        remaining_pool = [n for n in pool if n not in selected]
        remaining_weights = [w_list[n - 1] for n in remaining_pool]
        while len(selected) < NUMBERS_PER_GAME and remaining_pool:
            pick = self._rng.choices(remaining_pool, weights=remaining_weights, k=1)[0]
            idx = remaining_pool.index(pick)
            selected.append(pick)
            remaining_pool.pop(idx)
            remaining_weights.pop(idx)

        return selected

    @staticmethod
    def _is_valid(numbers: list[int]) -> bool:
        """홀짝 및 연속 번호 조건을 검증한다."""
        nums = sorted(numbers)
        # 홀짝 비율: 홀수 2~4개 허용
        odd_count = sum(1 for n in nums if n % 2 == 1)
        if not (2 <= odd_count <= 4):
            return False
        # 연속 번호 3개 이상 금지
        consec = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                consec += 1
                if consec >= 3:
                    return False
            else:
                consec = 1
        return True
