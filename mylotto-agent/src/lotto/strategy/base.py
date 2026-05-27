"""번호 생성 전략 추상 베이스 클래스."""

from abc import ABC, abstractmethod

import pandas as pd

# 로또 6/45 상수
LOTTO_MIN = 1
LOTTO_MAX = 45
NUMBERS_PER_GAME = 6


class BaseStrategy(ABC):
    """모든 번호 생성 전략이 구현해야 할 인터페이스."""

    name: str = "base"

    @abstractmethod
    def generate(self, n_games: int = 5, history: pd.DataFrame | None = None) -> list[list[int]]:
        """n_games 개의 게임을 생성하여 반환한다.

        Args:
            n_games:  생성할 게임 수 (기본 5)
            history:  과거 당첨번호 DataFrame (전략에 따라 활용)

        Returns:
            정렬된 숫자 6개 리스트의 리스트
            예: [[1, 7, 13, 27, 38, 45], ...]
        """
        ...

    # ── 공통 유틸 ──────────────────────────────────────────────────────────

    @staticmethod
    def validate_game(numbers: list[int]) -> bool:
        """유효한 로또 게임인지 검증한다."""
        if len(numbers) != NUMBERS_PER_GAME:
            return False
        if len(set(numbers)) != NUMBERS_PER_GAME:
            return False
        return all(LOTTO_MIN <= n <= LOTTO_MAX for n in numbers)
