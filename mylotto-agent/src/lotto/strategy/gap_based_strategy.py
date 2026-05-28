"""갭 기반 번호 생성 전략.

최근 장기간 미출현 번호에 가중치를 부여하되, 극단적인 집중을 방지한다.
홀짝 균형, 합계 범위, 연속·구간 제약을 준수한다.

로또는 독립 확률 추첨이다.
이 전략은 당첨 번호를 '예측'하지 않으며,
미출현 간격(gap) 통계를 활용한 확률적 샘플링을 수행한다.
"""
from __future__ import annotations

import logging
import random

import numpy as np
import pandas as pd

from .base import BaseStrategy, LOTTO_MIN, LOTTO_MAX, NUMBERS_PER_GAME
from .model_score_strategy import _check_constraints

logger = logging.getLogger(__name__)

_NUM_COLS = [f"num{j}" for j in range(1, 7)]


class GapBasedStrategy(BaseStrategy):
    """최근 미출현 번호에 가중치를 주는 전략.

    동작 방식:
        1. history에서 번호별 마지막 출현 이후 경과 회차(gap) 계산
        2. gap^alpha 비례 가중치 → 소프트맥스 온도 적용
        3. gap을 평균의 max_gap_ratio배로 클리핑하여 극단 집중 방지
        4. 제약 조건(홀짝·합계·연속·구간) 만족하는 게임 샘플링

    Args:
        seed:             재현성용 랜덤 시드
        alpha:            gap 가중치 지수 (높을수록 gap이 큰 번호 선호, 권장: 1.0~2.0)
        temperature:      소프트맥스 온도 (높을수록 균등 분포)
        max_gap_ratio:    gap 클리핑 임계값 (평균 대비 배수) — 극단 집중 방지
        apply_constraints: 홀짝·합계·연속·구간 제약 적용 여부
    """

    name = "gap_based"

    def __init__(
        self,
        seed: int | None = None,
        alpha: float = 1.2,
        temperature: float = 2.0,
        max_gap_ratio: float = 2.5,
        apply_constraints: bool = True,
    ):
        self._rng             = random.Random(seed)
        self._np_rng          = np.random.default_rng(seed)
        self._alpha           = alpha
        self._temperature     = temperature
        self._max_gap_ratio   = max_gap_ratio
        self._apply_constraints = apply_constraints

    # ── 팩토리 (백테스트 경량 모드) ──────────────────────────────────────

    @classmethod
    def from_weights(
        cls,
        weights: np.ndarray,
        seed: int | None = None,
        apply_constraints: bool = True,
    ) -> "GapBasedStrategy":
        """사전 계산된 가중치를 직접 주입하는 팩토리.

        모델 파일 I/O 없이 메모리 내 weights로 전략 인스턴스를 생성한다.
        주로 백테스트 배치 사전 계산 결과를 주입하는 데 사용한다.
        """
        instance = cls.__new__(cls)
        instance._preloaded_weights = weights.copy()
        instance._rng               = random.Random(seed)
        instance._np_rng            = np.random.default_rng(seed)
        instance._apply_constraints = apply_constraints
        return instance

    # ── 핵심 메서드 ───────────────────────────────────────────────────────

    def generate(
        self,
        n_games: int = 5,
        history: pd.DataFrame | None = None,
    ) -> list[list[int]]:
        """gap 가중치로 n_games개의 게임을 생성한다."""
        # 사전 계산 가중치 우선 사용 (백테스트 경량 모드)
        if hasattr(self, "_preloaded_weights"):
            weights = self._preloaded_weights
        elif history is None or len(history) == 0:
            logger.warning("history 없음 — 순수 랜덤으로 대체")
            return [
                sorted(self._rng.sample(range(LOTTO_MIN, LOTTO_MAX + 1), NUMBERS_PER_GAME))
                for _ in range(n_games)
            ]
        else:
            weights = self._compute_weights(history)

        numbers = np.arange(LOTTO_MIN, LOTTO_MAX + 1)
        games: list[list[int]] = []
        for _ in range(n_games):
            if self._apply_constraints:
                game = self._sample_with_constraints(numbers, weights)
            else:
                chosen = self._np_rng.choice(numbers, size=NUMBERS_PER_GAME, replace=False, p=weights)
                game   = sorted(chosen.tolist())
            games.append(game)
        return games

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def compute_gaps(self, history: pd.DataFrame) -> np.ndarray:
        """번호별 마지막 출현 이후 경과 회차를 반환한다.

        Returns:
            shape=(45,) float array.
            gaps[n] = 마지막 출현 이후 경과 회차 (출현 안 한 경우 = n_rounds).
        """
        history  = history.sort_values("round_no").reset_index(drop=True)
        n_rounds = len(history)
        row_idx  = np.arange(n_rounds)

        # last_seen[n] = 번호 n+1이 마지막으로 나온 행 인덱스, -1이면 미출현
        last_seen = np.full(45, -1, dtype=np.int32)
        for col in _NUM_COLS:
            nums  = history[col].values.astype(int) - 1   # 0-indexed
            valid = (nums >= 0) & (nums < 45)
            np.maximum.at(last_seen, nums[valid], row_idx[valid])

        # gap = 마지막 출현 이후 경과 회차
        gaps = np.where(
            last_seen >= 0,
            (n_rounds - 1) - last_seen,
            float(n_rounds),
        ).astype(float)
        return gaps

    def _compute_weights(self, history: pd.DataFrame) -> np.ndarray:
        """gap → 샘플링 가중치 벡터 (shape=(45,), sum=1.0)."""
        gaps = self.compute_gaps(history)

        # 극단 집중 방지: gap을 (평균 × max_gap_ratio)로 클리핑
        mean_gap = float(np.mean(gaps))
        cap      = mean_gap * self._max_gap_ratio
        gaps     = np.clip(gaps, 0.0, cap)

        # gap^alpha 비례 가중치 (+1로 gap=0 번호도 0이 되지 않게)
        raw = np.power(gaps + 1.0, self._alpha)

        # 소프트맥스 with temperature
        scaled  = raw / max(self._temperature, 1e-8)
        scaled -= scaled.max()
        exp_s   = np.exp(scaled)
        return exp_s / exp_s.sum()

    def _sample_with_constraints(
        self,
        numbers: np.ndarray,
        weights: np.ndarray,
        max_attempts: int = 300,
    ) -> list[int]:
        """제약 조건을 만족하는 게임 하나를 샘플링한다."""
        for _ in range(max_attempts):
            chosen = self._np_rng.choice(numbers, size=NUMBERS_PER_GAME, replace=False, p=weights)
            game   = sorted(chosen.tolist())
            if _check_constraints(game):
                return game
        # 폴백: 제약 없이 gap 가중치 샘플링
        chosen = self._np_rng.choice(numbers, size=NUMBERS_PER_GAME, replace=False, p=weights)
        return sorted(chosen.tolist())
