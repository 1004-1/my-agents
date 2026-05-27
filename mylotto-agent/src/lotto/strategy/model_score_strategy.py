"""모델 score 기반 번호 생성 전략.

로또는 독립 확률 추첨이다.
이 전략은 당첨 번호를 '예측'하지 않으며,
모델이 높게 평가한 번호를 우선 고려하는 확률적 샘플링을 수행한다.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, LOTTO_MIN, LOTTO_MAX, NUMBERS_PER_GAME

logger = logging.getLogger(__name__)

# 구간 정의 (balanced_strategy와 동일)
_BANDS = [
    (1,  9),
    (10, 19),
    (20, 29),
    (30, 39),
    (40, 45),
]


class ModelScoreStrategy(BaseStrategy):
    """모델 score 가중치를 사용해 번호를 생성하는 전략.

    동작 방식:
        1. 저장된 LogisticRegression 모델 로드 (lazy)
        2. 현재 history 기준으로 각 번호의 feature 계산
        3. 모델 predict_proba로 score 산출
        4. 소프트맥스 가중치 + 제약 조건으로 게임 샘플링

    제약 조건 (apply_constraints=True):
        - 홀수 2~4개
        - 번호 합계 70~200
        - 3개 이상 연속 번호 금지
        - 한 구간에 3개 이상 금지

    Args:
        model_path:          학습된 Pipeline 모델 파일 경로
        seed:                재현성용 랜덤 시드
        temperature:         소프트맥스 온도 (높을수록 균등, 낮을수록 score 집중)
        apply_constraints:   제약 조건 적용 여부
    """

    name = "model_score"

    def __init__(
        self,
        model_path: Path | str,
        seed: int | None = None,
        temperature: float = 2.0,
        apply_constraints: bool = True,
    ):
        self._model_path = Path(model_path)
        self._model: Any = None          # lazy load
        self._rng     = random.Random(seed)
        self._np_rng  = np.random.default_rng(seed)
        self._temperature = temperature
        self._apply_constraints = apply_constraints
        # _preloaded_scores는 from_scores()에서만 설정됨

    # ── 핵심 메서드 ───────────────────────────────────────────────────────

    def generate(
        self,
        n_games: int = 5,
        history: pd.DataFrame | None = None,
    ) -> list[list[int]]:
        """모델 score 가중치를 사용해 n_games개의 게임을 생성한다."""
        # from_scores()로 만든 인스턴스: preloaded scores 사용
        if hasattr(self, "_preloaded_scores"):
            return self._sample_games(self._preloaded_scores, n_games)

        if history is None or len(history) == 0:
            logger.warning("history 없음 — 순수 랜덤으로 대체")
            return [
                sorted(self._rng.sample(range(LOTTO_MIN, LOTTO_MAX + 1), NUMBERS_PER_GAME))
                for _ in range(n_games)
            ]

        scores = self._get_scores(history)
        return self._sample_games(scores, n_games)

    def generate_with_scores(
        self,
        n_games: int = 5,
        history: pd.DataFrame | None = None,
    ) -> tuple[list[list[int]], dict[int, float]]:
        """게임과 함께 번호별 score를 함께 반환한다."""
        if hasattr(self, "_preloaded_scores"):
            scores = self._preloaded_scores
        elif history is None or len(history) == 0:
            return self.generate(n_games, history), {}
        else:
            scores = self._get_scores(history)
        return self._sample_games(scores, n_games), scores

    # ── 팩토리 (백테스트용) ───────────────────────────────────────────────

    @classmethod
    def from_scores(
        cls,
        scores: dict[int, float],
        seed: int | None = None,
        temperature: float = 2.0,
        apply_constraints: bool = True,
    ) -> "ModelScoreStrategy":
        """사전 계산된 scores를 직접 주입하는 팩토리 (백테스트 경량 모드).

        모델 파일 I/O 없이 메모리 내 scores로 전략 인스턴스를 생성한다.
        """
        instance = cls.__new__(cls)
        instance._preloaded_scores = dict(scores)
        instance._rng     = random.Random(seed)
        instance._np_rng  = np.random.default_rng(seed)
        instance._temperature = temperature
        instance._apply_constraints = apply_constraints
        return instance

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _get_scores(self, history: pd.DataFrame) -> dict[int, float]:
        """현재 history 기준 각 번호의 모델 score를 반환한다.

        history는 예측 대상 회차보다 앞선 데이터만 포함해야 한다.
        """
        from ..ml.feature_builder import _compute_number_features, FEATURE_COLS

        model = self._ensure_model_loaded()
        feature_rows = [
            [_compute_number_features(history, n)[c] for c in FEATURE_COLS]
            for n in range(LOTTO_MIN, LOTTO_MAX + 1)
        ]
        X = np.nan_to_num(np.array(feature_rows, dtype=float), nan=0.0)
        probs = model.predict_proba(X)[:, 1]
        return {n: float(probs[n - 1]) for n in range(LOTTO_MIN, LOTTO_MAX + 1)}

    def _sample_games(
        self,
        scores: dict[int, float],
        n_games: int,
    ) -> list[list[int]]:
        """score 가중치로 n_games개의 게임을 생성한다."""
        numbers = list(range(LOTTO_MIN, LOTTO_MAX + 1))
        raw = np.array([scores.get(n, 1.0 / 45) for n in numbers], dtype=float)

        # 소프트맥스 with temperature
        scaled = raw / max(self._temperature, 1e-8)
        scaled -= scaled.max()
        exp_s  = np.exp(scaled)
        weights = exp_s / exp_s.sum()

        games = []
        for _ in range(n_games):
            if self._apply_constraints:
                game = self._sample_with_constraints(np.array(numbers), weights)
            else:
                chosen = self._np_rng.choice(
                    numbers, size=NUMBERS_PER_GAME, replace=False, p=weights
                )
                game = sorted(chosen.tolist())
            games.append(game)
        return games

    def _sample_with_constraints(
        self,
        numbers: np.ndarray,
        weights: np.ndarray,
        max_attempts: int = 200,
    ) -> list[int]:
        """제약을 만족하는 게임을 샘플링한다. 실패 시 단순 가중 샘플링으로 폴백."""
        for _ in range(max_attempts):
            chosen = self._np_rng.choice(
                numbers, size=NUMBERS_PER_GAME, replace=False, p=weights
            )
            game = sorted(chosen.tolist())
            if _check_constraints(game):
                return game
        # 제약 만족 실패 시 폴백
        chosen = self._np_rng.choice(
            numbers, size=NUMBERS_PER_GAME, replace=False, p=weights
        )
        return sorted(chosen.tolist())

    def _ensure_model_loaded(self) -> Any:
        """모델을 lazy load한다."""
        if self._model is None:
            from ..ml.model_trainer import load_model
            self._model = load_model(self._model_path)
        return self._model


# ── 제약 검사 (모듈 레벨 함수) ───────────────────────────────────────────

def _check_constraints(game: list[int]) -> bool:
    """홀짝 / 합계 / 연속 / 구간 제약을 검사한다."""
    # 홀수 2~4개
    odd_count = sum(1 for n in game if n % 2 == 1)
    if not (2 <= odd_count <= 4):
        return False
    # 합계 범위
    if not (70 <= sum(game) <= 200):
        return False
    # 3개 이상 연속 금지
    srt = sorted(game)
    consec = cur = 1
    for i in range(1, len(srt)):
        if srt[i] == srt[i - 1] + 1:
            cur += 1
            consec = max(consec, cur)
        else:
            cur = 1
    if consec >= 3:
        return False
    # 한 구간에 3개 이상 금지
    for lo, hi in _BANDS:
        if sum(1 for n in game if lo <= n <= hi) >= 3:
            return False
    return True
