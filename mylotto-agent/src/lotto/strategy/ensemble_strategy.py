"""앙상블 전략 — 여러 전략을 혼합해 번호 다양성과 coverage를 높인다.

고정 구성 (5게임):
    - model_score  2게임
    - balanced     1게임
    - random       1게임
    - gap_based    1게임

로또는 독립 확률 추첨이다.
이 전략은 당첨 번호를 '예측'하지 않으며,
다양한 샘플링 전략을 혼합해 번호 커버리지를 높이는 실험적 접근이다.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, LOTTO_MIN, LOTTO_MAX, NUMBERS_PER_GAME
from .model_score_strategy import _check_constraints

logger = logging.getLogger(__name__)

# 앙상블 구성 — (전략명, 게임 수) 순서 유지
_COMPOSITION: list[tuple[str, int]] = [
    ("model_score", 2),
    ("balanced",    1),
    ("random",      1),
    ("gap_based",   1),
]
ENSEMBLE_TOTAL = sum(n for _, n in _COMPOSITION)  # = 5


# ── 다양성 측정 유틸 (backtest에서도 재사용) ─────────────────────────────

def compute_diversity(games: list[list[int]]) -> dict[str, float]:
    """게임 목록의 다양성 지표를 계산한다.

    Args:
        games: 정렬된 6-숫자 게임 리스트

    Returns:
        coverage      : 사용된 고유 번호 수 / 45  (높을수록 좋음)
        avg_jaccard   : 게임 쌍 평균 Jaccard 유사도 (낮을수록 다양)
        diversity_score: coverage×0.5 + (1-avg_jaccard)×0.5  → 0~100 스케일
    """
    if not games:
        return {"coverage": 0.0, "avg_jaccard": 0.0, "diversity_score": 0.0}

    all_nums = set()
    for g in games:
        all_nums.update(g)
    coverage = len(all_nums) / 45.0

    # 쌍별 Jaccard 유사도
    jaccards: list[float] = []
    for i in range(len(games)):
        for j in range(i + 1, len(games)):
            a, b = set(games[i]), set(games[j])
            jaccards.append(len(a & b) / len(a | b))
    avg_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

    diversity_score = (coverage * 0.5 + (1.0 - avg_jaccard) * 0.5) * 100.0

    return {
        "coverage":        coverage,
        "avg_jaccard":     avg_jaccard,
        "diversity_score": diversity_score,
    }


# ── EnsembleStrategy ───────────────────────────────────────────────────────

class EnsembleStrategy(BaseStrategy):
    """여러 전략을 혼합해 5게임을 생성하는 앙상블 전략.

    구성: model_score×2 + balanced×1 + random×1 + gap_based×1 = 5게임

    다양성 후처리:
        1. 동일 게임 조합 재생성
        2. Jaccard 유사도가 max_jaccard를 초과하는 쌍을 재생성

    n_games 파라미터는 항상 ENSEMBLE_TOTAL(=5)로 처리된다.

    Args:
        model_path:    학습된 모델 .pkl 경로 (model_score 서브전략용)
        model_scores:  사전 계산된 점수 dict — 백테스트 경량 모드 전용
        seed:          재현성용 랜덤 시드
        max_jaccard:   허용 최대 Jaccard 유사도 (기본 0.5 = 4개 공유 기준)
    """

    name = "ensemble"

    def __init__(
        self,
        model_path: Path | str | None = None,
        model_scores: dict[int, float] | None = None,
        seed: int | None = None,
        max_jaccard: float = 0.5,
    ):
        self._model_path   = Path(model_path) if model_path else None
        self._model_scores = dict(model_scores) if model_scores else None
        self._rng          = random.Random(seed)
        self._np_rng       = np.random.default_rng(seed)
        self._max_jaccard  = max_jaccard
        self._seed         = seed

    # ── 핵심 메서드 ───────────────────────────────────────────────────────

    def generate(
        self,
        n_games: int = ENSEMBLE_TOTAL,
        history: pd.DataFrame | None = None,
    ) -> list[list[int]]:
        """앙상블 구성에 따라 5게임을 생성하고 다양성을 후처리한다.

        n_games는 항상 ENSEMBLE_TOTAL(=5)로 처리된다.
        """
        games = self._generate_from_composition(history)
        games = self._ensure_diversity(games)
        return games

    def generate_with_diversity(
        self,
        history: pd.DataFrame | None = None,
    ) -> tuple[list[list[int]], dict[str, float]]:
        """게임과 함께 다양성 지표를 반환한다."""
        games = self.generate(history=history)
        return games, compute_diversity(games)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _generate_from_composition(
        self,
        history: pd.DataFrame | None,
    ) -> list[list[int]]:
        """각 서브전략에서 게임을 생성해 합친다."""
        from .random_strategy import RandomStrategy
        from .balanced_strategy import BalancedStrategy
        from .gap_based_strategy import GapBasedStrategy
        from .model_score_strategy import ModelScoreStrategy

        all_games: list[list[int]] = []

        for strategy_name, count in _COMPOSITION:
            sub_seed = self._seed  # 서브전략도 동일 시드 → 재현성

            if strategy_name == "model_score":
                if self._model_scores is not None:
                    # 백테스트 경량 모드: 사전 계산 점수 사용
                    sub = ModelScoreStrategy.from_scores(
                        self._model_scores, seed=sub_seed
                    )
                elif (
                    self._model_path is not None
                    and self._model_path.exists()
                ):
                    sub = ModelScoreStrategy(
                        model_path=self._model_path, seed=sub_seed
                    )
                else:
                    logger.warning(
                        "model_score 모델 없음 — random으로 대체 (%d게임)", count
                    )
                    sub = RandomStrategy(seed=sub_seed)

            elif strategy_name == "balanced":
                sub = BalancedStrategy(seed=sub_seed)

            elif strategy_name == "random":
                sub = RandomStrategy(seed=sub_seed)

            elif strategy_name == "gap_based":
                sub = GapBasedStrategy(seed=sub_seed)

            else:
                logger.warning("알 수 없는 서브전략: %s → random", strategy_name)
                sub = RandomStrategy(seed=sub_seed)

            games = sub.generate(n_games=count, history=history)
            all_games.extend(games)

        return all_games

    def _ensure_diversity(
        self,
        games: list[list[int]],
        max_attempts: int = 200,
    ) -> list[list[int]]:
        """게임 간 과도한 유사도를 제거한다.

        1. 동일 게임 → 즉시 재생성 (랜덤 제약 만족)
        2. Jaccard > max_jaccard인 쌍의 두 번째 게임 재생성
        max_attempts 초과 시 현재 상태 그대로 반환.
        """
        games = [list(g) for g in games]   # shallow copy

        for attempt in range(max_attempts):
            replaced = False

            # ① 동일 게임 제거
            seen: set[tuple] = set()
            for idx, g in enumerate(games):
                key = tuple(g)
                if key in seen:
                    games[idx] = self._random_valid_game()
                    replaced   = True
                else:
                    seen.add(key)

            if replaced:
                continue

            # ② Jaccard 임계값 초과 쌍 탐색
            worst_idx  = -1
            worst_sim  = -1.0
            n          = len(games)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b  = set(games[i]), set(games[j])
                    sim   = len(a & b) / len(a | b)
                    if sim > worst_sim:
                        worst_sim = sim
                        worst_idx = j   # j번째 게임 교체

            if worst_sim <= self._max_jaccard:
                break  # 모든 쌍이 기준 이하 → 완료

            # 가장 유사한 쌍의 두 번째 게임 교체
            games[worst_idx] = self._random_valid_game()

        return games

    def _random_valid_game(self, max_attempts: int = 300) -> list[int]:
        """제약을 만족하는 랜덤 게임을 생성한다."""
        numbers = np.arange(LOTTO_MIN, LOTTO_MAX + 1)
        weights = np.ones(45) / 45.0
        for _ in range(max_attempts):
            chosen = self._np_rng.choice(numbers, size=NUMBERS_PER_GAME, replace=False, p=weights)
            game   = sorted(chosen.tolist())
            if _check_constraints(game):
                return game
        return sorted(self._rng.sample(range(LOTTO_MIN, LOTTO_MAX + 1), NUMBERS_PER_GAME))
