"""균형 전략 v2 — 더 엄격한 제약으로 통계적 다양성을 높인 고도화 전략.

개선 사항 (기존 balanced 대비):
    1. 홀짝: 2:4, 3:3, 4:2 만 허용
    2. 합계 범위: history 실제 분포의 5th~95th 백분위 기반 동적 계산
    3. 십단위 구간: 구간당 최대 3개 & 최소 3구간 커버
    4. 연속 번호: 최대 2개 연속 (3개 이상 연속 금지)
    5. 끝자리 (trailing digit): 같은 끝자리 최대 2개
    6. 이전 회차 겹침: 최대 max_prev_overlap개 제한
    7. 핫 번호 (hot number): 최근 hot_window 회차 내 hot_threshold 이상 출현 번호
       최대 max_hot_numbers개 제한
    8. 게임 간 Jaccard ≤ max_jaccard 다양성 후처리
    9. 최근 빈도 가중치 샘플링 (최근 recent_weight_window 회차 기준)

로또는 독립 확률 추첨이다.
이 전략은 당첨 번호를 '예측'하지 않으며,
통계적 제약을 통해 번호 다양성과 분포 균형을 높이는 실험적 접근이다.
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, LOTTO_MIN, LOTTO_MAX, NUMBERS_PER_GAME

logger = logging.getLogger(__name__)

_NUM_COLS = [f"num{j}" for j in range(1, 7)]

# 십단위 구간 사전 (O(1) 조회): 1–9 → 1, 10–19 → 2, 20–29 → 3, 30–39 → 4, 40–45 → 5
_DECADE_OF: dict[int, int] = {n: (1 if n < 10 else n // 10 + 1) for n in range(1, 46)}

# 합계 기본 범위 (history 없을 때 폴백; 실제 로또 분포 기반 근사치)
_SUM_LO_DEFAULT = 100
_SUM_HI_DEFAULT = 176


class BalancedV2Strategy(BaseStrategy):
    """더 엄격한 제약 + 최근 빈도 가중치 + 게임 간 다양성을 갖춘 고도화 균형 전략.

    기존 balanced 대비 개선 사항:
    - history 분포 기반 동적 합계 범위 (5th-95th 백분위)
    - 십단위 구간: 구간당 최대 3개 & 최소 3구간 커버
    - 끝자리(trailing digit): 같은 끝자리 최대 2개
    - 이전 회차 겹침 제한 (기본 최대 2개)
    - 핫 번호 집중 방지 (최근 5회차 2회 이상 출현 번호 최대 3개)
    - 게임 간 Jaccard 유사도 ≤ max_jaccard 후처리

    Args:
        seed:                 재현성용 랜덤 시드
        sum_percentile:       합계 허용 범위 백분위 (기본 (5.0, 95.0))
        max_per_decade:       십단위 구간당 최대 번호 수 (기본 3)
        min_decades:          최소 커버해야 할 십단위 구간 수 (기본 3)
        max_trailing_same:    같은 끝자리 최대 허용 수 (기본 2)
        max_prev_overlap:     이전 회차와 겹침 최대 수 (기본 2)
        max_hot_numbers:      핫 번호 최대 허용 수 (기본 3)
        hot_window:           핫 번호 판단 회차 수 (기본 5)
        hot_threshold:        핫 번호 출현 최소 횟수 (기본 2)
        recent_weight_window: 빈도 가중치 계산 최근 회차 수 (기본 100)
        max_jaccard:          게임 간 허용 최대 Jaccard 유사도 (기본 0.5)
    """

    name = "balanced_v2"

    def __init__(
        self,
        seed: int | None = None,
        sum_percentile: tuple[float, float] = (5.0, 95.0),
        max_per_decade: int = 3,
        min_decades: int = 3,
        max_trailing_same: int = 2,
        max_prev_overlap: int = 2,
        max_hot_numbers: int = 3,
        hot_window: int = 5,
        hot_threshold: int = 2,
        recent_weight_window: int = 100,
        max_jaccard: float = 0.5,
    ):
        self._rng                  = random.Random(seed)
        self._sum_pct_lo           = float(sum_percentile[0])
        self._sum_pct_hi           = float(sum_percentile[1])
        self._max_per_decade       = max_per_decade
        self._min_decades          = min_decades
        self._max_trailing_same    = max_trailing_same
        self._max_prev_overlap     = max_prev_overlap
        self._max_hot_numbers      = max_hot_numbers
        self._hot_window           = hot_window
        self._hot_threshold        = hot_threshold
        self._recent_weight_window = recent_weight_window
        self._max_jaccard          = max_jaccard

    # ── 핵심 메서드 ───────────────────────────────────────────────────────

    def generate(
        self,
        n_games: int = 5,
        history: pd.DataFrame | None = None,
    ) -> list[list[int]]:
        """n_games개의 게임을 생성하고 다양성 후처리를 수행한다."""
        ctx   = self._prepare_context(history)
        games = [self._generate_one(ctx) for _ in range(n_games)]
        return self._ensure_diversity(games, ctx)

    # ── 컨텍스트 준비 ─────────────────────────────────────────────────────

    def _prepare_context(self, history: pd.DataFrame | None) -> dict[str, Any]:
        """history에서 제약 계산용 컨텍스트를 준비한다.

        미래 데이터 누수 없음: 전달된 history는 항상 백테스트 시점 이전 데이터.
        """
        ctx: dict[str, Any] = {
            "sum_lo":    float(_SUM_LO_DEFAULT),
            "sum_hi":    float(_SUM_HI_DEFAULT),
            "weights":   None,
            "prev_nums": frozenset(),
            "hot_nums":  frozenset(),
        }
        if history is None or len(history) == 0:
            return ctx

        history = history.sort_values("round_no").reset_index(drop=True)

        # 1. 합계 백분위 (leakage-free: 제공된 history만 사용)
        sums = history[_NUM_COLS].sum(axis=1).values
        ctx["sum_lo"] = float(np.percentile(sums, self._sum_pct_lo))
        ctx["sum_hi"] = float(np.percentile(sums, self._sum_pct_hi))

        # 2. 최근 빈도 가중치 (recent_weight_window 회차 기준)
        recent  = history.tail(self._recent_weight_window)
        counter = Counter(recent[_NUM_COLS].values.flatten().tolist())
        total   = sum(counter.values()) or 1
        ctx["weights"] = {
            n: counter.get(n, 0) / total
            for n in range(LOTTO_MIN, LOTTO_MAX + 1)
        }

        # 3. 이전 회차 번호 (마지막 행)
        last_row = history.iloc[-1]
        ctx["prev_nums"] = frozenset(int(last_row[c]) for c in _NUM_COLS)

        # 4. 핫 번호 (최근 hot_window 회차 중 hot_threshold 이상 출현)
        last_n = history.tail(self._hot_window)
        cnt_n  = Counter(last_n[_NUM_COLS].values.flatten().tolist())
        ctx["hot_nums"] = frozenset(
            n for n, c in cnt_n.items() if c >= self._hot_threshold
        )

        return ctx

    # ── 제약 검증 ─────────────────────────────────────────────────────────

    def _is_valid(self, game: list[int], ctx: dict[str, Any]) -> bool:
        """모든 v2 제약 조건을 검증한다."""
        nums = sorted(game)

        # 1. 홀짝: 2:4, 3:3, 4:2 만 허용
        odd_count = sum(1 for n in nums if n % 2 == 1)
        if not (2 <= odd_count <= 4):
            return False

        # 2. 합계 범위 (history 분포 기반 동적 범위)
        s = sum(nums)
        if not (ctx["sum_lo"] <= s <= ctx["sum_hi"]):
            return False

        # 3. 십단위 구간: 구간당 최대 max_per_decade개 & 최소 min_decades구간
        decade_cnt = Counter(_DECADE_OF[n] for n in nums)
        if max(decade_cnt.values()) > self._max_per_decade:
            return False
        if len(decade_cnt) < self._min_decades:
            return False

        # 4. 연속 번호: 3개 이상 연속 금지
        consec = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                consec += 1
                if consec >= 3:
                    return False
            else:
                consec = 1

        # 5. 끝자리: 같은 끝자리 최대 max_trailing_same개
        trailing_cnt = Counter(n % 10 for n in nums)
        if max(trailing_cnt.values()) > self._max_trailing_same:
            return False

        # 6. 이전 회차 겹침 제한
        if ctx["prev_nums"] and len(set(nums) & ctx["prev_nums"]) > self._max_prev_overlap:
            return False

        # 7. 핫 번호 집중 방지
        if ctx["hot_nums"] and len(set(nums) & ctx["hot_nums"]) > self._max_hot_numbers:
            return False

        return True

    # ── 게임 생성 ─────────────────────────────────────────────────────────

    def _generate_one(self, ctx: dict[str, Any], max_retries: int = 500) -> list[int]:
        """제약을 만족하는 게임 하나를 생성한다."""
        for _ in range(max_retries):
            game = self._sample_with_decade_balance(ctx)
            if self._is_valid(game, ctx):
                return sorted(game)
        # 폴백: 순수 랜덤 (제약 불만족 가능성 있음)
        logger.warning("max_retries(%d) 초과 → 폴백 랜덤 반환", max_retries)
        return sorted(self._rng.sample(range(LOTTO_MIN, LOTTO_MAX + 1), NUMBERS_PER_GAME))

    def _sample_with_decade_balance(self, ctx: dict[str, Any]) -> list[int]:
        """최소 min_decades 이상 구간을 커버하도록 6개 번호를 샘플링한다."""
        weights = ctx.get("weights")
        pool    = list(range(LOTTO_MIN, LOTTO_MAX + 1))

        # min_decades 구간 각 1개씩 강제 선택 → 구간 커버 보장
        n_dec          = self._rng.randint(self._min_decades, 5)
        chosen_decades = self._rng.sample(range(1, 6), n_dec)
        selected: list[int] = []

        for d in chosen_decades[: self._min_decades]:
            band = [n for n in pool if _DECADE_OF[n] == d]
            bw   = ([weights[n] for n in band] if weights else None)
            pick = self._rng.choices(band, weights=bw, k=1)[0]
            selected.append(pick)

        # 나머지 슬롯: 전체 풀에서 가중치 샘플링 (중복 없이)
        remaining = [n for n in pool if n not in selected]
        rw: list[float] | None = (
            [weights[n] for n in remaining] if weights else None
        )
        while len(selected) < NUMBERS_PER_GAME and remaining:
            pick = self._rng.choices(remaining, weights=rw, k=1)[0]
            idx  = remaining.index(pick)
            selected.append(pick)
            remaining.pop(idx)
            if rw is not None:
                rw.pop(idx)

        return selected

    # ── 다양성 후처리 ─────────────────────────────────────────────────────

    def _ensure_diversity(
        self,
        games: list[list[int]],
        ctx: dict[str, Any],
        max_attempts: int = 100,
    ) -> list[list[int]]:
        """게임 간 과도한 유사도를 제거한다.

        1. 동일 게임 조합 → 재생성
        2. Jaccard > max_jaccard인 쌍의 두 번째 게임 재생성
        max_attempts 초과 시 현재 상태 그대로 반환.
        """
        games = [list(g) for g in games]

        for _ in range(max_attempts):
            replaced = False

            # ① 동일 게임 제거
            seen: set[tuple] = set()
            for i, g in enumerate(games):
                key = tuple(sorted(g))
                if key in seen:
                    games[i] = self._generate_one(ctx)
                    replaced  = True
                else:
                    seen.add(key)

            if replaced:
                continue

            # ② Jaccard 임계값 초과 쌍 탐색 (가장 유사한 쌍 우선 교체)
            worst_idx = -1
            worst_sim = -1.0
            for i in range(len(games)):
                for j in range(i + 1, len(games)):
                    a, b  = set(games[i]), set(games[j])
                    sim   = len(a & b) / len(a | b)
                    if sim > worst_sim:
                        worst_sim = sim
                        worst_idx = j

            if worst_sim <= self._max_jaccard:
                break  # 모든 쌍이 기준 이하

            games[worst_idx] = self._generate_one(ctx)

        return games
