"""전략 백테스트 엔진.

회차 t에 대한 테스트는 history[round_no < t]만 전략에 전달하여
미래 데이터 누수를 원천 차단한다.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


# ── 결과 데이터클래스 ─────────────────────────────────────────────────────

@dataclass
class RoundResult:
    """단일 회차 백테스트 결과."""
    round_no: int
    strategy: str
    games: list[list[int]]       # 생성된 n개 게임
    winning: list[int]            # 실제 당첨번호 6개
    bonus: int
    match_counts: list[int]      # 게임별 일치 번호 수
    diversity: dict = field(default_factory=dict)  # coverage/avg_jaccard/diversity_score

    @property
    def best_match(self) -> int:
        return max(self.match_counts) if self.match_counts else 0


@dataclass
class BacktestResult:
    """전략 백테스트 전체 결과."""
    strategy: str
    total_rounds: int
    per_round: list[RoundResult] = field(default_factory=list)

    @property
    def prize_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {"1등": 0, "2등": 0, "3등": 0, "4등": 0, "5등": 0, "꽝": 0}
        for rr in self.per_round:
            for match, game in zip(rr.match_counts, rr.games):
                has_bonus = rr.bonus in game
                p = _calc_prize(match, has_bonus)
                counts[p] = counts.get(p, 0) + 1
        return counts

    @property
    def best_match_distribution(self) -> dict[int, int]:
        dist: dict[int, int] = {i: 0 for i in range(7)}
        for rr in self.per_round:
            dist[rr.best_match] = dist.get(rr.best_match, 0) + 1
        return dist

    @property
    def avg_best_match(self) -> float:
        if not self.per_round:
            return 0.0
        return float(np.mean([rr.best_match for rr in self.per_round]))

    @property
    def match_3_plus(self) -> int:
        return sum(1 for rr in self.per_round if rr.best_match >= 3)

    @property
    def match_4_plus(self) -> int:
        return sum(1 for rr in self.per_round if rr.best_match >= 4)

    @property
    def match_5_plus(self) -> int:
        return sum(1 for rr in self.per_round if rr.best_match >= 5)

    @property
    def match_6_count(self) -> int:
        return sum(1 for rr in self.per_round if rr.best_match == 6)

    # ── 다양성 집계 ────────────────────────────────────────────────────────

    @property
    def avg_coverage(self) -> float:
        """회차별 사용 번호 coverage 평균 (사용 고유 번호 수 / 45)."""
        vals = [rr.diversity.get("coverage", 0.0) for rr in self.per_round if rr.diversity]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def avg_game_overlap(self) -> float:
        """회차별 게임 간 평균 Jaccard 유사도 평균."""
        vals = [rr.diversity.get("avg_jaccard", 0.0) for rr in self.per_round if rr.diversity]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def avg_diversity_score(self) -> float:
        """다양성 점수 평균 (0~100, 높을수록 다양)."""
        vals = [rr.diversity.get("diversity_score", 0.0) for rr in self.per_round if rr.diversity]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def game_stats(self) -> dict:
        """생성된 게임 전체의 통계 지표.

        Returns:
            avg_sum:             전 게임 평균 합계
            odd_distribution:    홀수 개수별 게임 수 {0~6: count}
            avg_decade_coverage: 게임당 평균 십단위 구간 커버 수 (최대 5)
            consecutive_rate:    연속 번호 쌍 포함 게임 비율
        """
        if not self.per_round:
            return {}

        all_games = [g for rr in self.per_round for g in rr.games]
        if not all_games:
            return {}

        def _decade_local(n: int) -> int:
            return 1 if n < 10 else n // 10 + 1

        def _has_consec(g: list[int]) -> bool:
            s = sorted(g)
            return any(s[i + 1] == s[i] + 1 for i in range(len(s) - 1))

        sums              = [sum(g) for g in all_games]
        odd_counts        = [sum(1 for n in g if n % 2 == 1) for g in all_games]
        decade_coverages  = [len({_decade_local(n) for n in g}) for g in all_games]
        has_consec_flags  = [_has_consec(g) for g in all_games]

        odd_dist: dict[int, int] = {}
        for c in odd_counts:
            odd_dist[c] = odd_dist.get(c, 0) + 1

        return {
            "avg_sum":             float(np.mean(sums)),
            "odd_distribution":    odd_dist,
            "avg_decade_coverage": float(np.mean(decade_coverages)),
            "consecutive_rate":    float(np.mean(has_consec_flags)),
        }


# ── 멀티 시드 결과 데이터클래스 ──────────────────────────────────────────

@dataclass
class StrategyMultiSeedStats:
    """전략의 멀티 시드 통계 — n_seeds 회 실행 결과를 집계한다."""

    strategy:     str
    n_seeds:      int
    total_rounds: int
    per_seed:     list[BacktestResult] = field(default_factory=list)

    # ── 시드별 값 목록 ─────────────────────────────────────────────────────

    @property
    def avg_best_match_values(self) -> list[float]:
        return [r.avg_best_match for r in self.per_seed]

    @property
    def match_3_plus_values(self) -> list[int]:
        return [r.match_3_plus for r in self.per_seed]

    @property
    def match_4_plus_values(self) -> list[int]:
        return [r.match_4_plus for r in self.per_seed]

    @property
    def match_5_plus_values(self) -> list[int]:
        return [r.match_5_plus for r in self.per_seed]

    @property
    def coverage_values(self) -> list[float]:
        return [r.avg_coverage for r in self.per_seed]

    @property
    def diversity_score_values(self) -> list[float]:
        return [r.avg_diversity_score for r in self.per_seed]

    # ── 집계 통계 ───────────────────────────────────────────────────────────

    @property
    def avg_best_match_mean(self) -> float:
        vals = self.avg_best_match_values
        return float(np.mean(vals)) if vals else 0.0

    @property
    def avg_best_match_std(self) -> float:
        vals = self.avg_best_match_values
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    @property
    def match_3_plus_mean(self) -> float:
        vals = self.match_3_plus_values
        return float(np.mean(vals)) if vals else 0.0

    @property
    def match_3_plus_std(self) -> float:
        vals = self.match_3_plus_values
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    @property
    def match_4_plus_mean(self) -> float:
        vals = self.match_4_plus_values
        return float(np.mean(vals)) if vals else 0.0

    @property
    def match_4_plus_std(self) -> float:
        vals = self.match_4_plus_values
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    @property
    def match_5_plus_mean(self) -> float:
        vals = self.match_5_plus_values
        return float(np.mean(vals)) if vals else 0.0

    @property
    def coverage_mean(self) -> float:
        vals = self.coverage_values
        return float(np.mean(vals)) if vals else 0.0

    @property
    def diversity_score_mean(self) -> float:
        vals = self.diversity_score_values
        return float(np.mean(vals)) if vals else 0.0


@dataclass
class MultiSeedResult:
    """멀티 시드 백테스트 전체 결과."""

    n_seeds:         int
    total_rounds:    int
    strategy_stats:  list[StrategyMultiSeedStats] = field(default_factory=list)

    def get_stats(self, strategy_name: str) -> "StrategyMultiSeedStats | None":
        for s in self.strategy_stats:
            if s.strategy == strategy_name:
                return s
        return None


# ── 내부 유틸 ─────────────────────────────────────────────────────────────

# compute_diversity는 ensemble_strategy에서 임포트 — 모듈 레벨로 올림
def _get_compute_diversity():
    from ..strategy.ensemble_strategy import compute_diversity as _cd
    return _cd


def _calc_prize(match_count: int, has_bonus: bool) -> str:
    """일치 번호 수와 보너스 여부로 등수를 반환한다."""
    if match_count == 6:
        return "1등"
    if match_count == 5 and has_bonus:
        return "2등"
    if match_count == 5:
        return "3등"
    if match_count == 4:
        return "4등"
    if match_count == 3:
        return "5등"
    return "꽝"


# ── 배치 사전 계산 ─────────────────────────────────────────────────────────

def _precompute_model_scores(
    history: pd.DataFrame,
    target_rounds: list[int],
    model: Any,
) -> dict[int, dict[int, float]]:
    """전체 target_rounds에 대한 모델 score를 배치로 사전 계산한다.

    build_features()의 벡터화 feature + 단일 배치 predict_proba 호출로
    회차별 45번 _compute_number_features 반복 호출을 완전히 제거한다.

    Returns:
        {round_no: {number(1..45): prob}} 딕셔너리
    """
    from .feature_builder import build_features, FEATURE_COLS

    t0 = time.monotonic()
    console.print("[dim]  → 모델 score 사전 계산 중 (feature 빌드)...[/dim]")

    # leakage-free feature 행렬을 벡터화로 한 번에 생성 (quiet 모드로 출력 억제)
    full_features = build_features(history, min_history_rounds=1, quiet=True)
    if full_features.empty:
        return {}

    # target_rounds에 해당하는 feature만 추출
    target_set = set(target_rounds)
    sub = full_features[full_features["target_round_no"].isin(target_set)]
    if sub.empty:
        return {}

    # 45 × len(target_rounds) 행을 한 번에 배치 예측
    X     = np.nan_to_num(sub[FEATURE_COLS].values.astype(np.float64), nan=0.0)
    probs = model.predict_proba(X)[:, 1]

    round_nos_arr = sub["target_round_no"].values
    numbers_arr   = sub["number"].values.astype(np.int32)

    scores_by_round: dict[int, dict[int, float]] = {}
    for i in range(len(probs)):
        rno = int(round_nos_arr[i])
        n   = int(numbers_arr[i])
        if rno not in scores_by_round:
            scores_by_round[rno] = {}
        scores_by_round[rno][n] = float(probs[i])

    elapsed = time.monotonic() - t0
    console.print(
        f"[dim]  ✓ 모델 score 사전 계산 완료: "
        f"{len(scores_by_round):,}회차 × 45번호, {elapsed:.2f}초[/dim]"
    )
    return scores_by_round


def _precompute_gap_weights(
    history: pd.DataFrame,
    target_rounds: list[int],
    alpha: float = 1.2,
    temperature: float = 2.0,
    max_gap_ratio: float = 2.5,
) -> dict[int, np.ndarray]:
    """전체 target_rounds에 대한 gap 기반 가중치를 벡터화로 사전 계산한다.

    회차별로 GapBasedStrategy._compute_weights()를 독립 호출하는 대신
    last_seen 누적 행렬을 O(n_rounds × 45) 한 번 빌드 후 O(1) 조회한다.

    Returns:
        {round_no: np.ndarray(shape=45, sum=1.0)} 딕셔너리
    """
    t0 = time.monotonic()
    console.print("[dim]  → Gap 가중치 사전 계산 중...[/dim]")

    history   = history.sort_values("round_no").reset_index(drop=True)
    n_rounds  = len(history)
    round_nos = history["round_no"].values
    num_cols  = [f"num{j}" for j in range(1, 7)]

    # appeared[i, n] = True if number n+1 appeared in round i
    appeared = np.zeros((n_rounds, 45), dtype=np.bool_)
    for col in num_cols:
        nums  = history[col].values.astype(int) - 1
        valid = (nums >= 0) & (nums < 45)
        appeared[np.where(valid)[0], nums[valid]] = True

    # last_seen[i, n] = 마지막으로 n이 출현한 행 인덱스 (rounds 0..i-1 기준), -1이면 미출현
    last_seen_mat = np.full((n_rounds + 1, 45), -1, dtype=np.int32)
    for i in range(1, n_rounds + 1):
        last_seen_mat[i] = np.where(appeared[i - 1], i - 1, last_seen_mat[i - 1])

    target_set = set(target_rounds)
    weights_by_round: dict[int, np.ndarray] = {}

    for i, rno in enumerate(round_nos):
        if rno not in target_set:
            continue

        ls   = last_seen_mat[i].astype(np.float64)
        # gap = 마지막 출현 이후 경과 회차; 미출현 시 = i (전체 이력 길이)
        gaps = np.where(ls >= 0.0, float(i - 1) - ls, float(i))

        # 극단 집중 방지: 평균 × max_gap_ratio 이상은 클리핑
        mean_gap = float(gaps.mean()) if i > 0 else 1.0
        cap      = mean_gap * max_gap_ratio
        gaps     = np.clip(gaps, 0.0, cap)

        # gap^alpha 비례 가중치 + 소프트맥스 temperature
        raw     = np.power(gaps + 1.0, alpha)
        scaled  = raw / max(temperature, 1e-8)
        scaled -= scaled.max()
        exp_s   = np.exp(scaled)
        weights_by_round[int(rno)] = (exp_s / exp_s.sum()).astype(np.float64)

    elapsed = time.monotonic() - t0
    console.print(
        f"[dim]  ✓ Gap 가중치 사전 계산 완료: "
        f"{len(weights_by_round):,}회차, {elapsed:.2f}초[/dim]"
    )
    return weights_by_round


def _load_strategy_model(
    strategy_name: str,
    model_path: "Path | None",
) -> Any:
    """전략에 필요한 모델을 로드한다.

    model_score → 필수 (없으면 ValueError).
    ensemble    → 선택적 (없으면 random 서브전략 대체).
    기타        → None 반환.
    """
    model = None
    if strategy_name == "model_score":
        if model_path is None:
            raise ValueError(
                "model_score 전략은 --model 경로가 필요합니다. "
                "`train-model` 명령을 먼저 실행하세요."
            )
        from .model_trainer import load_model
        model = load_model(model_path)
    elif strategy_name == "ensemble":
        if model_path is not None and Path(model_path).exists():
            from .model_trainer import load_model
            model = load_model(model_path)
        else:
            console.print(
                "[dim]ℹ ensemble: 모델 파일 없음 → model_score 서브전략 random으로 대체[/dim]"
            )
    return model


def _run_backtest_core(
    history: pd.DataFrame,
    strategy_name: str,
    target_rounds: list[int],
    model: Any,
    model_scores_cache: "dict[int, dict[int, float]]",
    gap_weights_cache: "dict[int, np.ndarray]",
    n_games: int,
    seed: "int | None",
    round_no_to_idx: "dict[int, int]",
    *,
    progress: Any = None,
    task: Any = None,
) -> BacktestResult:
    """백테스트 핵심 루프 — 사전 계산된 캐시를 사용하며 콘솔 출력 없음.

    run_backtest() / run_multiseed_backtest() 공용 내부 함수.
    progress/task를 전달하면 per-round 진행 상황을 업데이트한다.
    """
    compute_diversity = _get_compute_diversity()
    result = BacktestResult(strategy=strategy_name, total_rounds=len(target_rounds))

    for round_no in target_rounds:
        cur_idx = round_no_to_idx[round_no]
        past    = history.iloc[:cur_idx]
        current = history.iloc[cur_idx]

        winning = [int(current[f"num{j}"]) for j in range(1, 7)]
        bonus   = int(current["bonus"])

        scores_for_round  = model_scores_cache.get(round_no)
        weights_for_round = gap_weights_cache.get(round_no)

        strategy = _build_strategy(
            strategy_name, past, model, seed,
            model_scores=scores_for_round,
            gap_weights=weights_for_round,
        )
        games = strategy.generate(n_games=n_games, history=past)

        winning_set  = set(winning)
        match_counts = [len(set(g) & winning_set) for g in games]
        diversity    = compute_diversity(games)

        result.per_round.append(RoundResult(
            round_no=round_no,
            strategy=strategy_name,
            games=games,
            winning=winning,
            bonus=bonus,
            match_counts=match_counts,
            diversity=diversity,
        ))

        if progress is not None and task is not None:
            best  = max(match_counts)
            color = "green" if best >= 4 else "yellow" if best >= 3 else "dim"
            progress.update(
                task,
                advance=1,
                description=f"[{color}]회차 {round_no:>4} (최고:{best}개)[/{color}]",
            )

    return result


def _build_strategy(
    strategy_name: str,
    past: pd.DataFrame,
    model: Any,
    seed: int | None,
    *,
    model_scores: "dict[int, float] | None" = None,
    gap_weights: "np.ndarray | None" = None,
):
    """백테스트 루프에서 각 회차용 전략 인스턴스를 생성한다.

    model_scores / gap_weights가 제공된 경우 사전 계산된 값을 우선 사용한다.
    (온라인 폴백: model_scores/gap_weights가 None일 때만 실시간 계산)
    """
    from ..strategy.random_strategy import RandomStrategy
    from ..strategy.balanced_strategy import BalancedStrategy

    if strategy_name == "random":
        return RandomStrategy(seed=seed)

    if strategy_name == "balanced":
        return BalancedStrategy(seed=seed)

    if strategy_name == "balanced_v2":
        from ..strategy.balanced_v2_strategy import BalancedV2Strategy
        return BalancedV2Strategy(seed=seed)

    if strategy_name == "gap_based":
        from ..strategy.gap_based_strategy import GapBasedStrategy
        if gap_weights is not None:
            return GapBasedStrategy.from_weights(gap_weights, seed=seed)
        # 폴백: 실시간 계산 (사전 계산 없을 때만)
        return GapBasedStrategy(seed=seed)

    if strategy_name in ("model_score", "ensemble"):
        from ..strategy.model_score_strategy import ModelScoreStrategy

        scores: dict[int, float] | None = model_scores  # 사전 계산 우선

        # 사전 계산 없을 때만 온라인 계산 (느린 폴백)
        if scores is None and model is not None and len(past) > 0:
            from .feature_builder import _compute_number_features, FEATURE_COLS
            feature_rows = [
                [_compute_number_features(past, n)[c] for c in FEATURE_COLS]
                for n in range(1, 46)
            ]
            X = np.nan_to_num(np.array(feature_rows, dtype=float), nan=0.0)
            probs = model.predict_proba(X)[:, 1]
            scores = {n: float(probs[n - 1]) for n in range(1, 46)}

        if strategy_name == "model_score":
            if scores is None:
                return RandomStrategy(seed=seed)
            return ModelScoreStrategy.from_scores(scores, seed=seed)

        # ensemble: 사전 계산된 scores + gap_weights 주입
        from ..strategy.ensemble_strategy import EnsembleStrategy
        return EnsembleStrategy(model_scores=scores, gap_weights=gap_weights, seed=seed)

    raise ValueError(
        f"알 수 없는 전략: {strategy_name!r}. "
        "사용 가능: random, balanced, balanced_v2, gap_based, model_score, ensemble"
    )


# ── 핵심 공개 API ─────────────────────────────────────────────────────────

def run_backtest(
    history: pd.DataFrame,
    strategy_name: str,
    start_round: int | None = None,
    end_round: int | None = None,
    n_games: int = 5,
    min_history_rounds: int = 100,
    model_path: Path | None = None,
    seed: int | None = None,
) -> BacktestResult:
    """전략의 과거 성능을 시뮬레이션한다.

    회차 t의 테스트 시 history[round_no < t]만 전략에 전달하여 leakage를 방지한다.

    model_score 전략의 경우:
    - 매 회차 모델 재학습은 TODO로 남겨둔다
    - 사전 학습된 모델 + 각 회차 t-1까지의 feature 즉석 계산 (경량 방식)

    Args:
        history:            전체 당첨번호 DataFrame
        strategy_name:      전략 이름 (random / balanced / gap_based / model_score / ensemble)
        start_round:        시작 회차 (None → min_history_rounds 이후 첫 회차)
        end_round:          종료 회차 (None → 최신 회차)
        n_games:            회차당 생성 게임 수
        min_history_rounds: 전략에 제공할 최소 과거 회차 수
        model_path:         model_score / ensemble 전략용 모델 (.pkl) 경로
        seed:               재현성용 랜덤 시드

    Returns:
        BacktestResult

    성능:
        model_score / ensemble / gap_based 전략은 루프 전 사전 계산을 수행해
        회차당 feature 재계산을 제거한다 (300회차 기준 ~200× 이상 속도 향상).
    """
    history = history.sort_values("round_no").reset_index(drop=True)
    all_rounds = history["round_no"].values

    if len(all_rounds) == 0:
        raise ValueError("당첨번호 데이터가 없습니다.")

    # ── 범위 결정 ───────────────────────────────────────────────────────
    if end_round is None:
        end_round = int(all_rounds[-1])
    if start_round is None:
        if len(all_rounds) > min_history_rounds:
            start_round = int(all_rounds[min_history_rounds])
        else:
            start_round = int(all_rounds[0])

    target_rounds = [r for r in all_rounds if start_round <= r <= end_round]

    console.rule(f"[bold cyan]⏮ 백테스트 — {strategy_name}[/bold cyan]")
    console.print(
        f"  범위: [bold]{start_round:,}[/bold] ~ [bold]{end_round:,}[/bold] 회차  |  "
        f"총 [bold yellow]{len(target_rounds):,}[/bold yellow]회차  |  "
        f"{n_games}게임/회차"
    )

    # ── 모델 로드 ────────────────────────────────────────────────────────
    model: Any = _load_strategy_model(strategy_name, model_path)

    # ── 배치 사전 계산 (핵심 최적화) ─────────────────────────────────────
    #   model_score / ensemble: feature 행렬 1회 빌드 + 배치 predict_proba
    #   gap_based  / ensemble: last_seen 누적 행렬 1회 빌드 → 회차별 O(45) 가중치
    model_scores_cache: dict[int, dict[int, float]] = {}
    gap_weights_cache:  dict[int, np.ndarray]       = {}

    if strategy_name in ("model_score", "ensemble") and model is not None:
        model_scores_cache = _precompute_model_scores(history, target_rounds, model)

    if strategy_name in ("gap_based", "ensemble"):
        gap_weights_cache = _precompute_gap_weights(history, target_rounds)

    # ── 메인 루프 ─────────────────────────────────────────────────────────
    round_no_to_idx = {int(r): i for i, r in enumerate(history["round_no"].values)}
    t0 = time.monotonic()

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"{strategy_name} 백테스트", total=len(target_rounds))
        result = _run_backtest_core(
            history, strategy_name, target_rounds, model,
            model_scores_cache, gap_weights_cache, n_games, seed,
            round_no_to_idx,
            progress=progress, task=task,
        )

    elapsed = time.monotonic() - t0
    print_backtest_summary(result, elapsed)
    return result


def run_multiseed_backtest(
    history: pd.DataFrame,
    strategy_names: list[str],
    n_seeds: int = 10,
    start_round: int | None = None,
    end_round: int | None = None,
    n_games: int = 5,
    min_history_rounds: int = 100,
    model_path: Path | None = None,
) -> MultiSeedResult:
    """전략을 seed 1~n_seeds로 반복 실행해 통계적 안정성을 평가한다.

    핵심 최적화:
        model_score/ensemble의 feature 사전 계산 및 gap 가중치 사전 계산을
        전략당 1회만 수행하고, n_seeds 루프는 순수 샘플링 비용만 부담한다.

    Args:
        history:            전체 당첨번호 DataFrame
        strategy_names:     비교할 전략 이름 목록
        n_seeds:            반복 실행할 시드 수 (seed = 1 ~ n_seeds)
        start_round:        시작 회차 (None → min_history_rounds 이후 첫 회차)
        end_round:          종료 회차 (None → 최신 회차)
        n_games:            회차당 생성 게임 수
        min_history_rounds: 전략에 제공할 최소 과거 회차 수
        model_path:         model_score / ensemble 전략용 모델 (.pkl) 경로

    Returns:
        MultiSeedResult
    """
    history = history.sort_values("round_no").reset_index(drop=True)
    all_rounds = history["round_no"].values

    if len(all_rounds) == 0:
        raise ValueError("당첨번호 데이터가 없습니다.")

    # ── 범위 결정 (모든 전략 공통) ───────────────────────────────────────
    _end   = int(all_rounds[-1]) if end_round is None else end_round
    _start = start_round
    if _start is None:
        _start = int(all_rounds[min_history_rounds]) if len(all_rounds) > min_history_rounds else int(all_rounds[0])
    target_rounds = [int(r) for r in all_rounds if _start <= r <= _end]

    if not target_rounds:
        raise ValueError(f"대상 회차가 없습니다 (start={_start}, end={_end}).")

    total_tasks = len(strategy_names) * n_seeds

    console.rule("[bold cyan]📊 멀티 시드 백테스트[/bold cyan]")
    console.print(
        f"  전략: {', '.join(strategy_names)}  |  "
        f"범위: [bold]{_start:,}[/bold] ~ [bold]{_end:,}[/bold] 회차  |  "
        f"총 [bold yellow]{len(target_rounds):,}[/bold yellow]회차  |  "
        f"[bold green]{n_seeds}[/bold green] seeds  |  {n_games}게임/회차"
    )

    round_no_to_idx = {int(r): i for i, r in enumerate(history["round_no"].values)}
    msr = MultiSeedResult(n_seeds=n_seeds, total_rounds=len(target_rounds))
    t0  = time.monotonic()

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        outer = progress.add_task("멀티 시드 백테스트", total=total_tasks)

        for strategy_name in strategy_names:
            # ── 전략당 1회 모델 로드 + 사전 계산 ────────────────────────
            model: Any = _load_strategy_model(strategy_name, model_path)

            model_scores_cache: dict[int, dict[int, float]] = {}
            gap_weights_cache:  dict[int, np.ndarray]       = {}

            if strategy_name in ("model_score", "ensemble") and model is not None:
                model_scores_cache = _precompute_model_scores(history, target_rounds, model)
            if strategy_name in ("gap_based", "ensemble"):
                gap_weights_cache = _precompute_gap_weights(history, target_rounds)

            # ── seed 루프 (순수 샘플링 비용만) ───────────────────────────
            stats = StrategyMultiSeedStats(
                strategy=strategy_name,
                n_seeds=n_seeds,
                total_rounds=len(target_rounds),
            )
            for seed in range(1, n_seeds + 1):
                progress.update(
                    outer,
                    description=(
                        f"[cyan]{strategy_name}[/cyan] "
                        f"seed [bold]{seed:>3}[/bold]/{n_seeds}"
                    ),
                )
                result = _run_backtest_core(
                    history, strategy_name, target_rounds, model,
                    model_scores_cache, gap_weights_cache, n_games, seed,
                    round_no_to_idx,
                    # 멀티 시드 모드에서는 per-round 진행 표시 생략
                )
                stats.per_seed.append(result)
                progress.advance(outer)

            msr.strategy_stats.append(stats)

    elapsed = time.monotonic() - t0
    print_multiseed_summary(msr, elapsed)
    return msr


def save_backtest_results(result: BacktestResult, path: Path) -> None:
    """백테스트 결과를 CSV로 저장한다."""
    rows = []
    for rr in result.per_round:
        for g_idx, (game, match) in enumerate(zip(rr.games, rr.match_counts), 1):
            has_bonus = rr.bonus in game
            rows.append({
                "round_no":    rr.round_no,
                "strategy":    rr.strategy,
                "game_no":     g_idx,
                "num1": game[0], "num2": game[1], "num3": game[2],
                "num4": game[3], "num5": game[4], "num6": game[5],
                "match_count": match,
                "has_bonus":   int(has_bonus),
                "prize":       _calc_prize(match, has_bonus),
            })
    if rows:
        df = pd.DataFrame(rows)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 전략별 upsert (기존 다른 전략 결과는 유지)
        if path.exists():
            existing = pd.read_csv(path)
            combined = pd.concat(
                [existing[existing["strategy"] != result.strategy], df],
                ignore_index=True,
            )
        else:
            combined = df
        combined.to_csv(path, index=False, encoding="utf-8-sig")
        console.print(f"[green]✓ 백테스트 결과 저장: {path} ({len(df):,}행)[/green]")


def save_multiseed_results(result: MultiSeedResult, path: Path) -> None:
    """멀티 시드 백테스트 개별 결과를 CSV로 저장한다.

    컬럼: seed, round_no, strategy, game_no, num1~6, match_count, has_bonus, prize
    기존 backtest_results.csv 포맷과 호환 (seed 컬럼만 추가).
    """
    rows = []
    for stats in result.strategy_stats:
        for seed_no, bt_result in enumerate(stats.per_seed, 1):
            for rr in bt_result.per_round:
                for g_idx, (game, match) in enumerate(zip(rr.games, rr.match_counts), 1):
                    has_bonus = rr.bonus in game
                    rows.append({
                        "seed":        seed_no,
                        "round_no":    rr.round_no,
                        "strategy":    rr.strategy,
                        "game_no":     g_idx,
                        "num1": game[0], "num2": game[1], "num3": game[2],
                        "num4": game[3], "num5": game[4], "num6": game[5],
                        "match_count": match,
                        "has_bonus":   int(has_bonus),
                        "prize":       _calc_prize(match, has_bonus),
                    })
    if rows:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        console.print(f"[green]✓ 멀티 시드 결과 저장: {path} ({len(rows):,}행)[/green]")


def save_multiseed_summary(result: MultiSeedResult, path: Path) -> None:
    """멀티 시드 백테스트 전략별 요약 통계를 CSV로 저장한다.

    컬럼: strategy, n_seeds, total_rounds, 집계 지표 8개, vs_random_3plus_pct
    """
    random_stats = result.get_stats("random")
    rows = []
    for stats in result.strategy_stats:
        baseline = random_stats.match_3_plus_mean if random_stats else 0.0
        if baseline > 0 and stats.strategy != "random":
            vs_random = round((stats.match_3_plus_mean - baseline) / baseline * 100, 2)
        else:
            vs_random = float("nan")
        rows.append({
            "strategy":              stats.strategy,
            "n_seeds":               stats.n_seeds,
            "total_rounds":          stats.total_rounds,
            "avg_best_match_mean":   round(stats.avg_best_match_mean, 4),
            "avg_best_match_std":    round(stats.avg_best_match_std,  4),
            "match_3_plus_mean":     round(stats.match_3_plus_mean, 2),
            "match_3_plus_std":      round(stats.match_3_plus_std,  2),
            "match_4_plus_mean":     round(stats.match_4_plus_mean, 2),
            "match_4_plus_std":      round(stats.match_4_plus_std,  2),
            "coverage_mean":         round(stats.coverage_mean, 4),
            "diversity_score_mean":  round(stats.diversity_score_mean, 2),
            "vs_random_3plus_pct":   vs_random,
        })
    if rows:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        console.print(f"[green]✓ 멀티 시드 요약 저장: {path} ({len(rows)}행)[/green]")


def print_backtest_summary(result: BacktestResult, elapsed: float = 0.0) -> None:
    """Rich 패널로 백테스트 결과 요약을 출력한다."""
    dist  = result.best_match_distribution
    prize = result.prize_summary
    total = result.total_rounds

    table = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 2))
    table.add_column("항목", style="dim", min_width=18)
    table.add_column("값", style="bold")

    table.add_row("전략", result.strategy)
    table.add_row("테스트 회차", f"{total:,}회차")
    table.add_row("평균 최고 일치",  f"[cyan]{result.avg_best_match:.3f}[/cyan]개")

    pct3 = result.match_3_plus / total * 100 if total else 0
    table.add_row(
        "3개 이상 회차",
        f"{result.match_3_plus}회 ({pct3:.1f}%)",
    )
    table.add_row("4개 이상 회차", f"[yellow]{result.match_4_plus}[/yellow]회")
    table.add_row("5개 이상 회차", f"[orange3]{result.match_5_plus}[/orange3]회")
    table.add_row("6개 일치 회차",  f"[bold red]{result.match_6_count}[/bold red]회")

    # 일치 분포
    dist_str = "  ".join(
        f"{k}개:{v}" for k, v in sorted(dist.items()) if v > 0
    )
    table.add_row("일치 분포 (best)", dist_str)

    # 등수 요약
    prize_str = "  ".join(f"{k}:{v}" for k, v in prize.items() if v > 0)
    table.add_row("등수 요약", prize_str or "꽝만 있음")

    # 다양성 지표
    table.add_row("", "")  # 구분 공백
    cov_pct   = result.avg_coverage * 100
    overlap   = result.avg_game_overlap
    div_score = result.avg_diversity_score
    cov_color = "green" if cov_pct >= 55 else "yellow" if cov_pct >= 45 else "dim"
    ovl_color = "green" if overlap <= 0.25 else "yellow" if overlap <= 0.35 else "red"
    div_color = "green" if div_score >= 70 else "yellow" if div_score >= 60 else "dim"
    table.add_row(
        "번호 coverage",
        f"[{cov_color}]{cov_pct:.1f}%[/{cov_color}]  "
        f"[dim](고유 번호 수 / 45 평균)[/dim]",
    )
    table.add_row(
        "게임간 평균 overlap",
        f"[{ovl_color}]{overlap:.3f}[/{ovl_color}]  "
        f"[dim](Jaccard 유사도, 낮을수록 다양)[/dim]",
    )
    table.add_row(
        "다양성 점수",
        f"[{div_color}]{div_score:.1f}[/{div_color}]  "
        f"[dim](0~100, 높을수록 좋음)[/dim]",
    )

    # 게임 통계 (avg_sum, 홀짝 분포, 십단위 커버, 연속번호 포함률)
    stats = result.game_stats
    if stats:
        table.add_row("", "")  # 구분 공백
        avg_sum   = stats.get("avg_sum", 0.0)
        sum_color = "green" if 100 <= avg_sum <= 176 else "yellow"
        table.add_row(
            "평균 합계",
            f"[{sum_color}]{avg_sum:.1f}[/{sum_color}]  [dim](이론값: ~138)[/dim]",
        )
        odd_dist = stats.get("odd_distribution", {})
        odd_str  = "  ".join(
            f"홀{k}:{v}" for k, v in sorted(odd_dist.items()) if v > 0
        )
        table.add_row("홀수 개수 분포", odd_str or "-")
        dec_cov   = stats.get("avg_decade_coverage", 0.0)
        dec_color = "green" if dec_cov >= 3.5 else "yellow" if dec_cov >= 3.0 else "dim"
        table.add_row(
            "평균 십단위 구간",
            f"[{dec_color}]{dec_cov:.2f}[/{dec_color}]  [dim](최대 5구간)[/dim]",
        )
        consec_r  = stats.get("consecutive_rate", 0.0)
        con_color = "green" if consec_r <= 0.3 else "yellow" if consec_r <= 0.5 else "red"
        table.add_row(
            "연속번호 포함 게임",
            f"[{con_color}]{consec_r:.1%}[/{con_color}]",
        )

    if elapsed > 0:
        table.add_row("소요 시간", f"{elapsed:.1f}초")

    border = "green" if result.match_4_plus > 0 else "blue"
    console.print(
        Panel(
            table,
            title="[bold]⏮ 백테스트 결과[/bold]",
            border_style=border,
            padding=(1, 2),
        )
    )


def print_multiseed_summary(result: MultiSeedResult, elapsed: float = 0.0) -> None:
    """Rich 테이블로 멀티 시드 백테스트 요약을 출력한다.

    전략별 평균±std 지표를 7컬럼 테이블로 출력하고,
    random 대비 개선율(3개↑ 기준)은 테이블 아래 별도 줄로 표시한다.
    7컬럼 구성: 전략(11) 최고일치(9) 3개↑(8) 3개↑%(6) 4개↑(7) cover%(6) 다양성(6)
      padding (0,1)×7col×2=14 → 합계 67자 — 80자 터미널에서 여유롭게 표시됨.
    """
    random_stats = result.get_stats("random")

    # ── 제목 구분선 ───────────────────────────────────────────────────────
    console.rule(
        f"[bold cyan]📊 멀티 시드 백테스트 결과[/bold cyan]"
        f"  [dim]{result.n_seeds} seeds × {result.total_rounds:,}회차[/dim]"
    )

    # ── 7컬럼 지표 테이블 (vs-random은 아래 별도 출력) ────────────────────
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
        expand=False,
    )
    table.add_column("전략",            style="bold",  min_width=11, no_wrap=True)
    table.add_column("최고일치\n평균±σ", justify="right", min_width=9,  no_wrap=True)
    table.add_column("3개↑\n평균±σ",    justify="right", min_width=8,  no_wrap=True)
    table.add_column("3개↑%",           justify="right", min_width=6,  no_wrap=True)
    table.add_column("4개↑\n평균±σ",    justify="right", min_width=7,  no_wrap=True)
    table.add_column("cover%",          justify="right", min_width=6,  no_wrap=True)
    table.add_column("다양성",          justify="right", min_width=6,  no_wrap=True)

    vs_parts: list[str] = []   # vs-random 비교 텍스트를 모아 하단에 출력

    for stats in result.strategy_stats:
        m3_mean  = stats.match_3_plus_mean
        m3_std   = stats.match_3_plus_std
        m3_pct   = m3_mean / result.total_rounds * 100 if result.total_rounds else 0
        m4_mean  = stats.match_4_plus_mean
        m4_std   = stats.match_4_plus_std
        cov_pct  = stats.coverage_mean * 100
        div      = stats.diversity_score_mean

        bm_str = f"{stats.avg_best_match_mean:.2f}±{stats.avg_best_match_std:.2f}"

        m3_color = "green" if m3_pct >= 13 else "yellow" if m3_pct >= 9 else "dim"
        m3_str   = f"[{m3_color}]{m3_mean:.1f}±{m3_std:.1f}[/{m3_color}]"
        m3pct_str = f"[{m3_color}]{m3_pct:.1f}%[/{m3_color}]"

        m4_color = "green" if m4_mean >= 3 else "yellow" if m4_mean >= 1 else "dim"
        m4_str   = f"[{m4_color}]{m4_mean:.1f}±{m4_std:.1f}[/{m4_color}]"

        cov_color = "green" if cov_pct >= 55 else "yellow" if cov_pct >= 45 else "dim"
        cov_str   = f"[{cov_color}]{cov_pct:.1f}%[/{cov_color}]"

        div_color = "green" if div >= 70 else "yellow" if div >= 60 else "dim"
        div_str   = f"[{div_color}]{div:.1f}[/{div_color}]"

        table.add_row(stats.strategy, bm_str, m3_str, m3pct_str, m4_str, cov_str, div_str)

        # vs-random 비교 텍스트 누적
        if random_stats is not None and stats.strategy != "random":
            baseline = random_stats.match_3_plus_mean
            if baseline > 0:
                delta    = (m3_mean - baseline) / baseline * 100
                vs_color = "green" if delta > 2 else "red" if delta < -2 else "dim"
                vs_parts.append(f"{stats.strategy} [{vs_color}]{delta:+.1f}%[/{vs_color}]")

    console.print(table)

    # ── vs random 비교 (3개↑ 기준) ────────────────────────────────────────
    if vs_parts:
        console.print(
            "  [dim]vs random (3개↑):[/dim]  " + "  |  ".join(vs_parts)
        )

    if elapsed > 0:
        console.print(
            f"  [dim]소요 시간: {elapsed:.1f}초  |  "
            f"전략 {len(result.strategy_stats)}개 × {result.n_seeds} seeds[/dim]"
        )
