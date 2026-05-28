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


# ── 내부 유틸 ─────────────────────────────────────────────────────────────

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


def _build_strategy(
    strategy_name: str,
    past: pd.DataFrame,
    model: Any,
    seed: int | None,
):
    """백테스트 루프에서 각 회차용 전략 인스턴스를 생성한다."""
    from ..strategy.random_strategy import RandomStrategy
    from ..strategy.balanced_strategy import BalancedStrategy

    if strategy_name == "random":
        return RandomStrategy(seed=seed)

    if strategy_name == "balanced":
        return BalancedStrategy(seed=seed)

    if strategy_name == "gap_based":
        from ..strategy.gap_based_strategy import GapBasedStrategy
        return GapBasedStrategy(seed=seed)

    if strategy_name in ("model_score", "ensemble"):
        from ..strategy.model_score_strategy import ModelScoreStrategy
        from .feature_builder import _compute_number_features, FEATURE_COLS

        # 모델 없거나 과거 데이터 부족 시 랜덤 폴백 (model_score)
        # ensemble은 model_scores=None으로 GapBased+Balanced+Random만 사용
        scores: dict[int, float] | None = None
        if model is not None and len(past) > 0:
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

        # ensemble
        from ..strategy.ensemble_strategy import EnsembleStrategy
        return EnsembleStrategy(model_scores=scores, seed=seed)

    raise ValueError(
        f"알 수 없는 전략: {strategy_name!r}. "
        "사용 가능: random, balanced, gap_based, model_score, ensemble"
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
        strategy_name:      전략 이름 (random / balanced / model_score)
        start_round:        시작 회차 (None → min_history_rounds 이후 첫 회차)
        end_round:          종료 회차 (None → 최신 회차)
        n_games:            회차당 생성 게임 수
        min_history_rounds: 전략에 제공할 최소 과거 회차 수
        model_path:         model_score 전략용 모델 (.pkl) 경로
        seed:               재현성용 랜덤 시드

    Returns:
        BacktestResult
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

    # model_score / ensemble: 모델 미리 로드
    model: Any = None
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

    result = BacktestResult(strategy=strategy_name, total_rounds=len(target_rounds))
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

        for round_no in target_rounds:
            # leakage 방지: round_no 이전 데이터만 전달
            past    = history[history["round_no"] < round_no].copy()
            current = history[history["round_no"] == round_no].iloc[0]

            winning = [int(current[f"num{j}"]) for j in range(1, 7)]
            bonus   = int(current["bonus"])

            strategy = _build_strategy(strategy_name, past, model, seed)
            games    = strategy.generate(n_games=n_games, history=past)

            winning_set  = set(winning)
            match_counts = [len(set(g) & winning_set) for g in games]

            # 다양성 지표 계산
            from ..strategy.ensemble_strategy import compute_diversity
            diversity = compute_diversity(games)

            result.per_round.append(RoundResult(
                round_no=round_no,
                strategy=strategy_name,
                games=games,
                winning=winning,
                bonus=bonus,
                match_counts=match_counts,
                diversity=diversity,
            ))

            best  = max(match_counts)
            color = "green" if best >= 4 else "yellow" if best >= 3 else "dim"
            progress.update(
                task,
                advance=1,
                description=f"[{color}]회차 {round_no:>4} (최고:{best}개)[/{color}]",
            )

    elapsed = time.monotonic() - t0
    print_backtest_summary(result, elapsed)
    return result


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
