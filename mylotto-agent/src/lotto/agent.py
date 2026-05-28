"""LottoAgent — 수집·생성·저장·분석·학습·백테스트를 조율하는 오케스트레이터."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
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
from rich.text import Text

from .collector.dh_collector import CollectStats, DhCollector, FetchResult
from .collector.lotto_kr_collector import LottoKrCollector
from .storage.local_storage import LocalStorage
from .strategy.balanced_strategy import BalancedStrategy
from .strategy.random_strategy import RandomStrategy

logger = logging.getLogger(__name__)
console = Console()

# 기본 경로
DEFAULT_RESULTS_CSV   = Path("data/lotto_draw_results.csv")
DEFAULT_GAMES_CSV     = Path("data/generated_games.csv")
DEFAULT_STATS_CSV     = Path("data/number_stats.csv")
DEFAULT_FEATURES_PQ   = Path("data/features.parquet")
DEFAULT_MODEL_PATH    = Path("data/models/lr_model.pkl")
DEFAULT_BACKTEST_CSV  = Path("data/backtest_results.csv")


# ── 내부 유틸 ──────────────────────────────────────────────────────────────

def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}초"
    m, s = divmod(int(seconds), 60)
    return f"{m}분 {s}초"


def _build_progress() -> Progress:
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


# ── Agent ──────────────────────────────────────────────────────────────────

class LottoAgent:
    """로또 에이전트 — 수집·생성·분석·학습·백테스트를 단일 인터페이스로 제공한다."""

    def __init__(
        self,
        results_csv:    str | Path = DEFAULT_RESULTS_CSV,
        games_csv:      str | Path = DEFAULT_GAMES_CSV,
        stats_csv:      str | Path = DEFAULT_STATS_CSV,
        features_path:  str | Path = DEFAULT_FEATURES_PQ,
        model_path:     str | Path = DEFAULT_MODEL_PATH,
        backtest_csv:   str | Path = DEFAULT_BACKTEST_CSV,
        use_fallback:   bool = True,
    ):
        self.storage        = LocalStorage(results_csv, games_csv)
        self.stats_path     = Path(stats_csv)
        self.features_path  = Path(features_path)
        self.model_path     = Path(model_path)
        self.backtest_csv   = Path(backtest_csv)

        self._dh_collector  = DhCollector()
        self._kr_collector  = LottoKrCollector()
        self.collector      = self._resolve_collector(use_fallback)

    # ──────────────────────────────────────────────────────────────────────
    # 1. 수집기 선택
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_collector(self, use_fallback: bool):
        if not use_fallback:
            return self._dh_collector
        try:
            r = self._dh_collector._session.get(
                "https://www.dhlottery.co.kr/common.do",
                params={"method": "getLottoNumber", "drwNo": 1},
                timeout=5,
            )
            if r.status_code == 200 and "returnValue" in r.text:
                logger.info("수집기: DhCollector (동행복권 공식 API)")
                return self._dh_collector
        except Exception:
            pass
        logger.info("수집기: LottoKrCollector (lotto.co.kr 스크래핑)")
        console.print("[dim]⚡ 동행복권 공식 API 미응답 → lotto.co.kr 스크래핑 모드[/dim]")
        return self._kr_collector

    # ──────────────────────────────────────────────────────────────────────
    # 2. 당첨번호 수집
    # ──────────────────────────────────────────────────────────────────────

    def collect(self, force_full: bool = False) -> pd.DataFrame:
        """동행복권에서 최신 당첨번호를 수집하여 CSV를 업데이트한다."""
        console.rule("[bold cyan]📡 동행복권 당첨번호 수집[/bold cyan]")

        try:
            with console.status("[bold yellow]최신 회차 탐색 중...[/bold yellow]", spinner="dots"):
                latest_api = self.collector.fetch_latest_round()
        except requests.exceptions.ConnectionError:
            console.print("[red]✗ 네트워크 연결 오류.[/red]")
            raise
        except Exception as exc:
            console.print(f"[red]✗ 최신 회차 탐색 실패: {exc}[/red]")
            raise

        latest_local = 0 if force_full else self.storage.get_latest_round()
        if latest_api <= latest_local:
            console.print(
                f"[green]✓ 이미 최신 상태 "
                f"(로컬: {latest_local:,}회차 / API: {latest_api:,}회차)[/green]"
            )
            return self.storage.load_results()

        start      = 1 if force_full else latest_local + 1
        total      = latest_api - start + 1
        mode_label = "[bold red]전체 재수집[/bold red]" if force_full else "증분 수집"
        console.print(
            f"  API 최신: [bold]{latest_api:,}회차[/bold]  |  "
            f"로컬 최신: [bold]{latest_local:,}회차[/bold]  |  "
            f"수집 대상: [bold yellow]{total:,}회차[/bold yellow]  |  "
            f"모드: {mode_label}"
        )

        stats = CollectStats(start_round=start, end_round=latest_api)
        with _build_progress() as progress:
            task = progress.add_task(f"회차 {start:>4} 대기 중", total=total)

            def _on_progress(result: FetchResult) -> None:
                nonlocal stats
                if result.is_valid:
                    icon, color = "✓", "green"
                elif result.success and result.validation_errors:
                    icon, color = "⚠", "yellow"
                else:
                    icon, color = "✗", "red"
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"[{color}]{icon}[/{color}] "
                        f"회차 [bold]{result.round_no:>4}[/bold]"
                    ),
                )

            new_df, stats = self.collector.collect_range(
                start, latest_api, on_progress=_on_progress
            )

        if not new_df.empty:
            updated = self.storage.upsert_results(new_df)
        else:
            console.print("[yellow]⚠ 수집된 데이터가 없어 저장을 건너뜁니다.[/yellow]")
            updated = self.storage.load_results()

        self._print_collect_summary(stats, updated, self.storage.results_path)
        return updated

    # ──────────────────────────────────────────────────────────────────────
    # 3. 번호 생성
    # ──────────────────────────────────────────────────────────────────────

    def generate(
        self,
        n_games: int = 5,
        strategies: list[str] | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """전략별로 로또 번호를 생성하고 CSV에 저장한다."""
        if strategies is None:
            strategies = ["random", "balanced"]

        history      = self.storage.load_results()
        generated_at = datetime.now(tz=timezone.utc).isoformat()

        rows: list[dict] = []
        for strategy_name in strategies:
            strategy = self._build_strategy(strategy_name, seed, model_path=self.model_path)

            diversity: dict | None = None
            if strategy_name == "model_score":
                games, scores = strategy.generate_with_scores(
                    n_games=n_games, history=history
                )
            elif strategy_name == "ensemble":
                games, diversity = strategy.generate_with_diversity(history=history)
                scores = {}
            else:
                games  = strategy.generate(n_games=n_games, history=history)
                scores = {}

            console.print(
                f"\n[bold magenta]🎲 {strategy_name} 전략 — {len(games)}게임[/bold magenta]"
            )
            self._print_games(games, strategy_name, scores=scores)

            # ensemble: 다양성 지표 추가 출력
            if strategy_name == "ensemble" and diversity:
                self._print_diversity(diversity)

            for game_no, numbers in enumerate(games, 1):
                rows.append({
                    "generated_at": generated_at,
                    "strategy":     strategy_name,
                    "game_no":      game_no,
                    "num1": numbers[0], "num2": numbers[1], "num3": numbers[2],
                    "num4": numbers[3], "num5": numbers[4], "num6": numbers[5],
                })

        games_df = pd.DataFrame(rows)
        self.storage.append_games(games_df)
        console.print(
            f"\n[green]✓ 총 {len(games_df)}게임 저장: {self.storage.games_path}[/green]"
        )
        return games_df

    # ──────────────────────────────────────────────────────────────────────
    # 4. 최신 당첨번호 조회
    # ──────────────────────────────────────────────────────────────────────

    def show_latest(self, n: int = 5) -> pd.DataFrame:
        """저장된 최신 n회차 당첨번호를 출력하고 반환한다."""
        df = self.storage.load_results()
        if df.empty:
            console.print("[red]저장된 당첨번호가 없습니다. `collect` 명령을 먼저 실행하세요.[/red]")
            return df
        latest = df.nlargest(n, "round_no")
        self._print_results(latest)
        return latest

    # ──────────────────────────────────────────────────────────────────────
    # 5. 통계 분석
    # ──────────────────────────────────────────────────────────────────────

    def analyze(self, top_n: int = 10, save: bool = True) -> pd.DataFrame:
        """번호별 출현빈도·gap·홀짝·합계·구간 통계를 분석하고 출력한다."""
        from .analysis.stats_analyzer import StatsAnalyzer

        history = self.storage.load_results()
        if history.empty:
            console.print("[red]당첨번호 데이터가 없습니다. `collect` 명령을 먼저 실행하세요.[/red]")
            return pd.DataFrame()

        analyzer = StatsAnalyzer(history)
        analyzer.print_summary(top_n=top_n)

        if save:
            analyzer.save(self.stats_path)

        return analyzer.compute_number_stats()

    # ──────────────────────────────────────────────────────────────────────
    # 6. Feature 생성
    # ──────────────────────────────────────────────────────────────────────

    def build_features(self, min_history_rounds: int = 20) -> pd.DataFrame:
        """Leakage-free feature 행렬을 생성하고 parquet으로 저장한다."""
        from .ml.feature_builder import build_features, save_features

        history = self.storage.load_results()
        if history.empty:
            console.print("[red]당첨번호 데이터가 없습니다. `collect` 명령을 먼저 실행하세요.[/red]")
            return pd.DataFrame()

        features_df = build_features(history, min_history_rounds=min_history_rounds)
        if not features_df.empty:
            save_features(features_df, self.features_path)
        return features_df

    # ──────────────────────────────────────────────────────────────────────
    # 7. 모델 학습
    # ──────────────────────────────────────────────────────────────────────

    def train_model(self, val_ratio: float = 0.15, C: float = 0.1) -> None:
        """저장된 feature를 사용해 LogisticRegression을 학습한다."""
        from .ml.model_trainer import train_model as _train
        from .ml.feature_builder import load_features

        if not self.features_path.exists():
            console.print(
                "[red]Feature 파일이 없습니다. `build-features` 명령을 먼저 실행하세요.[/red]"
            )
            return

        features_df = load_features(self.features_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        _train(
            features_df=features_df,
            model_path=self.model_path,
            val_ratio=val_ratio,
            C=C,
        )

    # ──────────────────────────────────────────────────────────────────────
    # 8. 백테스트
    # ──────────────────────────────────────────────────────────────────────

    def backtest(
        self,
        strategy_names: list[str] | None = None,
        start_round: int | None = None,
        end_round: int | None = None,
        recent: int | None = None,
        n_games: int = 5,
        seed: int | None = None,
        save: bool = True,
    ) -> dict[str, "BacktestResult"]:
        """전략별 백테스트를 실행하고 결과를 반환한다."""
        from .ml.backtest import run_backtest, save_backtest_results, BacktestResult

        if strategy_names is None:
            strategy_names = ["random", "balanced"]

        history = self.storage.load_results()
        if history.empty:
            console.print("[red]당첨번호 데이터가 없습니다.[/red]")
            return {}

        # --recent 처리
        if recent is not None and start_round is None:
            latest = int(history["round_no"].max())
            start_round = latest - recent + 1

        results: dict[str, BacktestResult] = {}
        for name in strategy_names:
            model_path_arg = self.model_path if name in ("model_score", "ensemble") else None
            result = run_backtest(
                history=history,
                strategy_name=name,
                start_round=start_round,
                end_round=end_round,
                n_games=n_games,
                model_path=model_path_arg,
                seed=seed,
            )
            results[name] = result
            if save:
                save_backtest_results(result, self.backtest_csv)

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_collect_summary(
        stats: CollectStats,
        all_df: pd.DataFrame,
        save_path: Path | None = None,
    ) -> None:
        table = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 2), expand=False)
        table.add_column("항목", style="dim", min_width=16)
        table.add_column("값", style="bold")

        rate       = stats.success_rate
        rate_color = "green" if rate >= 99 else "yellow" if rate >= 90 else "red"

        table.add_row("수집 범위", f"{stats.start_round:,} ~ {stats.end_round:,} 회차")
        table.add_row("수집 성공", f"[green]{stats.collected:,}[/green] 회차")
        table.add_row(
            "실패 (API 오류)",
            f"[{'red' if stats.skipped else 'dim'}]{stats.skipped:,}[/{'red' if stats.skipped else 'dim'}] 회차",
        )
        table.add_row(
            "실패 (검증 오류)",
            f"[{'yellow' if stats.invalid else 'dim'}]{stats.invalid:,}[/{'yellow' if stats.invalid else 'dim'}] 회차",
        )
        table.add_row("성공률",   f"[{rate_color}]{rate:.1f}%[/{rate_color}]")
        table.add_row("소요 시간", _fmt_elapsed(stats.elapsed))
        table.add_row("누적 저장", f"[cyan]{len(all_df):,}[/cyan] 회차")
        table.add_row("저장 위치", Text(str(save_path or "data/lotto_draw_results.csv"), style="dim"))

        if stats.failed_rounds:
            shown = stats.failed_rounds[:10]
            extra = len(stats.failed_rounds) - 10
            failed_str = ", ".join(str(r) for r in shown)
            if extra > 0:
                failed_str += f" … 외 {extra}건"
            table.add_row("실패 회차", f"[red]{failed_str}[/red]")

        border = "green" if stats.skipped == 0 and stats.invalid == 0 else "yellow"
        console.print(
            Panel(table, title="[bold green]✓ 수집 완료[/bold green]",
                  border_style=border, padding=(1, 2))
        )

    @staticmethod
    def _build_strategy(
        name: str,
        seed: int | None,
        model_path: Path | None = None,
    ):
        """전략 이름으로 Strategy 인스턴스를 생성한다."""
        name_lower = name.lower()
        if name_lower == "random":
            return RandomStrategy(seed=seed)
        if name_lower == "balanced":
            return BalancedStrategy(seed=seed)
        if name_lower == "gap_based":
            from .strategy.gap_based_strategy import GapBasedStrategy
            return GapBasedStrategy(seed=seed)
        if name_lower == "model_score":
            if model_path is None or not model_path.exists():
                raise ValueError(
                    "model_score 전략은 모델 파일이 필요합니다. "
                    "`train-model` 명령을 먼저 실행하세요.\n"
                    f"  찾은 경로: {model_path}"
                )
            from .strategy.model_score_strategy import ModelScoreStrategy
            return ModelScoreStrategy(model_path=model_path, seed=seed)
        if name_lower == "ensemble":
            from .strategy.ensemble_strategy import EnsembleStrategy
            if model_path is not None and model_path.exists():
                return EnsembleStrategy(model_path=model_path, seed=seed)
            console.print(
                "[dim]ℹ ensemble: 모델 없음 → model_score 서브전략 random으로 대체[/dim]"
            )
            return EnsembleStrategy(seed=seed)
        raise ValueError(
            f"알 수 없는 전략: {name!r}. "
            "사용 가능: random, balanced, gap_based, model_score, ensemble"
        )

    @staticmethod
    def _print_games(
        games: list[list[int]],
        strategy: str,
        scores: dict[int, float] | None = None,
    ) -> None:
        table = Table(
            title=f"전략: {strategy}",
            show_header=True,
            header_style="bold blue",
            box=box.ROUNDED,
        )
        table.add_column("게임", style="dim", width=6)
        for i in range(1, 7):
            table.add_column(f"번호{i}", justify="right")

        for i, nums in enumerate(games, 1):
            table.add_row(str(i), *[str(n) for n in nums])

        console.print(table)

    @staticmethod
    def _print_diversity(diversity: dict) -> None:
        """앙상블 다양성 지표를 Rich 테이블로 출력한다."""
        from rich import box as _box

        cov   = diversity.get("coverage", 0.0)
        jac   = diversity.get("avg_jaccard", 0.0)
        score = diversity.get("diversity_score", 0.0)

        table = Table(
            title="앙상블 다양성 지표",
            show_header=True,
            header_style="bold cyan",
            box=_box.SIMPLE,
        )
        table.add_column("지표",    style="dim",  min_width=20)
        table.add_column("값",      justify="right")
        table.add_column("설명",    style="dim")

        cov_color   = "green" if cov >= 0.55   else "yellow" if cov >= 0.45   else "white"
        jac_color   = "green" if jac <= 0.25   else "yellow" if jac <= 0.35   else "red"
        score_color = "green" if score >= 70.0 else "yellow" if score >= 60.0 else "white"

        table.add_row(
            "번호 coverage",
            f"[{cov_color}]{cov:.1%}[/{cov_color}]",
            f"고유 번호 {int(round(cov * 45))}개 / 45",
        )
        table.add_row(
            "게임간 평균 overlap",
            f"[{jac_color}]{jac:.3f}[/{jac_color}]",
            "Jaccard 유사도 (낮을수록 다양)",
        )
        table.add_row(
            "다양성 점수",
            f"[{score_color}]{score:.1f}[/{score_color}]",
            "0~100 (높을수록 다양)",
        )
        console.print(table)

    @staticmethod
    def _print_results(df: pd.DataFrame) -> None:
        cols = ["round_no", "date", "num1", "num2", "num3", "num4", "num5", "num6", "bonus"]
        table = Table(
            title="최신 당첨번호",
            show_header=True,
            header_style="bold green",
            box=box.ROUNDED,
        )
        for col in cols:
            table.add_column(col, justify="right")
        for _, row in df.sort_values("round_no", ascending=False).iterrows():
            table.add_row(*[str(row[c]) for c in cols])
        console.print(table)
