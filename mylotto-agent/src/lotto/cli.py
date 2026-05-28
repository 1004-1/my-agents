"""Typer CLI — `python main.py <command>` 또는 `lotto <command>`."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(
    name="lotto",
    help="🎱 로또 번호 생성 & 당첨번호 수집 Agent",
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()

# ── 공통 옵션 타입 별칭 ──────────────────────────────────────────────────
ResultsCsvOpt = Annotated[
    Path,
    typer.Option("--results", "-r", help="당첨번호 CSV 경로"),
]
GamesCsvOpt = Annotated[
    Path,
    typer.Option("--games", "-g", help="생성게임 CSV 경로"),
]
FeaturesOpt = Annotated[
    Path,
    typer.Option("--features", help="Feature parquet 경로"),
]
ModelOpt = Annotated[
    Path,
    typer.Option("--model", help="모델 .pkl 경로"),
]
VerboseOpt = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="디버그 로그 출력"),
]

_DEFAULT_RESULTS  = Path(os.getenv("LOTTO_RESULTS_CSV",   "data/lotto_draw_results.csv"))
_DEFAULT_GAMES    = Path(os.getenv("GENERATED_GAMES_CSV", "data/generated_games.csv"))
_DEFAULT_FEATURES = Path("data/features.parquet")
_DEFAULT_MODEL    = Path("data/models/lr_model.pkl")
_DEFAULT_BACKTEST = Path("data/backtest_results.csv")
_DEFAULT_STATS    = Path("data/number_stats.csv")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_agent(
    results: Path = _DEFAULT_RESULTS,
    games: Path = _DEFAULT_GAMES,
    features: Path = _DEFAULT_FEATURES,
    model: Path = _DEFAULT_MODEL,
    stats: Path = _DEFAULT_STATS,
    backtest_csv: Path = _DEFAULT_BACKTEST,
):
    from .agent import LottoAgent
    return LottoAgent(
        results_csv=results,
        games_csv=games,
        stats_csv=stats,
        features_path=features,
        model_path=model,
        backtest_csv=backtest_csv,
    )


# ══════════════════════════════════════════════════════════════════════════
# Commands
# ══════════════════════════════════════════════════════════════════════════

@app.command()
def collect(
    results: ResultsCsvOpt = _DEFAULT_RESULTS,
    games:   GamesCsvOpt   = _DEFAULT_GAMES,
    full:    Annotated[bool, typer.Option("--full", help="전체 회차를 처음부터 수집")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """📡 동행복권에서 최신 당첨번호를 수집하여 CSV를 업데이트한다."""
    _setup_logging(verbose)
    _make_agent(results, games).collect(force_full=full)


@app.command()
def generate(
    n_games:  Annotated[int, typer.Option("--n-games", "-n", help="생성할 게임 수")] = 5,
    strategy: Annotated[
        Optional[list[str]],
        typer.Option("--strategy", "-s", help="전략 (random/balanced/gap_based/model_score/ensemble, 반복 가능)"),
    ] = None,
    seed:    Annotated[Optional[int], typer.Option("--seed", help="랜덤 시드")] = None,
    results: ResultsCsvOpt = _DEFAULT_RESULTS,
    games:   GamesCsvOpt   = _DEFAULT_GAMES,
    model:   ModelOpt      = _DEFAULT_MODEL,
    verbose: VerboseOpt = False,
) -> None:
    """🎲 로또 번호를 전략별로 생성하여 CSV에 저장한다."""
    _setup_logging(verbose)
    strategies = strategy or ["random", "balanced"]
    _make_agent(results, games, model=model).generate(
        n_games=n_games, strategies=strategies, seed=seed
    )


@app.command()
def show(
    n:       Annotated[int, typer.Option("--n", help="표시할 최근 회차 수")] = 5,
    results: ResultsCsvOpt = _DEFAULT_RESULTS,
    games:   GamesCsvOpt   = _DEFAULT_GAMES,
) -> None:
    """📋 저장된 최신 당첨번호를 출력한다."""
    _make_agent(results, games).show_latest(n=n)


@app.command()
def run(
    n_games:      Annotated[int, typer.Option("--n-games", "-n")] = 5,
    results:      ResultsCsvOpt = _DEFAULT_RESULTS,
    games:        GamesCsvOpt   = _DEFAULT_GAMES,
    seed:         Annotated[Optional[int], typer.Option("--seed")] = None,
    skip_collect: Annotated[
        bool, typer.Option("--skip-collect", help="수집 건너뜀 (오프라인 모드)")
    ] = False,
    verbose: VerboseOpt = False,
) -> None:
    """🚀 수집 → 생성을 한 번에 실행하는 올인원 커맨드."""
    _setup_logging(verbose)
    agent = _make_agent(results, games)
    if not skip_collect:
        try:
            agent.collect()
        except Exception as exc:
            console.print(f"[yellow]⚠ 수집 실패 (로컬 데이터 사용): {exc}[/yellow]")
    agent.generate(n_games=n_games, strategies=["random", "balanced"], seed=seed)


@app.command()
def analyze(
    top_n:   Annotated[int, typer.Option("--top-n", help="빈출·미출현 상위 N개 표시")] = 10,
    no_save: Annotated[bool, typer.Option("--no-save", help="CSV 저장 건너뜀")] = False,
    results: ResultsCsvOpt = _DEFAULT_RESULTS,
    games:   GamesCsvOpt   = _DEFAULT_GAMES,
    stats:   Annotated[Path, typer.Option("--stats", help="통계 CSV 저장 경로")] = _DEFAULT_STATS,
    verbose: VerboseOpt = False,
) -> None:
    """📊 번호별 출현빈도·gap·홀짝·합계·구간 통계를 분석한다."""
    _setup_logging(verbose)
    _make_agent(results, games, stats=stats).analyze(top_n=top_n, save=not no_save)


@app.command(name="build-features")
def build_features(
    min_history: Annotated[int, typer.Option("--min-history", help="최소 과거 회차 수")] = 20,
    results:     ResultsCsvOpt = _DEFAULT_RESULTS,
    games:       GamesCsvOpt   = _DEFAULT_GAMES,
    features:    FeaturesOpt   = _DEFAULT_FEATURES,
    verbose:     VerboseOpt    = False,
) -> None:
    """🔧 ML Feature 행렬을 생성하고 parquet으로 저장한다 (leakage-free)."""
    _setup_logging(verbose)
    _make_agent(results, games, features=features).build_features(
        min_history_rounds=min_history
    )


@app.command(name="train-model")
def train_model(
    val_ratio:   Annotated[float, typer.Option("--val-ratio", help="검증셋 비율 (시간순)")] = 0.15,
    c_param:     Annotated[float, typer.Option("--C", help="L2 역규제 강도")] = 0.1,
    results:     ResultsCsvOpt = _DEFAULT_RESULTS,
    games:       GamesCsvOpt   = _DEFAULT_GAMES,
    features:    FeaturesOpt   = _DEFAULT_FEATURES,
    model:       ModelOpt      = _DEFAULT_MODEL,
    verbose:     VerboseOpt    = False,
) -> None:
    """🤖 LogisticRegression 모델을 시간순 split으로 학습한다."""
    _setup_logging(verbose)
    _make_agent(results, games, features=features, model=model).train_model(
        val_ratio=val_ratio, C=c_param
    )


@app.command()
def backtest(
    strategy: Annotated[
        Optional[list[str]],
        typer.Option("--strategy", "-s", help="전략명 (random/balanced/gap_based/model_score/ensemble, 반복 가능)"),
    ] = None,
    start:   Annotated[Optional[int], typer.Option("--start-round", help="시작 회차")] = None,
    end:     Annotated[Optional[int], typer.Option("--end-round",   help="종료 회차")] = None,
    recent:  Annotated[Optional[int], typer.Option("--recent",      help="최근 N회차 대상")] = None,
    n_games: Annotated[int, typer.Option("--n-games", "-n", help="회차당 생성 게임 수")] = 5,
    seed:    Annotated[Optional[int], typer.Option("--seed")] = None,
    no_save: Annotated[bool, typer.Option("--no-save", help="CSV 저장 건너뜀")] = False,
    results: ResultsCsvOpt = _DEFAULT_RESULTS,
    games:   GamesCsvOpt   = _DEFAULT_GAMES,
    model:   ModelOpt      = _DEFAULT_MODEL,
    backtest_out: Annotated[
        Path, typer.Option("--output", "-o", help="백테스트 결과 CSV")
    ] = _DEFAULT_BACKTEST,
    verbose: VerboseOpt = False,
) -> None:
    """⏮ 전략별 과거 성능을 검증한다.

    회차 t에서 t-1까지의 데이터만 전략에 제공하여 미래 누수를 차단한다.

    예시:

        python main.py backtest --strategy random
        python main.py backtest --strategy model_score --recent 100
        python main.py backtest -s random -s balanced --start-round 1100
    """
    _setup_logging(verbose)
    strategies = strategy or ["random"]
    _make_agent(results, games, model=model, backtest_csv=backtest_out).backtest(
        strategy_names=strategies,
        start_round=start,
        end_round=end,
        recent=recent,
        n_games=n_games,
        seed=seed,
        save=not no_save,
    )
