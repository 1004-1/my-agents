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

logger = logging.getLogger(__name__)

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

_DEFAULT_RESULTS           = Path(os.getenv("LOTTO_RESULTS_CSV",   "data/lotto_draw_results.csv"))
_DEFAULT_GAMES             = Path(os.getenv("GENERATED_GAMES_CSV", "data/generated_games.csv"))
_DEFAULT_FEATURES          = Path("data/features.parquet")
_DEFAULT_MODEL             = Path("data/models/lr_model.pkl")
_DEFAULT_BACKTEST          = Path("data/backtest_results.csv")
_DEFAULT_STATS             = Path("data/number_stats.csv")
_DEFAULT_MULTISEED_RESULTS = Path("data/backtest_multiseed_results.csv")
_DEFAULT_MULTISEED_SUMMARY = Path("data/backtest_multiseed_summary.csv")
_DEFAULT_PREDICTION        = Path("data/prediction_results.csv")


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
        typer.Option("--strategy", "-s", help="전략 (random/balanced/balanced_v2/gap_based/model_score/ensemble, 반복 가능)"),
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
    random_games: Annotated[
        int, typer.Option("--random-games", help="random 전략 게임 수")
    ] = 5,
    balanced_games: Annotated[
        int, typer.Option("--balanced-games", help="balanced 전략 게임 수")
    ] = 5,
    total_games: Annotated[
        Optional[int],
        typer.Option("--total-games", help="총 게임 수 (random+balanced 균등 분배, 개별 옵션보다 우선)"),
    ] = None,
    balanced_v2_games: Annotated[
        int, typer.Option("--balanced-v2-games", help="balanced_v2 전략 게임 수 (기본 0)")
    ] = 0,
    model_score_games: Annotated[
        int, typer.Option("--model-score-games", help="[실험적] model_score 전략 게임 수 (기본 0)")
    ] = 0,
    ensemble_games: Annotated[
        int, typer.Option("--ensemble-games", help="[실험적] ensemble 전략 게임 수 — 항상 5게임 고정 (기본 0)")
    ] = 0,
    results:      ResultsCsvOpt = _DEFAULT_RESULTS,
    games:        GamesCsvOpt   = _DEFAULT_GAMES,
    model:        ModelOpt      = _DEFAULT_MODEL,
    seed:         Annotated[Optional[int], typer.Option("--seed")] = None,
    skip_collect: Annotated[
        bool, typer.Option("--skip-collect", help="수집 건너뜀 (오프라인 모드)")
    ] = False,
    verbose: VerboseOpt = False,
) -> None:
    """🚀 수집 → 생성을 한 번에 실행하는 주간 올인원 커맨드.

    기본: random 5게임 + balanced 5게임 = 총 10게임.
    전략 간 Jaccard ≥ 0.7인 게임(4개↑ 공유)을 자동으로 재생성해 중복을 방지한다.

    예시:

        python main.py run
        python main.py run --random-games 7 --balanced-games 3
        python main.py run --total-games 10
        python main.py run --balanced-v2-games 5 --random-games 0 --balanced-games 5
        python main.py run --random-games 4 --balanced-games 4 --model-score-games 2
        python main.py run --skip-collect
    """
    _setup_logging(verbose)
    agent = _make_agent(results, games, model=model)

    if not skip_collect:
        try:
            agent.collect()
        except Exception as exc:
            console.print(f"[yellow]⚠ 수집 실패 (로컬 데이터 사용): {exc}[/yellow]")

    # --total-games: random+balanced 균등 분배 (개별 옵션 무시)
    if total_games is not None:
        balanced_games = total_games // 2
        random_games   = total_games - balanced_games

    # 전략별 게임 수 dict (0 이하는 제외)
    strategy_games: dict[str, int] = {}
    if random_games      > 0: strategy_games["random"]       = random_games
    if balanced_games    > 0: strategy_games["balanced"]      = balanced_games
    if balanced_v2_games > 0: strategy_games["balanced_v2"]   = balanced_v2_games
    if model_score_games > 0: strategy_games["model_score"]   = model_score_games
    if ensemble_games    > 0: strategy_games["ensemble"]      = ensemble_games

    if not strategy_games:
        console.print(
            "[red]생성할 게임이 없습니다. "
            "--random-games 또는 --balanced-games를 1 이상으로 설정하세요.[/red]"
        )
        raise typer.Exit(1)

    agent.generate(
        strategy_games=strategy_games,
        seed=seed,
        cross_dedup=True,
    )


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
        typer.Option("--strategy", "-s", help="전략명 (random/balanced/balanced_v2/gap_based/model_score/ensemble, 반복 가능)"),
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
        python main.py backtest -s random -s balanced -s balanced_v2 --recent 300
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


@app.command(name="buy-lotto")
def buy_lotto(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="브라우저를 열지 않고 입력할 번호만 미리 출력"),
    ] = False,
    max_games: Annotated[
        int,
        typer.Option("--max-games", help="최대 구매 게임 수 (1~5, 기본 5)"),
    ] = 5,
    games:   GamesCsvOpt   = _DEFAULT_GAMES,
    verbose: VerboseOpt    = False,
) -> None:
    """🛒 생성된 번호를 동행복권 사이트에 자동 입력한다 (구매 버튼은 클릭 안 함).

    generated_games.csv의 최신 번호를 읽어 Chromium 브라우저를 열고,
    사용자가 로그인하면 번호를 자동으로 입력한 뒤 구매 버튼 직전에서 대기한다.

    세션은 data/browser-profile 에 저장되므로 이후 실행 시 재로그인이 불필요하다.
    스크린샷은 screenshots/ 폴더에 자동 저장된다.

    예시:

        python main.py buy-lotto
        python main.py buy-lotto --dry-run
        python main.py buy-lotto --max-games 3
    """
    import asyncio
    from rich import box as rich_box
    from rich.table import Table

    _setup_logging(verbose)

    from .automation.playwright_buyer import LottoBuyer

    headless = os.getenv("HEADLESS", "false").lower() in ("true", "1", "yes")
    buyer = LottoBuyer(
        games_csv=games,
        max_games=max_games,
        headless=headless,
    )

    # ── 최신 게임 로드 ────────────────────────────────────────────────────
    try:
        loaded_games = buyer.load_latest_games()
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]✗ {exc}[/red]")
        raise typer.Exit(1)

    # ── 번호 미리 출력 ────────────────────────────────────────────────────
    console.rule("[bold cyan]🛒 구매 번호 확인[/bold cyan]")
    tbl = Table(
        show_header=True,
        header_style="bold blue",
        box=rich_box.SIMPLE_HEAD,
        padding=(0, 1),
    )
    tbl.add_column("게임", style="dim", width=5)
    for i in range(1, 7):
        tbl.add_column(f"번호{i}", justify="right", min_width=5)
    for idx, nums in enumerate(loaded_games, 1):
        tbl.add_row(str(idx), *[str(n) for n in nums])
    console.print(tbl)
    console.print(f"  [dim]총 {len(loaded_games)}게임  |  소스: {games}[/dim]\n")

    if dry_run:
        console.print("[yellow]--dry-run 모드: 브라우저를 열지 않습니다.[/yellow]")
        return

    # ── 브라우저 자동화 실행 ──────────────────────────────────────────────
    async def _run() -> None:
        try:
            screenshot, purchased_count = await buyer.run(loaded_games)
            total = len(loaded_games)
            if purchased_count == total:
                console.print(
                    f"\n[bold green]✓ 구매 완료!  {purchased_count}/{total}게임[/bold green]\n"
                    f"  스크린샷: [dim]{screenshot}[/dim]\n"
                )
            elif purchased_count > 0:
                console.print(
                    f"\n[bold yellow]⚠ 부분 구매: {purchased_count}/{total}게임 완료[/bold yellow]\n"
                    f"  스크린샷: [dim]{screenshot}[/dim]\n"
                )
            else:
                console.print(
                    f"\n[bold red]✗ 구매 실패 — 브라우저에서 직접 확인하세요.[/bold red]\n"
                    f"  스크린샷: [dim]{screenshot}[/dim]\n"
                )
        except TimeoutError as exc:
            console.print(f"[red]✗ {exc}[/red]")
        except Exception as exc:
            console.print(f"[red]✗ 자동화 오류: {exc}[/red]")
            logger.exception("buy-lotto 오류")

    asyncio.run(_run())


@app.command(name="backtest-multiseed")
def backtest_multiseed(
    strategy: Annotated[
        Optional[list[str]],
        typer.Option(
            "--strategy", "-s",
            help="전략명 (random/balanced/balanced_v2/gap_based/model_score/ensemble, 반복 가능)",
        ),
    ] = None,
    seeds: Annotated[
        int,
        typer.Option("--seeds", help="반복 실행할 시드 수 (seed 1~N)"),
    ] = 10,
    recent: Annotated[Optional[int], typer.Option("--recent",      help="최근 N회차 대상")] = None,
    start:  Annotated[Optional[int], typer.Option("--start-round", help="시작 회차")] = None,
    end:    Annotated[Optional[int], typer.Option("--end-round",   help="종료 회차")] = None,
    n_games: Annotated[int, typer.Option("--n-games", "-n", help="회차당 생성 게임 수")] = 5,
    no_save: Annotated[bool, typer.Option("--no-save", help="CSV 저장 건너뜀")] = False,
    results: ResultsCsvOpt = _DEFAULT_RESULTS,
    games:   GamesCsvOpt   = _DEFAULT_GAMES,
    model:   ModelOpt      = _DEFAULT_MODEL,
    verbose: VerboseOpt    = False,
) -> None:
    """📊 전략을 여러 시드로 반복 백테스트해 통계적 안정성을 평가한다.

    seed 변동에 따른 성능 분포(평균±std)와 random baseline 대비 개선율을 출력한다.

    예시:

        python main.py backtest-multiseed -s random -s balanced --seeds 20 --recent 300
        python main.py backtest-multiseed -s random -s balanced -s model_score -s ensemble --seeds 10 --recent 300
        python main.py backtest-multiseed -s random -s balanced_v2 --seeds 30 --recent 200
    """
    _setup_logging(verbose)
    strategies = strategy or ["random", "balanced"]
    _make_agent(results, games, model=model).backtest_multiseed(
        strategy_names=strategies,
        n_seeds=seeds,
        start_round=start,
        end_round=end,
        recent=recent,
        n_games=n_games,
        save=not no_save,
    )


# ══════════════════════════════════════════════════════════════════════════
# check-results  /  strategy-report  — 피드백 루프
# ══════════════════════════════════════════════════════════════════════════

PredictionCsvOpt = Annotated[
    Path,
    typer.Option("--prediction", help="적중 결과 CSV 경로"),
]


@app.command(name="check-results")
def check_results(
    results:    ResultsCsvOpt    = _DEFAULT_RESULTS,
    games:      GamesCsvOpt      = _DEFAULT_GAMES,
    prediction: PredictionCsvOpt = _DEFAULT_PREDICTION,
    verbose:    VerboseOpt       = False,
) -> None:
    """🔍 생성된 번호와 실제 당첨번호를 비교해 적중 결과를 기록한다.

    생성 날짜 이후 최초 추첨 회차와 비교하며, 아직 추첨이 없는 게임은 건너뜁니다.

    결과는 [b]data/prediction_results.csv[/b]에 누적 저장됩니다.

    예시:

        python main.py check-results
        python main.py check-results --games data/generated_games.csv
    """
    from rich.table import Table

    _setup_logging(verbose)

    from .analysis.result_checker import ResultChecker

    checker = ResultChecker(
        games_path=games,
        results_path=results,
        prediction_path=prediction,
    )

    console.print("\n[bold cyan]═══ 적중 결과 확인 ═══[/bold cyan]")
    df, added, skipped = checker.check_all()

    if added == 0 and skipped == 0 and df.empty:
        console.print("  [yellow]생성된 게임 데이터가 없습니다.[/yellow]")
        return

    console.print(
        f"  신규 체크: [green]{added}[/green]건  "
        f"미추첨 스킵: [yellow]{skipped}[/yellow]건  "
        f"누적 총계: [cyan]{len(df)}[/cyan]건"
    )

    if df.empty:
        return

    import pandas as _pd
    df["match_count"] = _pd.to_numeric(df["match_count"], errors="coerce").fillna(0).astype(int)

    # 최근 20건 테이블 출력
    recent = df.tail(20).copy()
    table = Table(title="최근 결과 (최대 20건)", show_lines=False)
    table.add_column("생성일",    style="dim",     width=12)
    table.add_column("회차",      justify="right", width=6)
    table.add_column("전략",      width=16)
    table.add_column("내번호",    width=20)
    table.add_column("당첨번호",  width=20)
    table.add_column("적중",      justify="center", width=4)
    table.add_column("등수",      justify="center", width=5)

    for _, r in recent.iterrows():
        rank_val = str(r["rank"]) if str(r["rank"]) not in ("", "nan") else "-"
        rank_style = {
            "1": "bold magenta", "2": "bold red", "3": "bold yellow",
            "4": "yellow", "5": "green",
        }.get(rank_val, "dim")
        mc = int(r["match_count"])
        mc_style = "green" if mc >= 3 else "dim"
        table.add_row(
            str(r["generated_at"])[:10],
            str(r["target_round_no"]),
            str(r["strategy_name"]),
            str(r["numbers"]),
            str(r["winning_numbers"]),
            f"[{mc_style}]{mc}[/{mc_style}]",
            f"[{rank_style}]{rank_val}[/{rank_style}]",
        )

    console.print(table)
    console.print(f"\n  [dim]저장 위치: {prediction}[/dim]\n")


@app.command(name="strategy-report")
def strategy_report(
    prediction: PredictionCsvOpt = _DEFAULT_PREDICTION,
    verbose:    VerboseOpt       = False,
) -> None:
    """📈 전략별 실제 적중 성과를 통계로 요약한다.

    [b]check-results[/b] 실행 후 쌓인 [b]data/prediction_results.csv[/b]를 기반으로
    전략별 평균 적중 수, 등수 분포 등을 표로 출력합니다.

    예시:

        python main.py strategy-report
        python main.py strategy-report --prediction data/prediction_results.csv
    """
    from rich.table import Table

    _setup_logging(verbose)

    from .analysis.result_checker import ResultChecker

    checker = ResultChecker(prediction_path=prediction)

    console.print("\n[bold cyan]═══ 전략별 성과 리포트 ═══[/bold cyan]")
    summary = checker.strategy_summary()

    if summary.empty:
        console.print(
            f"  [yellow]데이터가 없습니다. 먼저 [bold]check-results[/bold]를 실행하세요.[/yellow]\n"
        )
        return

    table = Table(title="전략별 실적 (prediction_results.csv 기준)", show_lines=True)
    table.add_column("전략",         width=18)
    table.add_column("총게임",        justify="right", width=7)
    table.add_column("평균적중",      justify="right", width=8)
    table.add_column("3+",           justify="right", width=5)
    table.add_column("4+",           justify="right", width=5)
    table.add_column("5+",           justify="right", width=5)
    table.add_column("최대",          justify="right", width=5)
    table.add_column("5등",           justify="right", width=5)
    table.add_column("4등",           justify="right", width=5)
    table.add_column("3등",           justify="right", width=5)
    table.add_column("2등",           justify="right", width=5)
    table.add_column("1등",           justify="right", width=5)

    for _, r in summary.iterrows():
        avg_str = f"{r['avg_match']:.3f}"
        table.add_row(
            str(r["strategy"]),
            str(r["total_games"]),
            avg_str,
            str(r["match_3plus"]),
            str(r["match_4plus"]),
            str(r["match_5plus"]),
            str(r["max_match"]),
            str(r["rank_5"]),
            str(r["rank_4"]),
            str(r["rank_3"]),
            str(r["rank_2"]),
            str(r["rank_1"]),
        )

    console.print(table)
    console.print()
