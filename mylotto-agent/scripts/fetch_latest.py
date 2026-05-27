"""동행복권 최근 N회차 당첨번호를 실시간으로 가져와 출력한다.

lotto.co.kr 스크래핑 방식 사용 — 한국 내외 어디서나 동작.

사용법:
    python scripts/fetch_latest.py            # 최근 10회차
    python scripts/fetch_latest.py --n 20     # 최근 20회차
    python scripts/fetch_latest.py --from 1200  # 1200회차부터 최신까지
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

# ── Windows UTF-8 ─────────────────────────────────────────────────────────
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.lotto.collector.lotto_kr_collector import LottoKrCollector
from src.lotto.collector.validator import validate_draw

console = Console()

# ── 공 색상 (동행복권 공식 색상) ──────────────────────────────────────────
_BALL_STYLE: list[tuple[range, str, str]] = [
    (range(1,  10), "bold yellow",      "●"),
    (range(10, 20), "bold blue",         "●"),
    (range(20, 30), "bold red",          "●"),
    (range(30, 40), "bold bright_black", "●"),
    (range(40, 46), "bold green",        "●"),
]

def _ball_text(n: int, bonus: bool = False) -> Text:
    """번호를 색상이 입혀진 Rich Text로 반환한다."""
    style = "bold white"
    symbol = "●"
    for rng, st, sym in _BALL_STYLE:
        if n in rng:
            style, symbol = st, sym
            break
    label = f" {n:2d} "
    t = Text(label, style=style)
    if bonus:
        t = Text("+", style="dim") + t
    return t


def _build_table(rows: list[dict], title: str) -> Table:
    """당첨번호 목록을 Rich Table로 반환한다."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold green",
        show_lines=True,
        min_width=80,
    )
    table.add_column("회차",   justify="right",  style="bold cyan",  min_width=6)
    table.add_column("날짜",   justify="center", style="dim",        min_width=12)
    table.add_column("  1",    justify="center", min_width=5)
    table.add_column("  2",    justify="center", min_width=5)
    table.add_column("  3",    justify="center", min_width=5)
    table.add_column("  4",    justify="center", min_width=5)
    table.add_column("  5",    justify="center", min_width=5)
    table.add_column("  6",    justify="center", min_width=5)
    table.add_column(" 보너스", justify="center", min_width=6)

    for row in sorted(rows, key=lambda r: r["round_no"], reverse=True):
        nums  = [row[f"num{i}"] for i in range(1, 7)]
        bonus = row["bonus"]
        table.add_row(
            str(row["round_no"]),
            str(row["date"]),
            *[_ball_text(n) for n in nums],
            _ball_text(bonus, bonus=True),
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="동행복권 당첨번호를 lotto.co.kr에서 실시간으로 출력합니다."
    )
    parser.add_argument(
        "--n", type=int, default=10, metavar="N",
        help="가져올 회차 수 (기본: 10)",
    )
    parser.add_argument(
        "--from", dest="from_round", type=int, default=None, metavar="ROUND",
        help="이 회차부터 최신까지 가져옴 (--n 무시)",
    )
    args = parser.parse_args()

    console.print(Rule("[bold cyan]🎱 동행복권 당첨번호 실시간 조회[/bold cyan]"))
    console.print("[dim]데이터 출처: lotto.co.kr[/dim]\n")

    collector = LottoKrCollector()

    # ── Step 1: 최신 회차 탐색 ───────────────────────────────────────────
    with console.status(
        "[bold yellow]최신 회차 확인 중...[/bold yellow]", spinner="dots"
    ):
        latest = collector.fetch_latest_round()

    console.print(
        Panel(
            f"[bold green]최신 회차:[/bold green] [bold white]{latest}[/bold white] 회차",
            expand=False,
            border_style="green",
            padding=(0, 2),
        )
    )

    # ── Step 2: 수집 범위 결정 ───────────────────────────────────────────
    if args.from_round:
        start_rn = max(1, args.from_round)
        end_rn   = latest
    else:
        end_rn   = latest
        start_rn = max(1, end_rn - args.n + 1)

    total = end_rn - start_rn + 1
    console.print(
        f"  조회 범위: [bold cyan]{start_rn}[/bold cyan] ~ "
        f"[bold cyan]{end_rn}[/bold cyan] 회차  "
        f"(총 [bold yellow]{total}[/bold yellow]회차)\n"
    )

    # ── Step 3: 수집 ─────────────────────────────────────────────────────
    t0 = time.monotonic()
    rows: list[dict] = []
    failed: list[int] = []

    with console.status("", spinner="dots") as status:
        df, stats = collector.collect_range(
            start_rn, end_rn,
            on_progress=lambda r: (
                status.update(
                    f"[cyan]수집 중 — 회차 [bold]{r.round_no}[/bold] "
                    f"{'[green]✓[/green]' if r.is_valid else '[red]✗[/red]'}[/cyan]"
                )
            ),
        )

    # stats에서 추출
    rows = df.to_dict("records") if not df.empty else []
    failed = stats.failed_rounds
    elapsed = stats.elapsed

    console.print()

    # ── Step 4: 결과 테이블 출력 ─────────────────────────────────────────
    if rows:
        table = _build_table(
            rows,
            title=f"🎱 동행복권 당첨번호  ({start_rn}~{end_rn}회차)"
        )
        console.print(table)
    else:
        console.print("[red]수집된 데이터가 없습니다.[/red]")
        return

    # ── Step 5: 요약 ─────────────────────────────────────────────────────
    console.print()
    summary = (
        f"[green]성공[/green]: {stats.collected}회차  "
        f"[{'red' if failed else 'dim'}]실패[/{'red' if failed else 'dim'}]: {len(failed)}회차  "
        f"소요: {elapsed:.1f}초"
    )
    if failed:
        summary += f"\n실패 회차: [red]{', '.join(map(str, failed[:10]))}[/red]"
        if len(failed) > 10:
            summary += f" 외 {len(failed)-10}건"

    console.print(
        Panel(
            summary,
            title="[bold]수집 완료[/bold]",
            border_style="green" if not failed else "yellow",
            expand=False,
            padding=(0, 2),
        )
    )


if __name__ == "__main__":
    main()
