"""번호별 통계 분석 모듈."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# 구간 정의 (로또 번호 1~45, 분석 스펙 기준)
BAND_DEFS: list[tuple[int, int, str]] = [
    (1, 10, "1~10"),
    (11, 20, "11~20"),
    (21, 30, "21~30"),
    (31, 40, "31~40"),
    (41, 45, "41~45"),
]

NUM_COLS = [f"num{i}" for i in range(1, 7)]


def _band_of(number: int) -> str:
    for lo, hi, label in BAND_DEFS:
        if lo <= number <= hi:
            return label
    return "?"


class StatsAnalyzer:
    """당첨번호 DataFrame 기반 번호별 통계를 분석한다."""

    def __init__(self, history: pd.DataFrame):
        self._history = history.sort_values("round_no").reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────

    def compute_number_stats(self) -> pd.DataFrame:
        """번호별(1~45) 통계 DataFrame을 반환한다.

        컬럼: number, band, is_odd, total_count, total_freq_pct,
              recent_10, recent_30, recent_50, recent_100,
              last_round, current_gap
        """
        df = self._history
        n_rounds = len(df)
        latest_round = int(df["round_no"].max()) if n_rounds > 0 else 0

        records = []
        for number in range(1, 46):
            mask = (df[NUM_COLS] == number).any(axis=1)
            count = int(mask.sum())

            def cnt_recent(k: int) -> int:
                return int((df.tail(k)[NUM_COLS] == number).any(axis=1).sum())

            if count > 0:
                last_idx = int(np.where(mask.values)[0][-1])
                last_round = int(df.iloc[last_idx]["round_no"])
                current_gap = latest_round - last_round
            else:
                last_round = 0
                current_gap = latest_round

            records.append({
                "number": number,
                "band": _band_of(number),
                "is_odd": number % 2 == 1,
                "total_count": count,
                "total_freq_pct": round(count / n_rounds * 100, 2) if n_rounds > 0 else 0.0,
                "recent_10": cnt_recent(10),
                "recent_30": cnt_recent(30),
                "recent_50": cnt_recent(50),
                "recent_100": cnt_recent(100),
                "last_round": last_round,
                "current_gap": current_gap,
            })

        return pd.DataFrame(records)

    def compute_draw_stats(self) -> dict:
        """회차별 집계 통계(합계·홀짝·구간)를 반환한다."""
        df = self._history
        sums = df[NUM_COLS].sum(axis=1)
        odd_counts = (df[NUM_COLS] % 2 == 1).sum(axis=1)
        band_means: dict[str, float] = {}
        for lo, hi, label in BAND_DEFS:
            band_means[label] = float(
                ((df[NUM_COLS] >= lo) & (df[NUM_COLS] <= hi)).sum(axis=1).mean()
            )
        return {
            "sum_mean": float(sums.mean()),
            "sum_std": float(sums.std()),
            "sum_min": int(sums.min()),
            "sum_max": int(sums.max()),
            "odd_mean": float(odd_counts.mean()),
            "odd_distribution": odd_counts.value_counts().sort_index().to_dict(),
            "band_means": band_means,
        }

    def print_summary(self, top_n: int = 10) -> None:
        """Rich 테이블로 번호별 통계와 집계 요약을 출력한다."""
        stats_df = self.compute_number_stats()
        draw_stats = self.compute_draw_stats()
        n_rounds = len(self._history)

        console.rule("[bold cyan]📊 번호별 통계 분석[/bold cyan]")
        console.print(f"[dim]분석 대상: {n_rounds:,}회차[/dim]\n")

        # ── 번호별 전체 통계 테이블 ─────────────────────────────────────
        table = Table(
            title="번호별 출현 통계",
            box=box.ROUNDED,
            header_style="bold green",
            show_lines=False,
        )
        table.add_column("번호", justify="right", style="bold cyan", width=5)
        table.add_column("구간", justify="center", width=7)
        table.add_column("홀짝", justify="center", width=5)
        table.add_column("전체횟수", justify="right", width=8)
        table.add_column("전체%", justify="right", width=6)
        table.add_column("최근10", justify="right", width=6)
        table.add_column("최근30", justify="right", width=6)
        table.add_column("최근100", justify="right", width=7)
        table.add_column("마지막회차", justify="right", width=9)
        table.add_column("미출현간격", justify="right", width=9)

        for _, row in stats_df.iterrows():
            gap = int(row["current_gap"])
            gap_color = "green" if gap < 10 else "yellow" if gap < 20 else "red"
            table.add_row(
                str(int(row["number"])),
                str(row["band"]),
                "홀" if row["is_odd"] else "짝",
                str(int(row["total_count"])),
                f"{row['total_freq_pct']:.1f}%",
                str(int(row["recent_10"])),
                str(int(row["recent_30"])),
                str(int(row["recent_100"])),
                str(int(row["last_round"])),
                f"[{gap_color}]{gap}[/{gap_color}]",
            )

        console.print(table)
        console.print()

        # ── 회차 요약 통계 ──────────────────────────────────────────────
        odd_dist_str = "  ".join(
            f"{k}개:{v}" for k, v in sorted(draw_stats["odd_distribution"].items())
        )
        band_str = "  ".join(
            f"{k}:{v:.2f}" for k, v in draw_stats["band_means"].items()
        )
        console.print(
            Panel(
                f"[green]합계[/green]  평균: [bold]{draw_stats['sum_mean']:.1f}[/bold]  "
                f"±{draw_stats['sum_std']:.1f}  "
                f"범위: {draw_stats['sum_min']} ~ {draw_stats['sum_max']}\n"
                f"[green]홀수[/green]  평균: [bold]{draw_stats['odd_mean']:.2f}개[/bold]  "
                f"분포: {odd_dist_str}\n"
                f"[green]구간[/green]  평균 개수: {band_str}",
                title="[bold]회차 집계 통계[/bold]",
                border_style="blue",
                padding=(0, 2),
            )
        )

        # ── 빈출·미출현 순위 ────────────────────────────────────────────
        top_freq = stats_df.nlargest(top_n, "total_count")
        bot_freq = stats_df.nsmallest(top_n, "total_count")
        top_gap = stats_df.nlargest(top_n, "current_gap")

        def _num_list_count(sub: pd.DataFrame) -> str:
            return "  ".join(
                f"[bold]{int(r['number']):2d}[/bold]({int(r['total_count'])})"
                for _, r in sub.iterrows()
            )

        def _num_list_gap(sub: pd.DataFrame) -> str:
            return "  ".join(
                f"[bold]{int(r['number']):2d}[/bold](+{int(r['current_gap'])})"
                for _, r in sub.iterrows()
            )

        console.print(
            Panel(
                f"[green]빈출 상위 {top_n}[/green]: {_num_list_count(top_freq)}\n"
                f"[yellow]빈출 하위 {top_n}[/yellow]: {_num_list_count(bot_freq)}\n"
                f"[red]미출현 상위 {top_n}[/red]: {_num_list_gap(top_gap)}",
                title="[bold]번호 순위[/bold]",
                border_style="green",
                padding=(0, 2),
            )
        )

    def save(self, path: Path) -> None:
        """번호별 통계 DataFrame을 CSV로 저장한다."""
        stats_df = self.compute_number_stats()
        path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("번호 통계 저장: %s (%d행)", path, len(stats_df))
        console.print(f"[green]✓ 번호 통계 저장: {path}[/green]")
