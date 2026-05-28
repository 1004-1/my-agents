"""ML Feature 생성 모듈 — leakage-free.

회차 t에서 번호 n의 feature는 t보다 앞선 회차 데이터만 사용한다.
미래 데이터 누수(data leakage)를 원천 차단하는 것이 핵심 설계 원칙이다.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
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

logger = logging.getLogger(__name__)
console = Console()

# Feature 컬럼 순서 (모델 학습·추론 시 일관성 보장)
FEATURE_COLS: list[str] = [
    "total_frequency_before",
    "recent_10_frequency",
    "recent_30_frequency",
    "recent_50_frequency",
    "recent_100_frequency",
    "gap_since_last_seen",
    "rolling_avg_gap",
    "appeared_in_previous_round",
    "number_mod_2",
    "number_decade",
]

# 출력 컬럼 전체 순서
OUTPUT_COLS: list[str] = [
    "target_round_no",
    "number",
    "appeared_in_target_round",
] + FEATURE_COLS


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

def _build_appeared_matrix(history: pd.DataFrame) -> np.ndarray:
    """appeared[i, n] = True if number n+1 appeared in round i.

    Shape: (n_rounds, 45).  완전 벡터화로 빠르게 구축한다.
    """
    n = len(history)
    appeared = np.zeros((n, 45), dtype=np.bool_)
    for col in (f"num{j}" for j in range(1, 7)):
        nums = history[col].values.astype(int) - 1  # 0-indexed (0..44)
        valid = (nums >= 0) & (nums < 45)
        appeared[np.where(valid)[0], nums[valid]] = True
    return appeared


# ── 핵심 공개 API ─────────────────────────────────────────────────────────

def build_features(
    history: pd.DataFrame,
    min_history_rounds: int = 20,
    quiet: bool = False,
) -> pd.DataFrame:
    """전체 history에서 leakage-free feature 행렬을 생성한다.

    회차 t의 feature는 index 0 ~ t-1 회차 데이터(history 기준)만 참조한다.
    min_history_rounds 미만 인덱스의 회차는 feature 계산용으로만 쓰고
    출력 DataFrame에는 포함하지 않는다.

    Args:
        history:              DRAW_COLUMNS 형식의 당첨번호 DataFrame (round_no 오름차순)
        min_history_rounds:   feature 계산에 필요한 최소 과거 회차 수
        quiet:                True이면 콘솔 출력을 억제한다 (배치 사전 계산용)

    Returns:
        DataFrame, columns = OUTPUT_COLS
        행 수 = (len(history) - min_history_rounds) × 45
    """
    if history.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    history = history.sort_values("round_no").reset_index(drop=True)
    n_rounds = len(history)
    round_nos = history["round_no"].values

    if n_rounds <= min_history_rounds:
        logger.warning(
            "history(%d회차) ≤ min_history_rounds(%d). Feature 없음.",
            n_rounds, min_history_rounds,
        )
        return pd.DataFrame(columns=OUTPUT_COLS)

    t_start = time.monotonic()
    n_target = n_rounds - min_history_rounds

    if not quiet:
        console.print(
            f"[cyan]Feature 생성: {n_rounds:,}회차 × 45번호 "
            f"(학습 시작 인덱스: {min_history_rounds})[/cyan]"
        )

    # ── 1. appeared 행렬 ──────────────────────────────────────────────
    appeared = _build_appeared_matrix(history)

    # ── 2. 누적 카운트 ────────────────────────────────────────────────
    # cum[i, n] = n이 rounds[0..i-1]에 나온 횟수
    cum = np.zeros((n_rounds + 1, 45), dtype=np.int32)
    cum[1:] = np.cumsum(appeared.astype(np.int32), axis=0)

    # ── 3. 마지막 출현 인덱스 ─────────────────────────────────────────
    # last_seen[i, n] = n이 rounds[0..i-1]에 마지막으로 나온 인덱스, -1이면 없음
    last_seen = np.full((n_rounds + 1, 45), -1, dtype=np.int32)
    for i in range(1, n_rounds + 1):
        last_seen[i] = np.where(appeared[i - 1], i - 1, last_seen[i - 1])

    # ── 4. 누적 gap 합계 · 횟수 (rolling_avg_gap) ────────────────────
    sum_gaps = np.zeros((n_rounds + 1, 45), dtype=np.int64)
    n_gaps   = np.zeros((n_rounds + 1, 45), dtype=np.int32)
    for i in range(1, n_rounds + 1):
        sum_gaps[i] = sum_gaps[i - 1]
        n_gaps[i]   = n_gaps[i - 1]
        appeared_now = appeared[i - 1]
        had_seen     = last_seen[i - 1] >= 0
        update_mask  = appeared_now & had_seen
        new_gaps     = np.where(update_mask, (i - 1) - last_seen[i - 1], 0)
        sum_gaps[i] += new_gaps
        n_gaps[i]   += update_mask.astype(np.int32)

    # ── 5. 결과 배열 사전 할당 ────────────────────────────────────────
    n_total = n_target * 45
    arr_round  = np.empty(n_total, dtype=np.int32)
    arr_num    = np.empty(n_total, dtype=np.int8)
    arr_target = np.empty(n_total, dtype=np.int8)
    arr_tf     = np.empty(n_total, dtype=np.float32)
    arr_r10    = np.empty(n_total, dtype=np.float32)
    arr_r30    = np.empty(n_total, dtype=np.float32)
    arr_r50    = np.empty(n_total, dtype=np.float32)
    arr_r100   = np.empty(n_total, dtype=np.float32)
    arr_gap    = np.empty(n_total, dtype=np.int32)
    arr_avg    = np.empty(n_total, dtype=np.float32)
    arr_prev   = np.empty(n_total, dtype=np.int8)

    numbers = np.arange(1, 46, dtype=np.int8)  # 1..45

    # ── 6. 메인 루프 (회차 단위, 번호 축은 벡터화) ───────────────────
    def _run_main_loop(progress_ctx):
        task = progress_ctx.add_task("Feature 계산 중", total=n_target) if progress_ctx else None
        for k, i in enumerate(range(min_history_rounds, n_rounds)):
            s, e = k * 45, k * 45 + 45
            arr_round[s:e]  = round_nos[i]
            arr_num[s:e]    = numbers
            arr_target[s:e] = appeared[i].astype(np.int8)
            arr_tf[s:e]     = cum[i].astype(np.float32) / float(i)
            for kk, arr_r in ((10, arr_r10), (30, arr_r30), (50, arr_r50), (100, arr_r100)):
                start_i = max(0, i - kk)
                window  = float(i - start_i)
                arr_r[s:e] = (cum[i] - cum[start_i]).astype(np.float32) / window
            ls = last_seen[i]
            arr_gap[s:e] = np.where(ls >= 0, (i - 1) - ls, i).astype(np.int32)
            ng = n_gaps[i]
            sg = sum_gaps[i]
            arr_avg[s:e] = np.where(
                ng > 0,
                (sg.astype(np.float64) / np.maximum(ng, 1)).astype(np.float32),
                np.float32(i),
            )
            arr_prev[s:e] = appeared[i - 1].astype(np.int8)
            if progress_ctx and task is not None:
                progress_ctx.advance(task)

    if quiet:
        _run_main_loop(None)
    else:
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
            _run_main_loop(progress)

    # 정적 feature (모든 회차 동일)
    arr_mod2   = np.tile(numbers % 2, n_target).astype(np.int8)
    arr_decade = np.tile((numbers // 10 + 1).astype(np.int8), n_target)

    df = pd.DataFrame({
        "target_round_no":        arr_round,
        "number":                 arr_num.astype(np.int16),
        "appeared_in_target_round": arr_target.astype(np.int8),
        "total_frequency_before": arr_tf,
        "recent_10_frequency":    arr_r10,
        "recent_30_frequency":    arr_r30,
        "recent_50_frequency":    arr_r50,
        "recent_100_frequency":   arr_r100,
        "gap_since_last_seen":    arr_gap,
        "rolling_avg_gap":        arr_avg,
        "appeared_in_previous_round": arr_prev.astype(np.int8),
        "number_mod_2":           arr_mod2,
        "number_decade":          arr_decade,
    })

    elapsed = time.monotonic() - t_start
    if not quiet:
        console.print(
            f"[green]✓ Feature 생성 완료: {len(df):,}행 "
            f"({n_target:,}회차 × 45번호)  소요: {elapsed:.1f}초[/green]"
        )
    logger.info("Feature 생성: %d행, %.1f초", len(df), elapsed)
    return df


def _compute_number_features(past_df: pd.DataFrame, number: int) -> dict[str, float]:
    """단일 번호의 feature를 과거 데이터만으로 계산한다 (온라인 추론용).

    이 함수는 모델 inference 시점(generate / backtest)에서 사용된다.
    past_df는 예측 대상 회차보다 앞선 데이터만 포함해야 한다 (leakage 방지).
    """
    n_past = len(past_df)
    num_cols = [f"num{j}" for j in range(1, 7)]

    # 빈 이력
    if n_past == 0:
        return {
            "total_frequency_before": 0.0,
            "recent_10_frequency":    0.0,
            "recent_30_frequency":    0.0,
            "recent_50_frequency":    0.0,
            "recent_100_frequency":   0.0,
            "gap_since_last_seen":    0,
            "rolling_avg_gap":        0.0,
            "appeared_in_previous_round": 0,
            "number_mod_2":   int(number % 2),
            "number_decade":  int(number // 10 + 1),
        }

    mask = (past_df[num_cols] == number).any(axis=1)
    count = int(mask.sum())
    total_freq = count / n_past

    def _recent(k: int) -> float:
        recent = past_df.tail(k)
        actual_k = len(recent)
        if actual_k == 0:
            return 0.0
        return float((recent[num_cols] == number).any(axis=1).sum() / actual_k)

    if count > 0:
        indices = np.where(mask.values)[0]
        last_idx = int(indices[-1])
        gap = (n_past - 1) - last_idx
        rolling_avg = float(np.diff(indices).mean()) if count >= 2 else float(n_past)
    else:
        gap = n_past
        rolling_avg = float(n_past)

    appeared_prev = 0
    if n_past > 0:
        last_row = past_df.iloc[-1]
        appeared_prev = int(any(int(last_row[c]) == number for c in num_cols))

    return {
        "total_frequency_before": float(total_freq),
        "recent_10_frequency":    _recent(10),
        "recent_30_frequency":    _recent(30),
        "recent_50_frequency":    _recent(50),
        "recent_100_frequency":   _recent(100),
        "gap_since_last_seen":    int(gap),
        "rolling_avg_gap":        float(rolling_avg),
        "appeared_in_previous_round": int(appeared_prev),
        "number_mod_2":   int(number % 2),
        "number_decade":  int(number // 10 + 1),
    }


# ── 저장·로드 ─────────────────────────────────────────────────────────────

def save_features(features_df: pd.DataFrame, path: Path) -> None:
    """Feature DataFrame을 parquet으로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(path, index=False)
    logger.info("Feature 저장: %s (%d행)", path, len(features_df))
    console.print(f"[green]✓ Feature 저장: {path} ({len(features_df):,}행)[/green]")


def load_features(path: Path) -> pd.DataFrame:
    """parquet에서 Feature DataFrame을 로드한다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature 파일 없음: {path}\n`build-features` 명령을 먼저 실행하세요."
        )
    df = pd.read_parquet(path)
    logger.info("Feature 로드: %s (%d행)", path, len(df))
    return df
