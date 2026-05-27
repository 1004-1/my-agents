"""LogisticRegression 모델 학습 모듈.

로또는 독립 확률 추첨이므로 이 모듈은 당첨 번호를 '예측'하지 않는다.
학습 기반 번호 선택 전략의 score 산출에 사용할 통계적 패턴 모델을 구축한다.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

try:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .feature_builder import FEATURE_COLS, load_features


@dataclass
class TrainResult:
    """모델 학습 결과 요약."""
    train_rounds: int
    val_rounds: int
    train_auc: float
    val_auc: float
    train_brier: float
    val_brier: float
    top6_hit_avg: float    # 검증 회차별 상위 6개 중 실제 당첨 평균 수
    top10_hit_avg: float   # 검증 회차별 상위 10개 중 실제 당첨 평균 수
    feature_coefs: dict[str, float]
    model_path: Path


def train_model(
    features_df: pd.DataFrame | None = None,
    features_path: Path | None = None,
    model_path: Path = Path("data/models/lr_model.pkl"),
    val_ratio: float = 0.15,
    C: float = 0.1,
    max_iter: int = 1000,
    random_state: int = 42,
) -> TrainResult:
    """LogisticRegression 모델을 시간순 split으로 학습한다.

    핵심 보장:
    - shuffle=False: 시간 순서 유지
    - round_no 기준 split (번호 단위가 아닌 회차 단위)
    - StandardScaler → Pipeline으로 묶어 test-set leakage 방지
    - class_weight="balanced": 양성(6개)/음성(39개) 불균형 보정

    Args:
        features_df:    build_features()로 만든 DataFrame (None이면 features_path 로드)
        features_path:  parquet 경로 (features_df가 None일 때 사용)
        model_path:     학습된 모델 저장 경로 (.pkl)
        val_ratio:      검증셋 비율 (시간순으로 뒤에서부터)
        C:              LogisticRegression L2 역규제 강도 (작을수록 강한 규제)
        max_iter:       최대 반복 횟수
        random_state:   재현성용 랜덤 시드

    Returns:
        TrainResult (성능 지표 + 저장 경로)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn이 필요합니다: pip install scikit-learn joblib"
        )

    console.rule("[bold cyan]🤖 모델 학습 시작[/bold cyan]")
    t0 = time.monotonic()

    # 데이터 로드
    if features_df is None:
        if features_path is None:
            raise ValueError("features_df 또는 features_path 중 하나를 제공해야 합니다.")
        features_df = load_features(Path(features_path))

    console.print(
        f"[cyan]Feature: {len(features_df):,}행 × {len(FEATURE_COLS)}개 feature[/cyan]"
    )

    # 시간순 split (round_no 기준 — shuffle 없음)
    all_rounds = sorted(features_df["target_round_no"].unique())
    n_val = max(1, int(len(all_rounds) * val_ratio))
    train_round_list = all_rounds[:-n_val]
    val_round_list   = all_rounds[-n_val:]

    train_df = features_df[features_df["target_round_no"].isin(train_round_list)]
    val_df   = features_df[features_df["target_round_no"].isin(val_round_list)]

    console.print(
        f"  학습: [bold]{len(train_round_list):,}[/bold]회차 ({len(train_df):,}행)  |  "
        f"검증: [bold]{len(val_round_list):,}[/bold]회차 ({len(val_df):,}행)"
    )

    X_train = np.nan_to_num(train_df[FEATURE_COLS].values.astype(float), nan=0.0)
    y_train = train_df["appeared_in_target_round"].values.astype(int)
    X_val   = np.nan_to_num(val_df[FEATURE_COLS].values.astype(float), nan=0.0)
    y_val   = val_df["appeared_in_target_round"].values.astype(int)

    # Pipeline: StandardScaler + LogisticRegression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    with console.status("[bold yellow]LogisticRegression 학습 중...[/bold yellow]", spinner="dots"):
        pipeline.fit(X_train, y_train)

    # 평가 지표
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    val_proba   = pipeline.predict_proba(X_val)[:, 1]
    train_auc   = float(roc_auc_score(y_train, train_proba))
    val_auc     = float(roc_auc_score(y_val,   val_proba))
    train_brier = float(brier_score_loss(y_train, train_proba))
    val_brier   = float(brier_score_loss(y_val,   val_proba))

    # top-k hit 평균
    top6_hits, top10_hits = [], []
    for rn in val_round_list:
        rn_df = val_df[val_df["target_round_no"] == rn].copy()
        rn_X  = np.nan_to_num(rn_df[FEATURE_COLS].values.astype(float), nan=0.0)
        rn_df = rn_df.assign(score=pipeline.predict_proba(rn_X)[:, 1])
        top6_hits.append(int(rn_df.nlargest(6,  "score")["appeared_in_target_round"].sum()))
        top10_hits.append(int(rn_df.nlargest(10, "score")["appeared_in_target_round"].sum()))

    top6_avg  = float(np.mean(top6_hits))
    top10_avg = float(np.mean(top10_hits))

    # 모델 저장
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    # 메타데이터
    coefs: dict[str, float] = dict(
        zip(FEATURE_COLS, pipeline["clf"].coef_[0].tolist())
    )
    meta = {
        "train_rounds": len(train_round_list),
        "val_rounds":   len(val_round_list),
        "train_auc":    train_auc,
        "val_auc":      val_auc,
        "train_brier":  train_brier,
        "val_brier":    val_brier,
        "top6_hit_avg": top6_avg,
        "top10_hit_avg": top10_avg,
        "C":             C,
        "feature_cols": FEATURE_COLS,
        "model_path":   str(model_path),
    }
    meta_path = model_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.monotonic() - t0
    result = TrainResult(
        train_rounds=len(train_round_list),
        val_rounds=len(val_round_list),
        train_auc=train_auc,
        val_auc=val_auc,
        train_brier=train_brier,
        val_brier=val_brier,
        top6_hit_avg=top6_avg,
        top10_hit_avg=top10_avg,
        feature_coefs=coefs,
        model_path=model_path,
    )
    _print_train_result(result, elapsed)
    return result


def _print_train_result(result: TrainResult, elapsed: float) -> None:
    """학습 결과를 Rich 패널로 출력한다."""
    table = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 2))
    table.add_column("항목", style="dim", min_width=18)
    table.add_column("값", style="bold")

    auc_color = "green" if result.val_auc >= 0.55 else "yellow"
    table.add_row("학습 회차", f"{result.train_rounds:,}회차")
    table.add_row("검증 회차", f"{result.val_rounds:,}회차")
    table.add_row("Train AUC",  f"[cyan]{result.train_auc:.4f}[/cyan]")
    table.add_row("Val AUC",    f"[{auc_color}]{result.val_auc:.4f}[/{auc_color}]")
    table.add_row("Train Brier", f"{result.train_brier:.4f}")
    table.add_row("Val Brier",   f"{result.val_brier:.4f}")
    table.add_row("Top-6 Hit 평균",  f"[yellow]{result.top6_hit_avg:.3f}[/yellow] / 6개")
    table.add_row("Top-10 Hit 평균", f"[yellow]{result.top10_hit_avg:.3f}[/yellow] / 6개")
    table.add_row("소요 시간", f"{elapsed:.1f}초")
    table.add_row("모델 저장", str(result.model_path))

    console.print(
        Panel(
            table,
            title="[bold green]✓ 모델 학습 완료[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    top5 = sorted(result.feature_coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    coef_str = "  ".join(f"[cyan]{k}[/cyan]({v:+.3f})" for k, v in top5)
    console.print(f"[dim]주요 Feature 계수:[/dim] {coef_str}")


def load_model(model_path: Path | str) -> Any:
    """저장된 Pipeline 모델을 로드한다."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn이 필요합니다.")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"모델 파일 없음: {model_path}\n`train-model` 명령을 먼저 실행하세요."
        )
    return joblib.load(model_path)
