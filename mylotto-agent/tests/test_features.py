"""Feature 생성 단위 테스트 (leakage 방지 포함)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.lotto.ml.feature_builder import (
    OUTPUT_COLS,
    FEATURE_COLS,
    build_features,
    _compute_number_features,
    save_features,
    load_features,
)


class TestBuildFeatures:
    def test_row_count(self, larger_history):
        min_hist = 5
        features = build_features(larger_history, min_history_rounds=min_hist)
        expected = (len(larger_history) - min_hist) * 45
        assert len(features) == expected

    def test_output_columns(self, larger_history):
        features = build_features(larger_history, min_history_rounds=5)
        assert list(features.columns) == OUTPUT_COLS

    def test_target_is_binary(self, larger_history):
        features = build_features(larger_history, min_history_rounds=5)
        unique_vals = set(features["appeared_in_target_round"].unique())
        assert unique_vals.issubset({0, 1})

    def test_number_range(self, larger_history):
        features = build_features(larger_history, min_history_rounds=5)
        assert features["number"].min() == 1
        assert features["number"].max() == 45

    def test_frequency_between_0_and_1(self, larger_history):
        features = build_features(larger_history, min_history_rounds=5)
        for col in ["total_frequency_before", "recent_10_frequency",
                    "recent_30_frequency", "recent_50_frequency", "recent_100_frequency"]:
            assert features[col].between(0, 1).all(), f"{col} 범위 이상"

    def test_gap_non_negative(self, larger_history):
        features = build_features(larger_history, min_history_rounds=5)
        assert (features["gap_since_last_seen"] >= 0).all()

    def test_rolling_avg_gap_positive(self, larger_history):
        features = build_features(larger_history, min_history_rounds=5)
        assert (features["rolling_avg_gap"] >= 0).all()

    def test_empty_history_returns_empty_df(self):
        empty = pd.DataFrame(columns=["round_no", "date",
                                       "num1","num2","num3","num4","num5","num6","bonus"])
        result = build_features(empty, min_history_rounds=5)
        assert result.empty

    def test_insufficient_history_returns_empty_df(self, sample_history):
        # sample_history는 10회차 → min=20이면 빈 DataFrame
        result = build_features(sample_history, min_history_rounds=20)
        assert result.empty


class TestNoFutureLeakage:
    """핵심 leakage 방지 테스트."""

    def test_target_round_always_greater_than_min_history(self, larger_history):
        min_hist = 10
        features = build_features(larger_history, min_history_rounds=min_hist)
        round_nos = larger_history.sort_values("round_no")["round_no"].values
        # features의 최소 target_round_no는 round_nos[min_hist] 이상이어야 함
        min_allowed = int(round_nos[min_hist])
        assert int(features["target_round_no"].min()) >= min_allowed

    def test_appeared_in_target_matches_actual(self, larger_history):
        """target 레이블이 실제 당첨번호와 일치하는지 검증."""
        min_hist = 5
        features = build_features(larger_history, min_history_rounds=min_hist)
        history = larger_history.sort_values("round_no").reset_index(drop=True)
        num_cols = [f"num{j}" for j in range(1, 7)]

        # 임의 샘플 5개 회차만 검증 (전체는 오래 걸림)
        sample_rounds = features["target_round_no"].unique()[:5]
        for rn in sample_rounds:
            actual_row = history[history["round_no"] == rn].iloc[0]
            winning_set = set(int(actual_row[c]) for c in num_cols)
            feat_rows = features[features["target_round_no"] == rn]
            for _, row in feat_rows.iterrows():
                n = int(row["number"])
                expected = 1 if n in winning_set else 0
                assert int(row["appeared_in_target_round"]) == expected, (
                    f"회차 {rn}, 번호 {n}: expected={expected}, got={int(row['appeared_in_target_round'])}"
                )

    def test_feature_uses_only_past_data(self, larger_history):
        """각 target_round_no의 feature는 해당 회차보다 앞선 데이터만 사용했는지 구조적 검증."""
        min_hist = 5
        features = build_features(larger_history, min_history_rounds=min_hist)
        history = larger_history.sort_values("round_no").reset_index(drop=True)
        num_cols = [f"num{j}" for j in range(1, 7)]

        # 각 target 회차에서 total_frequency_before가 그 회차를 포함하지 않는지 확인
        # total_frequency = count_in_past / n_past
        # n_past = 해당 회차의 index (round_no 기준 앞선 회차 수)
        for target_rn in features["target_round_no"].unique()[:3]:
            past = history[history["round_no"] < target_rn]
            n_past = len(past)
            feat_rows = features[features["target_round_no"] == target_rn]

            for _, row in feat_rows.iterrows():
                n = int(row["number"])
                mask = (past[num_cols] == n).any(axis=1)
                actual_count = int(mask.sum())
                expected_freq = actual_count / n_past if n_past > 0 else 0.0
                assert abs(float(row["total_frequency_before"]) - expected_freq) < 1e-5, (
                    f"회차 {target_rn}, 번호 {n}: freq mismatch"
                )


class TestComputeNumberFeatures:
    def test_empty_past_returns_zeros(self):
        empty = pd.DataFrame(columns=["round_no","date",
                                       "num1","num2","num3","num4","num5","num6","bonus"])
        feats = _compute_number_features(empty, 7)
        assert feats["total_frequency_before"] == 0.0
        assert feats["appeared_in_previous_round"] == 0

    def test_feature_cols_complete(self, sample_history):
        feats = _compute_number_features(sample_history, 23)
        for col in FEATURE_COLS:
            assert col in feats, f"누락 feature: {col}"

    def test_number_mod_2_correct(self, sample_history):
        for n in [1, 2, 7, 44]:
            feats = _compute_number_features(sample_history, n)
            assert feats["number_mod_2"] == n % 2

    def test_number_decade_correct(self, sample_history):
        cases = [(1, 1), (9, 1), (10, 2), (19, 2), (20, 3), (30, 4), (40, 5), (45, 5)]
        for n, expected in cases:
            feats = _compute_number_features(sample_history, n)
            assert feats["number_decade"] == expected, f"번호 {n}: expected decade {expected}"

    def test_appeared_in_previous_round(self, sample_history):
        # sample_history의 마지막 회차(10)에서 num1=9 → 번호 9 appeared_in_previous_round=1
        feats = _compute_number_features(sample_history, 9)
        assert feats["appeared_in_previous_round"] == 1

    def test_gap_for_unseen_number(self, sample_history):
        # 번호 45는 sample_history의 당첨번호에 없음 (bonus=45인 회차는 있지만 num 컬럼에 없어야 함)
        feats = _compute_number_features(sample_history, 45)
        # num 컬럼에 없으면 gap = n_past
        num_cols = [f"num{j}" for j in range(1, 7)]
        mask = (sample_history[num_cols] == 45).any(axis=1)
        if not mask.any():
            assert feats["gap_since_last_seen"] == len(sample_history)


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path, larger_history):
        features = build_features(larger_history, min_history_rounds=5)
        path = tmp_path / "features.parquet"
        save_features(features, path)
        loaded = load_features(path)
        assert len(loaded) == len(features)
        assert list(loaded.columns) == list(features.columns)

    def test_load_raises_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_features(tmp_path / "nonexistent.parquet")
