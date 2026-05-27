"""StatsAnalyzer 단위 테스트."""
from __future__ import annotations

import pandas as pd
import pytest

from src.lotto.analysis.stats_analyzer import StatsAnalyzer


class TestComputeNumberStats:
    def test_returns_45_rows(self, sample_history):
        analyzer = StatsAnalyzer(sample_history)
        stats = analyzer.compute_number_stats()
        assert len(stats) == 45

    def test_total_count_sum_equals_rounds_times_6(self, sample_history):
        analyzer = StatsAnalyzer(sample_history)
        stats = analyzer.compute_number_stats()
        assert int(stats["total_count"].sum()) == len(sample_history) * 6

    def test_gap_is_non_negative(self, sample_history):
        analyzer = StatsAnalyzer(sample_history)
        stats = analyzer.compute_number_stats()
        assert (stats["current_gap"] >= 0).all()

    def test_never_appeared_number_has_zero_last_round(self, sample_history):
        # 번호 45는 sample_history에 없음 (bonus에만 있을 수 있음)
        analyzer = StatsAnalyzer(sample_history)
        stats = analyzer.compute_number_stats()
        row_45 = stats[stats["number"] == 45].iloc[0]
        if int(row_45["total_count"]) == 0:
            assert int(row_45["last_round"]) == 0

    def test_total_freq_pct_between_0_and_100(self, sample_history):
        analyzer = StatsAnalyzer(sample_history)
        stats = analyzer.compute_number_stats()
        assert (stats["total_freq_pct"] >= 0).all()
        assert (stats["total_freq_pct"] <= 100).all()

    def test_band_column_present(self, sample_history):
        analyzer = StatsAnalyzer(sample_history)
        stats = analyzer.compute_number_stats()
        assert "band" in stats.columns

    def test_is_odd_correct(self, sample_history):
        analyzer = StatsAnalyzer(sample_history)
        stats = analyzer.compute_number_stats()
        for _, row in stats.iterrows():
            expected_odd = (int(row["number"]) % 2 == 1)
            assert bool(row["is_odd"]) == expected_odd


class TestComputeDrawStats:
    def test_sum_mean_positive(self, sample_history):
        analyzer = StatsAnalyzer(sample_history)
        ds = analyzer.compute_draw_stats()
        assert ds["sum_mean"] > 0

    def test_odd_mean_between_0_and_6(self, sample_history):
        analyzer = StatsAnalyzer(sample_history)
        ds = analyzer.compute_draw_stats()
        assert 0 <= ds["odd_mean"] <= 6

    def test_band_means_keys(self, sample_history):
        analyzer = StatsAnalyzer(sample_history)
        ds = analyzer.compute_draw_stats()
        expected_keys = {"1~10", "11~20", "21~30", "31~40", "41~45"}
        assert set(ds["band_means"].keys()) == expected_keys

    def test_band_means_sum_equals_6(self, sample_history):
        analyzer = StatsAnalyzer(sample_history)
        ds = analyzer.compute_draw_stats()
        total = sum(ds["band_means"].values())
        assert abs(total - 6.0) < 1e-6


class TestSave:
    def test_save_creates_csv(self, tmp_path, sample_history):
        analyzer = StatsAnalyzer(sample_history)
        out = tmp_path / "stats.csv"
        analyzer.save(out)
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 45
