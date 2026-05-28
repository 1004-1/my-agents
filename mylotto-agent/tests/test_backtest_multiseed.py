"""tests/test_backtest_multiseed.py — 멀티 시드 백테스트 단위 테스트."""
from __future__ import annotations

import random

import pandas as pd
import pytest

from src.lotto.ml.backtest import (
    MultiSeedResult,
    StrategyMultiSeedStats,
    BacktestResult,
    run_multiseed_backtest,
    save_multiseed_results,
    save_multiseed_summary,
)


# ── 픽스처 ────────────────────────────────────────────────────────────────

def _make_history(n: int = 150, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        nums = sorted(rng.sample(range(1, 46), 6))
        bonus = rng.choice([n for n in range(1, 46) if n not in nums])
        rows.append({
            "round_no": i, "date": f"2020-01-01",
            "num1": nums[0], "num2": nums[1], "num3": nums[2],
            "num4": nums[3], "num5": nums[4], "num6": nums[5],
            "bonus": bonus,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def history_df() -> pd.DataFrame:
    return _make_history(150)


# ── StrategyMultiSeedStats ────────────────────────────────────────────────

class TestStrategyMultiSeedStats:
    def _make_stats(self) -> StrategyMultiSeedStats:
        history = _make_history(80)
        stats = StrategyMultiSeedStats(
            strategy="random", n_seeds=3, total_rounds=10
        )
        for seed in range(1, 4):
            from src.lotto.ml.backtest import run_backtest
            result = run_backtest(
                history=history,
                strategy_name="random",
                start_round=60,
                end_round=70,
                n_games=3,
                min_history_rounds=10,
                seed=seed,
            )
            stats.per_seed.append(result)
        return stats

    def test_n_seeds_values_length(self):
        stats = self._make_stats()
        assert len(stats.avg_best_match_values) == 3
        assert len(stats.match_3_plus_values) == 3
        assert len(stats.match_4_plus_values) == 3
        assert len(stats.coverage_values) == 3
        assert len(stats.diversity_score_values) == 3

    def test_mean_in_valid_range(self):
        stats = self._make_stats()
        assert 0.0 <= stats.avg_best_match_mean <= 6.0
        assert stats.match_3_plus_mean >= 0
        assert stats.match_4_plus_mean >= 0
        assert 0.0 <= stats.coverage_mean <= 1.0
        assert 0.0 <= stats.diversity_score_mean <= 100.0

    def test_std_non_negative(self):
        stats = self._make_stats()
        assert stats.avg_best_match_std >= 0.0
        assert stats.match_3_plus_std >= 0.0
        assert stats.match_4_plus_std >= 0.0

    def test_std_zero_for_single_seed(self):
        """시드 1개이면 표준편차는 0."""
        history = _make_history(80)
        from src.lotto.ml.backtest import run_backtest
        result = run_backtest(
            history=history, strategy_name="random",
            start_round=60, end_round=65,
            n_games=2, min_history_rounds=10, seed=1,
        )
        stats = StrategyMultiSeedStats(strategy="random", n_seeds=1, total_rounds=6)
        stats.per_seed.append(result)
        assert stats.avg_best_match_std == 0.0


# ── MultiSeedResult ────────────────────────────────────────────────────────

class TestMultiSeedResult:
    def _make_msr(self) -> MultiSeedResult:
        msr = MultiSeedResult(n_seeds=2, total_rounds=5)
        for sname in ("random", "balanced"):
            s = StrategyMultiSeedStats(strategy=sname, n_seeds=2, total_rounds=5)
            msr.strategy_stats.append(s)
        return msr

    def test_get_stats_existing(self):
        msr = self._make_msr()
        s = msr.get_stats("random")
        assert s is not None
        assert s.strategy == "random"

    def test_get_stats_missing(self):
        msr = self._make_msr()
        assert msr.get_stats("model_score") is None

    def test_strategy_stats_count(self):
        msr = self._make_msr()
        assert len(msr.strategy_stats) == 2


# ── run_multiseed_backtest ─────────────────────────────────────────────────

class TestRunMultiseedBacktest:
    def test_returns_multiseed_result(self, history_df):
        result = run_multiseed_backtest(
            history=history_df,
            strategy_names=["random"],
            n_seeds=2,
            start_round=120,
            end_round=130,
            n_games=3,
            min_history_rounds=10,
        )
        assert isinstance(result, MultiSeedResult)

    def test_correct_n_seeds(self, history_df):
        result = run_multiseed_backtest(
            history=history_df,
            strategy_names=["random"],
            n_seeds=3,
            start_round=120,
            end_round=125,
            n_games=2,
            min_history_rounds=10,
        )
        stats = result.get_stats("random")
        assert stats is not None
        assert len(stats.per_seed) == 3

    def test_multiple_strategies(self, history_df):
        result = run_multiseed_backtest(
            history=history_df,
            strategy_names=["random", "balanced"],
            n_seeds=2,
            start_round=120,
            end_round=125,
            n_games=2,
            min_history_rounds=10,
        )
        assert len(result.strategy_stats) == 2
        assert result.get_stats("random") is not None
        assert result.get_stats("balanced") is not None

    def test_each_seed_is_different(self, history_df):
        """시드가 다르면 결과가 달라야 한다 (확률적)."""
        result = run_multiseed_backtest(
            history=history_df,
            strategy_names=["random"],
            n_seeds=5,
            start_round=120,
            end_round=130,
            n_games=5,
            min_history_rounds=10,
        )
        stats = result.get_stats("random")
        # 5개 시드의 match_3_plus가 모두 동일할 가능성은 극히 낮음
        vals = stats.match_3_plus_values
        assert not all(v == vals[0] for v in vals), "모든 시드 결과가 동일 — seed 미작동 의심"

    def test_empty_history_raises(self):
        empty = pd.DataFrame(
            columns=["round_no", "date", "num1", "num2", "num3", "num4", "num5", "num6", "bonus"]
        )
        with pytest.raises(ValueError):
            run_multiseed_backtest(empty, strategy_names=["random"], n_seeds=2)

    def test_balanced_v2_supported(self, history_df):
        result = run_multiseed_backtest(
            history=history_df,
            strategy_names=["balanced_v2"],
            n_seeds=2,
            start_round=120,
            end_round=125,
            n_games=2,
            min_history_rounds=10,
        )
        assert result.get_stats("balanced_v2") is not None

    def test_target_rounds_count(self, history_df):
        result = run_multiseed_backtest(
            history=history_df,
            strategy_names=["random"],
            n_seeds=2,
            start_round=121,
            end_round=130,
            n_games=2,
            min_history_rounds=10,
        )
        assert result.total_rounds == 10

    def test_total_rounds_matches_per_seed(self, history_df):
        result = run_multiseed_backtest(
            history=history_df,
            strategy_names=["random"],
            n_seeds=2,
            start_round=121,
            end_round=130,
            n_games=2,
            min_history_rounds=10,
        )
        stats = result.get_stats("random")
        for bt in stats.per_seed:
            assert len(bt.per_round) == result.total_rounds


# ── 저장 함수 ─────────────────────────────────────────────────────────────

class TestSaveMultiseed:
    def _run(self, history_df) -> MultiSeedResult:
        return run_multiseed_backtest(
            history=history_df,
            strategy_names=["random", "balanced"],
            n_seeds=2,
            start_round=120,
            end_round=124,
            n_games=2,
            min_history_rounds=10,
        )

    def test_save_results_creates_file(self, tmp_path, history_df):
        result = self._run(history_df)
        out = tmp_path / "multiseed_results.csv"
        save_multiseed_results(result, out)
        assert out.exists()

    def test_save_results_has_seed_column(self, tmp_path, history_df):
        result = self._run(history_df)
        out = tmp_path / "multiseed_results.csv"
        save_multiseed_results(result, out)
        df = pd.read_csv(out)
        assert "seed" in df.columns
        assert set(df["seed"].unique()) == {1, 2}

    def test_save_results_row_count(self, tmp_path, history_df):
        result = self._run(history_df)
        out = tmp_path / "multiseed_results.csv"
        save_multiseed_results(result, out)
        df = pd.read_csv(out)
        # 2전략 × 2시드 × 5회차 × 2게임 = 40행
        assert len(df) == 2 * 2 * 5 * 2

    def test_save_summary_creates_file(self, tmp_path, history_df):
        result = self._run(history_df)
        out = tmp_path / "multiseed_summary.csv"
        save_multiseed_summary(result, out)
        assert out.exists()

    def test_save_summary_has_strategy_rows(self, tmp_path, history_df):
        result = self._run(history_df)
        out = tmp_path / "multiseed_summary.csv"
        save_multiseed_summary(result, out)
        df = pd.read_csv(out)
        assert set(df["strategy"]) == {"random", "balanced"}

    def test_save_summary_columns(self, tmp_path, history_df):
        result = self._run(history_df)
        out = tmp_path / "multiseed_summary.csv"
        save_multiseed_summary(result, out)
        df = pd.read_csv(out)
        expected_cols = {
            "strategy", "n_seeds", "total_rounds",
            "avg_best_match_mean", "avg_best_match_std",
            "match_3_plus_mean", "match_3_plus_std",
            "match_4_plus_mean", "match_4_plus_std",
            "coverage_mean", "diversity_score_mean", "vs_random_3plus_pct",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_vs_random_is_nan_for_random(self, tmp_path, history_df):
        result = self._run(history_df)
        out = tmp_path / "multiseed_summary.csv"
        save_multiseed_summary(result, out)
        df = pd.read_csv(out)
        random_row = df[df["strategy"] == "random"].iloc[0]
        # random의 vs_random은 NaN
        assert pd.isna(random_row["vs_random_3plus_pct"])


# ── 기존 run_backtest 회귀 테스트 ─────────────────────────────────────────

class TestRunBacktestRegression:
    """refactoring 후 run_backtest가 기존과 동일하게 동작하는지 확인한다."""

    def test_result_structure_unchanged(self, history_df):
        from src.lotto.ml.backtest import run_backtest
        result = run_backtest(
            history=history_df,
            strategy_name="random",
            start_round=120,
            end_round=130,
            n_games=3,
            min_history_rounds=10,
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy == "random"
        assert len(result.per_round) == 11

    def test_match_counts_valid_range(self, history_df):
        from src.lotto.ml.backtest import run_backtest
        result = run_backtest(
            history=history_df,
            strategy_name="balanced",
            start_round=120,
            end_round=125,
            n_games=3,
            min_history_rounds=10,
        )
        for rr in result.per_round:
            for mc in rr.match_counts:
                assert 0 <= mc <= 6

    def test_unknown_strategy_still_raises(self, history_df):
        from src.lotto.ml.backtest import run_backtest
        with pytest.raises(ValueError, match="알 수 없는 전략"):
            run_backtest(
                history=history_df,
                strategy_name="no_such_strategy",
                start_round=120,
                end_round=125,
                min_history_rounds=10,
            )
