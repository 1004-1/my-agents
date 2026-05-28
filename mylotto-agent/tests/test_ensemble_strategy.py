"""EnsembleStrategy 및 compute_diversity 단위 테스트."""
from __future__ import annotations

import random

import pandas as pd
import pytest

from src.lotto.strategy.ensemble_strategy import (
    EnsembleStrategy,
    compute_diversity,
    ENSEMBLE_TOTAL,
)


def _make_history(n_rounds: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    records = []
    for i in range(1, n_rounds + 1):
        nums   = sorted(rng.sample(range(1, 46), 6))
        others = [n for n in range(1, 46) if n not in nums]
        bonus  = rng.choice(others)
        records.append({
            "round_no": i,
            "date": "2002-12-07",
            "num1": nums[0], "num2": nums[1], "num3": nums[2],
            "num4": nums[3], "num5": nums[4], "num6": nums[5],
            "bonus": bonus,
        })
    return pd.DataFrame(records)


# ── compute_diversity ─────────────────────────────────────────────────────

class TestComputeDiversity:
    def test_empty_games(self):
        d = compute_diversity([])
        assert d["coverage"] == 0.0
        assert d["avg_jaccard"] == 0.0
        assert d["diversity_score"] == 0.0

    def test_single_game(self):
        d = compute_diversity([[1, 2, 3, 4, 5, 6]])
        assert abs(d["coverage"] - 6 / 45) < 1e-9
        assert d["avg_jaccard"] == 0.0   # 쌍 없음

    def test_identical_games_max_jaccard(self):
        games = [[1, 2, 3, 4, 5, 6]] * 3
        d = compute_diversity(games)
        assert abs(d["avg_jaccard"] - 1.0) < 1e-9

    def test_disjoint_games_zero_jaccard(self):
        """6개 번호가 완전히 겹치지 않는 두 게임 → Jaccard=0."""
        g1 = [1, 2, 3, 4, 5, 6]
        g2 = [7, 8, 9, 10, 11, 12]
        d = compute_diversity([g1, g2])
        assert abs(d["avg_jaccard"] - 0.0) < 1e-9

    def test_coverage_range(self):
        games = _make_history(1)  # not games, just checking API
        g = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
             [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24],
             [25, 26, 27, 28, 29, 30]]
        d = compute_diversity(g)
        assert abs(d["coverage"] - 30 / 45) < 1e-9

    def test_diversity_score_in_range(self):
        games = [[i*6+j+1 for j in range(6)] for i in range(5)]
        d = compute_diversity(games)
        assert 0.0 <= d["diversity_score"] <= 100.0

    def test_jaccard_formula(self):
        """4개 공유: Jaccard = 4 / (6+6-4) = 4/8 = 0.5."""
        g1 = [1, 2, 3, 4, 5, 6]
        g2 = [1, 2, 3, 4, 20, 21]   # 4개 공유
        d = compute_diversity([g1, g2])
        assert abs(d["avg_jaccard"] - 0.5) < 1e-9


# ── EnsembleStrategy 기본 생성 ────────────────────────────────────────────

class TestEnsembleGenerate:
    def test_generates_ensemble_total_games(self, larger_history):
        strategy = EnsembleStrategy(seed=42)
        games = strategy.generate(history=larger_history)
        assert len(games) == ENSEMBLE_TOTAL  # = 5

    def test_each_game_has_6_numbers(self, larger_history):
        strategy = EnsembleStrategy(seed=42)
        for game in strategy.generate(history=larger_history):
            assert len(game) == 6

    def test_numbers_in_range(self, larger_history):
        strategy = EnsembleStrategy(seed=42)
        for game in strategy.generate(history=larger_history):
            assert all(1 <= n <= 45 for n in game)

    def test_no_duplicates_within_game(self, larger_history):
        strategy = EnsembleStrategy(seed=42)
        for game in strategy.generate(history=larger_history):
            assert len(set(game)) == 6

    def test_sorted_output(self, larger_history):
        strategy = EnsembleStrategy(seed=42)
        for game in strategy.generate(history=larger_history):
            assert game == sorted(game)

    def test_reproducible_with_seed(self, larger_history):
        s1 = EnsembleStrategy(seed=7)
        s2 = EnsembleStrategy(seed=7)
        assert s1.generate(history=larger_history) == s2.generate(history=larger_history)

    def test_without_history(self):
        """history=None이어도 예외 없이 게임을 생성한다."""
        strategy = EnsembleStrategy(seed=42)
        games = strategy.generate(history=None)
        assert len(games) == ENSEMBLE_TOTAL
        for g in games:
            assert len(g) == 6

    def test_with_model_scores_bypass(self, larger_history):
        """from_scores 방식(백테스트 경량 모드) 동작 확인."""
        scores   = {n: 1.0 / 45 for n in range(1, 46)}
        strategy = EnsembleStrategy(model_scores=scores, seed=0)
        games    = strategy.generate(history=larger_history)
        assert len(games) == ENSEMBLE_TOTAL


# ── 다양성 후처리 ─────────────────────────────────────────────────────────

class TestEnsembleDiversity:
    def test_no_identical_games_usually(self, larger_history):
        """다양성 후처리로 동일 게임 조합이 없어야 한다."""
        strategy = EnsembleStrategy(seed=42)
        games    = strategy.generate(history=larger_history)
        tuples   = [tuple(g) for g in games]
        # 5게임 중 유니크 게임이 최소 4개이어야 함
        assert len(set(tuples)) >= 4

    def test_generate_with_diversity_returns_tuple(self, larger_history):
        strategy = EnsembleStrategy(seed=42)
        games, div = strategy.generate_with_diversity(history=larger_history)
        assert len(games) == ENSEMBLE_TOTAL
        assert "coverage" in div
        assert "avg_jaccard" in div
        assert "diversity_score" in div

    def test_diversity_coverage_reasonable(self, larger_history):
        """5게임이면 최소 6개, 최대 30개의 고유 번호가 사용되어야 한다."""
        strategy = EnsembleStrategy(seed=42)
        games, div = strategy.generate_with_diversity(history=larger_history)
        all_nums = set()
        for g in games:
            all_nums.update(g)
        assert 6 <= len(all_nums) <= 30
        assert abs(div["coverage"] - len(all_nums) / 45) < 1e-9

    def test_max_jaccard_threshold_respected(self, larger_history):
        """max_jaccard=0.5일 때 모든 쌍의 Jaccard가 0.5 이하이어야 한다 (대부분)."""
        strategy = EnsembleStrategy(seed=42, max_jaccard=0.5)
        # 10번 반복해 평균적으로 임계 초과가 드물어야 함
        violations = 0
        for s in range(10):
            strategy2 = EnsembleStrategy(seed=s, max_jaccard=0.5)
            games = strategy2.generate(history=larger_history)
            for i in range(len(games)):
                for j in range(i + 1, len(games)):
                    a, b = set(games[i]), set(games[j])
                    if len(a & b) / len(a | b) > 0.5:
                        violations += 1
        # 10 시드 × 10 쌍 = 100회 중 위반 10회 이하 허용
        assert violations <= 10


# ── 백테스트 호환 ─────────────────────────────────────────────────────────

class TestEnsembleBacktestCompat:
    def test_backtest_ensemble_runs(self, larger_history):
        """백테스트에서 ensemble 전략이 정상 실행되는지 확인."""
        from src.lotto.ml.backtest import run_backtest

        result = run_backtest(
            history=larger_history,
            strategy_name="ensemble",
            start_round=30,
            end_round=35,
            n_games=5,
            min_history_rounds=10,
        )
        assert result.strategy == "ensemble"
        assert result.total_rounds == 6  # 30~35 inclusive
        for rr in result.per_round:
            assert len(rr.games) == ENSEMBLE_TOTAL

    def test_backtest_diversity_fields_populated(self, larger_history):
        """백테스트 결과에 diversity 필드가 채워지는지 확인."""
        from src.lotto.ml.backtest import run_backtest

        result = run_backtest(
            history=larger_history,
            strategy_name="ensemble",
            start_round=30,
            end_round=33,
            n_games=5,
            min_history_rounds=10,
        )
        assert result.avg_coverage > 0.0
        assert result.avg_diversity_score > 0.0
        for rr in result.per_round:
            assert "coverage" in rr.diversity
            assert "avg_jaccard" in rr.diversity

    def test_backtest_gap_based_runs(self, larger_history):
        """gap_based 전략도 백테스트에서 정상 실행."""
        from src.lotto.ml.backtest import run_backtest

        result = run_backtest(
            history=larger_history,
            strategy_name="gap_based",
            start_round=25,
            end_round=30,
            n_games=3,
            min_history_rounds=10,
        )
        assert result.strategy == "gap_based"
        assert result.total_rounds == 6
