"""전략(Strategy) 단위 테스트."""

import pytest

from src.lotto.strategy.random_strategy import RandomStrategy
from src.lotto.strategy.balanced_strategy import BalancedStrategy
from src.lotto.strategy.base import LOTTO_MIN, LOTTO_MAX, NUMBERS_PER_GAME


# ══════════════════════════════════════════════════════════════════════════
# 공통 헬퍼
# ══════════════════════════════════════════════════════════════════════════

def assert_valid_games(games: list[list[int]], n: int) -> None:
    """생성된 게임 목록이 규칙에 맞는지 검증한다."""
    assert len(games) == n, f"게임 수 불일치: {len(games)} != {n}"
    for game in games:
        assert len(game) == NUMBERS_PER_GAME, f"번호 수 오류: {game}"
        assert len(set(game)) == NUMBERS_PER_GAME, f"중복 번호: {game}"
        assert all(LOTTO_MIN <= n <= LOTTO_MAX for n in game), f"범위 초과: {game}"
        assert game == sorted(game), f"정렬 오류: {game}"


# ══════════════════════════════════════════════════════════════════════════
# RandomStrategy
# ══════════════════════════════════════════════════════════════════════════

class TestRandomStrategy:
    def test_generate_5_games_default(self):
        strategy = RandomStrategy(seed=42)
        games = strategy.generate()
        assert_valid_games(games, 5)

    def test_generate_custom_n(self):
        strategy = RandomStrategy(seed=0)
        games = strategy.generate(n_games=3)
        assert_valid_games(games, 3)

    def test_reproducible_with_seed(self):
        games1 = RandomStrategy(seed=123).generate(5)
        games2 = RandomStrategy(seed=123).generate(5)
        assert games1 == games2

    def test_different_seeds_differ(self):
        games1 = RandomStrategy(seed=1).generate(5)
        games2 = RandomStrategy(seed=2).generate(5)
        assert games1 != games2

    def test_history_ignored(self, sample_history):
        """history를 전달해도 동일하게 동작해야 한다."""
        games = RandomStrategy(seed=7).generate(5, history=sample_history)
        assert_valid_games(games, 5)

    def test_validate_game_true(self):
        assert RandomStrategy.validate_game([1, 5, 10, 20, 30, 45])

    def test_validate_game_wrong_count(self):
        assert not RandomStrategy.validate_game([1, 5, 10, 20, 30])

    def test_validate_game_duplicate(self):
        assert not RandomStrategy.validate_game([1, 1, 10, 20, 30, 45])

    def test_validate_game_out_of_range(self):
        assert not RandomStrategy.validate_game([0, 5, 10, 20, 30, 45])


# ══════════════════════════════════════════════════════════════════════════
# BalancedStrategy
# ══════════════════════════════════════════════════════════════════════════

class TestBalancedStrategy:
    def test_generate_5_games_default(self):
        strategy = BalancedStrategy(seed=42)
        games = strategy.generate()
        assert_valid_games(games, 5)

    def test_generate_with_history(self, sample_history):
        strategy = BalancedStrategy(seed=42)
        games = strategy.generate(n_games=5, history=sample_history)
        assert_valid_games(games, 5)

    def test_generate_without_history(self):
        strategy = BalancedStrategy(seed=99)
        games = strategy.generate(n_games=5, history=None)
        assert_valid_games(games, 5)

    def test_reproducible_with_seed(self, sample_history):
        games1 = BalancedStrategy(seed=77).generate(5, sample_history)
        games2 = BalancedStrategy(seed=77).generate(5, sample_history)
        assert games1 == games2

    @pytest.mark.parametrize("seed", range(5))
    def test_odd_even_balance(self, seed):
        """홀수 개수가 2~4개 범위인지 확인한다 (대부분의 게임)."""
        strategy = BalancedStrategy(seed=seed * 100)
        games = strategy.generate(n_games=20)
        violations = []
        for game in games:
            odd_count = sum(1 for n in game if n % 2 == 1)
            if not (2 <= odd_count <= 4):
                violations.append((game, odd_count))
        # 폴백 케이스가 있을 수 있으므로 10% 미만만 허용
        assert len(violations) / len(games) < 0.1, f"홀짝 위반 게임: {violations}"

    def test_name_attribute(self):
        assert BalancedStrategy.name == "balanced"

    def test_random_strategy_name(self):
        assert RandomStrategy.name == "random"
