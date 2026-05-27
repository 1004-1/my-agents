"""LocalStorage 단위 테스트 — tmp_path 픽스처로 임시 파일 사용."""

import pandas as pd
import pytest

from src.lotto.storage.local_storage import LocalStorage, DRAW_COLUMNS, GAME_COLUMNS


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(
        results_path=tmp_path / "results.csv",
        games_path=tmp_path / "games.csv",
    )


@pytest.fixture
def draw_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"round_no": 1, "date": "2002-12-07", "num1": 10, "num2": 23, "num3": 29,
         "num4": 33, "num5": 37, "num6": 40, "bonus": 16},
        {"round_no": 2, "date": "2002-12-14", "num1": 9, "num2": 13, "num3": 21,
         "num4": 25, "num5": 32, "num6": 42, "bonus": 2},
    ])


@pytest.fixture
def game_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"generated_at": "2024-01-01T00:00:00+00:00", "strategy": "random",
         "game_no": 1, "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6},
    ])


# ══════════════════════════════════════════════════════════════════════════
# load_results / save_results
# ══════════════════════════════════════════════════════════════════════════

class TestResultsCSV:
    def test_load_returns_empty_when_no_file(self, storage):
        df = storage.load_results()
        assert df.empty
        assert list(df.columns) == DRAW_COLUMNS

    def test_save_and_load_roundtrip(self, storage, draw_df):
        storage.save_results(draw_df)
        loaded = storage.load_results()
        assert len(loaded) == 2
        assert loaded["round_no"].tolist() == [1, 2]

    def test_save_sorts_by_round_no(self, storage, draw_df):
        shuffled = draw_df.iloc[::-1].reset_index(drop=True)
        storage.save_results(shuffled)
        loaded = storage.load_results()
        assert loaded["round_no"].is_monotonic_increasing

    def test_upsert_adds_new_rows(self, storage, draw_df):
        storage.save_results(draw_df)
        new_row = pd.DataFrame([{
            "round_no": 3, "date": "2002-12-21",
            "num1": 11, "num2": 16, "num3": 19, "num4": 21, "num5": 27, "num6": 31, "bonus": 30,
        }])
        merged = storage.upsert_results(new_row)
        assert len(merged) == 3

    def test_upsert_deduplicates(self, storage, draw_df):
        storage.save_results(draw_df)
        # 같은 회차를 다시 upsert
        merged = storage.upsert_results(draw_df)
        assert len(merged) == 2  # 중복 제거

    def test_get_latest_round_empty(self, storage):
        assert storage.get_latest_round() == 0

    def test_get_latest_round_after_save(self, storage, draw_df):
        storage.save_results(draw_df)
        assert storage.get_latest_round() == 2


# ══════════════════════════════════════════════════════════════════════════
# load_games / save_games / append_games
# ══════════════════════════════════════════════════════════════════════════

class TestGamesCSV:
    def test_load_returns_empty_when_no_file(self, storage):
        df = storage.load_games()
        assert df.empty

    def test_save_and_load_roundtrip(self, storage, game_df):
        storage.save_games(game_df)
        loaded = storage.load_games()
        assert len(loaded) == 1

    def test_append_grows_file(self, storage, game_df):
        storage.save_games(game_df)
        storage.append_games(game_df)
        loaded = storage.load_games()
        assert len(loaded) == 2
