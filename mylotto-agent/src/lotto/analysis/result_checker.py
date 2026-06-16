"""생성된 번호의 실제 당첨번호 대비 적중 결과를 추적한다."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS = [
    "generated_at", "target_round_no", "strategy_name",
    "numbers", "winning_numbers", "bonus",
    "match_count", "bonus_matched", "rank", "reward_estimate", "checked_at",
]

# 등수별 예상 상금 (1·2등은 회차마다 달라 문자열로 표기)
_RANK_REWARDS: dict[int | None, int | str] = {
    1: "jackpot",
    2: "varies",
    3: 1_500_000,
    4: 50_000,
    5: 5_000,
}


def get_rank(match_count: int, bonus_matched: bool) -> int | None:
    """적중 개수와 보너스 일치 여부로 로또 등수를 반환한다."""
    if match_count == 6:
        return 1
    if match_count == 5 and bonus_matched:
        return 2
    if match_count == 5:
        return 3
    if match_count == 4:
        return 4
    if match_count == 3:
        return 5
    return None


def get_reward_estimate(rank: int | None) -> int | str:
    return _RANK_REWARDS.get(rank, 0)


class ResultChecker:
    """generated_games.csv ↔ lotto_draw_results.csv 적중 결과 비교기."""

    def __init__(
        self,
        games_path:      Path = Path("data/generated_games.csv"),
        results_path:    Path = Path("data/lotto_draw_results.csv"),
        prediction_path: Path = Path("data/prediction_results.csv"),
    ):
        self.games_path      = Path(games_path)
        self.results_path    = Path(results_path)
        self.prediction_path = Path(prediction_path)

    # ──────────────────────────────────────────────────────────────────────
    # 로드 헬퍼
    # ──────────────────────────────────────────────────────────────────────

    def _load_draws(self) -> pd.DataFrame:
        if not self.results_path.exists():
            return pd.DataFrame(
                columns=["round_no", "date", "num1", "num2", "num3",
                         "num4", "num5", "num6", "bonus", "draw_date"]
            )
        df = pd.read_csv(self.results_path)
        df["round_no"]  = df["round_no"].astype(int)
        df["draw_date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values("round_no").reset_index(drop=True)

    def _load_games(self) -> pd.DataFrame:
        if not self.games_path.exists():
            return pd.DataFrame()
        from ..storage.local_storage import LocalStorage
        st = LocalStorage(results_path=self.results_path, games_path=self.games_path)
        return st.load_games()

    def load_predictions(self) -> pd.DataFrame:
        if not self.prediction_path.exists():
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        return pd.read_csv(
            self.prediction_path,
            dtype={"rank": object, "reward_estimate": object},
        )

    # ──────────────────────────────────────────────────────────────────────
    # 핵심 로직
    # ──────────────────────────────────────────────────────────────────────

    def find_target_round(
        self, generated_at: str, draws_df: pd.DataFrame
    ) -> int | None:
        """generated_at 이후 가장 가까운 추첨 회차를 반환한다.

        추첨이 아직 없으면 None.
        """
        try:
            gen_date = pd.Timestamp(generated_at).date()
        except Exception:
            return None
        future = draws_df[draws_df["draw_date"] > gen_date]
        if future.empty:
            return None
        return int(future["round_no"].min())

    @staticmethod
    def _game_key(generated_at: str, numbers: str) -> str:
        return f"{generated_at}|{numbers}"

    def check_all(self) -> tuple[pd.DataFrame, int, int]:
        """모든 생성 게임에 대해 적중 결과를 확인한다.

        Returns:
            (전체_prediction_df, 새로_추가된_수, 미추첨으로_건너뛴_수)
        """
        games_df   = self._load_games()
        draws_df   = self._load_draws()
        existing   = self.load_predictions()

        if games_df.empty:
            return existing, 0, 0

        # 이미 체크된 키 집합 (generated_at + numbers)
        if not existing.empty:
            existing_keys: set[str] = set(
                existing["generated_at"].astype(str) + "|" + existing["numbers"].astype(str)
            )
        else:
            existing_keys = set()

        new_rows: list[dict] = []
        skipped = 0
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        for _, row in games_df.iterrows():
            game_nums = sorted(int(row[f"num{i}"]) for i in range(1, 7))
            numbers   = ",".join(str(n) for n in game_nums)
            key       = self._game_key(str(row["generated_at"]), numbers)

            if key in existing_keys:
                continue

            target_round = self.find_target_round(str(row["generated_at"]), draws_df)
            if target_round is None:
                skipped += 1
                continue

            draw_row = draws_df[draws_df["round_no"] == target_round]
            if draw_row.empty:
                skipped += 1
                continue

            draw    = draw_row.iloc[0]
            winning = sorted(int(draw[f"num{i}"]) for i in range(1, 7))
            bonus   = int(draw["bonus"])

            match_count   = len(set(game_nums) & set(winning))
            bonus_matched = (bonus in game_nums) and (match_count < 6)
            rank          = get_rank(match_count, bonus_matched)
            reward        = get_reward_estimate(rank)

            new_rows.append({
                "generated_at":   row["generated_at"],
                "target_round_no": target_round,
                "strategy_name":  row["strategy"],
                "numbers":        numbers,
                "winning_numbers": ",".join(str(n) for n in winning),
                "bonus":          bonus,
                "match_count":    match_count,
                "bonus_matched":  bonus_matched,
                "rank":           rank if rank is not None else "",
                "reward_estimate": reward,
                "checked_at":     now,
            })

        if new_rows:
            combined = pd.concat(
                [existing, pd.DataFrame(new_rows)], ignore_index=True
            )
        else:
            combined = existing.copy()

        self.prediction_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(self.prediction_path, index=False, encoding="utf-8-sig")
        logger.info(
            "prediction_results 저장: %d행 (신규 %d, 스킵 %d)",
            len(combined), len(new_rows), skipped,
        )
        return combined, len(new_rows), skipped

    # ──────────────────────────────────────────────────────────────────────
    # 전략별 성과 집계
    # ──────────────────────────────────────────────────────────────────────

    def strategy_summary(self) -> pd.DataFrame:
        """prediction_results.csv 기준 전략별 성과를 집계한다."""
        df = self.load_predictions()
        if df.empty:
            return pd.DataFrame()

        df["match_count"] = pd.to_numeric(df["match_count"], errors="coerce").fillna(0).astype(int)
        df["rank_int"]    = pd.to_numeric(df["rank"], errors="coerce")

        rows = []
        for strategy, grp in df.groupby("strategy_name"):
            rank_dist = (
                grp["rank_int"].dropna().astype(int).value_counts().to_dict()
            )
            rows.append({
                "strategy":      strategy,
                "total_games":   len(grp),
                "avg_match":     round(grp["match_count"].mean(), 3),
                "match_3plus":   int((grp["match_count"] >= 3).sum()),
                "match_4plus":   int((grp["match_count"] >= 4).sum()),
                "match_5plus":   int((grp["match_count"] >= 5).sum()),
                "max_match":     int(grp["match_count"].max()),
                "rank_5":        rank_dist.get(5, 0),
                "rank_4":        rank_dist.get(4, 0),
                "rank_3":        rank_dist.get(3, 0),
                "rank_2":        rank_dist.get(2, 0),
                "rank_1":        rank_dist.get(1, 0),
            })

        return pd.DataFrame(rows).sort_values("avg_match", ascending=False)
