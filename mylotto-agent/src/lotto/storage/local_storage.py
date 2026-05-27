"""로컬 CSV 파일 스토리지."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# CSV 컬럼 정의
DRAW_COLUMNS = ["round_no", "date", "num1", "num2", "num3", "num4", "num5", "num6", "bonus"]
GAME_COLUMNS = ["generated_at", "strategy", "game_no", "num1", "num2", "num3", "num4", "num5", "num6"]


class LocalStorage:
    """당첨번호·생성게임을 로컬 CSV 파일로 읽고 쓴다."""

    def __init__(self, results_path: str | Path, games_path: str | Path):
        self.results_path = Path(results_path)
        self.games_path = Path(games_path)
        self._ensure_dirs()

    # ──────────────────────────────────────────────────────────────────────
    # 당첨번호 (lotto_draw_results.csv)
    # ──────────────────────────────────────────────────────────────────────

    def load_results(self) -> pd.DataFrame:
        """저장된 당첨번호 CSV를 읽어 DataFrame으로 반환한다.

        파일이 없으면 빈 DataFrame을 반환한다.
        """
        if not self.results_path.exists():
            logger.info("당첨번호 파일 없음 → 빈 DataFrame 반환: %s", self.results_path)
            return pd.DataFrame(columns=DRAW_COLUMNS)
        df = pd.read_csv(self.results_path)
        df["round_no"] = df["round_no"].astype(int)
        logger.info("당첨번호 %d 회차 로드: %s", len(df), self.results_path)
        return df

    def save_results(self, df: pd.DataFrame) -> None:
        """당첨번호 DataFrame을 CSV로 저장한다 (덮어쓰기)."""
        df = df.sort_values("round_no").reset_index(drop=True)
        df.to_csv(self.results_path, index=False, encoding="utf-8-sig")
        logger.info("당첨번호 %d 회차 저장: %s", len(df), self.results_path)

    def upsert_results(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """기존 CSV에 새로운 행을 병합(upsert)하고 저장한다.

        round_no를 기준으로 중복을 제거하며 최신 데이터를 우선한다.
        """
        existing = self.load_results()
        if existing.empty:
            merged = new_df
        else:
            combined = pd.concat([existing, new_df], ignore_index=True)
            merged = combined.drop_duplicates(subset=["round_no"], keep="last")
        self.save_results(merged)
        logger.info("upsert 완료 → 총 %d 회차", len(merged))
        return merged

    def get_latest_round(self) -> int:
        """저장된 가장 최신 회차 번호를 반환한다. 없으면 0."""
        df = self.load_results()
        if df.empty:
            return 0
        return int(df["round_no"].max())

    # ──────────────────────────────────────────────────────────────────────
    # 생성 게임 (generated_games.csv)
    # ──────────────────────────────────────────────────────────────────────

    def load_games(self) -> pd.DataFrame:
        """생성된 게임 CSV를 읽어 DataFrame으로 반환한다."""
        if not self.games_path.exists():
            return pd.DataFrame(columns=GAME_COLUMNS)
        return pd.read_csv(self.games_path)

    def save_games(self, df: pd.DataFrame) -> None:
        """생성된 게임 DataFrame을 CSV로 저장한다 (덮어쓰기)."""
        df.to_csv(self.games_path, index=False, encoding="utf-8-sig")
        logger.info("생성 게임 %d개 저장: %s", len(df), self.games_path)

    def append_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """기존 생성 게임 CSV에 새 행을 추가한다."""
        existing = self.load_games()
        merged = pd.concat([existing, df], ignore_index=True)
        self.save_games(merged)
        return merged

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        """필요한 디렉터리를 생성한다."""
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.games_path.parent.mkdir(parents=True, exist_ok=True)
