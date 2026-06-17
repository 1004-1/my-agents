"""로컬 CSV 파일 스토리지."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# CSV 컬럼 정의
DRAW_COLUMNS = ["round_no", "date", "num1", "num2", "num3", "num4", "num5", "num6", "bonus"]
GAME_COLUMNS = [
    "generated_at", "strategy", "game_no", "target_round_no",
    "num1", "num2", "num3", "num4", "num5", "num6",
    "purchased", "purchased_at",
]


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
        df = pd.read_csv(self.games_path, dtype={"purchased_at": object})
        # 기존 CSV 컬럼 없을 때 하위 호환 추가
        if "purchased" not in df.columns:
            df["purchased"] = False
        if "purchased_at" not in df.columns:
            df["purchased_at"] = ""
        if "target_round_no" not in df.columns:
            df["target_round_no"] = 0  # 마이그레이션 대상임을 0으로 표시
        # 빈 값으로 인해 float64로 추론된 경우 object로 강제 변환 (pandas 3.x 호환)
        df["purchased_at"]   = df["purchased_at"].fillna("").astype(object)
        df["purchased"]      = df["purchased"].fillna(False)
        df["target_round_no"] = pd.to_numeric(df["target_round_no"], errors="coerce").fillna(0).astype(int)
        return df

    def save_games(self, df: pd.DataFrame) -> None:
        """생성된 게임 DataFrame을 CSV로 저장한다 (덮어쓰기).

        GAME_COLUMNS 순서로 컬럼을 정렬하고 그 외 컬럼은 뒤에 붙인다.
        """
        ordered = [c for c in GAME_COLUMNS if c in df.columns]
        extra   = [c for c in df.columns if c not in GAME_COLUMNS]
        df = df[ordered + extra]
        df.to_csv(self.games_path, index=False, encoding="utf-8-sig")
        logger.info("생성 게임 %d개 저장: %s", len(df), self.games_path)

    def append_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """기존 생성 게임 CSV에 새 행을 추가한다."""
        existing = self.load_games()
        merged = pd.concat([existing, df], ignore_index=True)
        self.save_games(merged)
        return merged

    def mark_purchased_games(self, games: list[list[int]]) -> None:
        """지정된 번호 조합을 구매 완료로 표시한다."""
        df = self.load_games()
        if df.empty:
            return
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for game in games:
            s = sorted(int(x) for x in game)
            mask = (
                (df["num1"].astype(int) == s[0]) &
                (df["num2"].astype(int) == s[1]) &
                (df["num3"].astype(int) == s[2]) &
                (df["num4"].astype(int) == s[3]) &
                (df["num5"].astype(int) == s[4]) &
                (df["num6"].astype(int) == s[5])
            )
            df.loc[mask, "purchased"]    = True
            df.loc[mask, "purchased_at"] = ts
        self.save_games(df)
        logger.info("구매 완료 표시: %d게임", len(games))

    def get_purchased_combos(self) -> set[frozenset[int]]:
        """구매 완료된 번호 조합 집합을 반환한다."""
        df = self.load_games()
        if df.empty or "purchased" not in df.columns:
            return set()
        purchased = df[df["purchased"] == True]
        return {
            frozenset(int(row[f"num{i}"]) for i in range(1, 7))
            for _, row in purchased.iterrows()
        }

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        """필요한 디렉터리를 생성한다."""
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.games_path.parent.mkdir(parents=True, exist_ok=True)
