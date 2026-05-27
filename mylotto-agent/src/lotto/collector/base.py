"""수집기 추상 베이스 클래스."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseCollector(ABC):
    """모든 수집기가 구현해야 할 인터페이스."""

    @abstractmethod
    def fetch_latest_round(self) -> int:
        """동행복권 최신 회차 번호를 반환한다."""
        ...

    @abstractmethod
    def fetch_draw(self, round_no: int) -> dict:
        """특정 회차의 당첨 결과를 딕셔너리로 반환한다.

        반환 형식:
            {
                "round_no": int,
                "date": str (YYYY-MM-DD),
                "num1": int, "num2": int, "num3": int,
                "num4": int, "num5": int, "num6": int,
                "bonus": int,
            }
        """
        ...

    @abstractmethod
    def fetch_range(self, start: int, end: int) -> pd.DataFrame:
        """start~end 회차 범위의 당첨 결과를 DataFrame으로 반환한다."""
        ...
