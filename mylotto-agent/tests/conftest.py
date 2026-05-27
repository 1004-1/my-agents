"""pytest 공용 fixture."""

import random as _random

import pandas as pd
import pytest


@pytest.fixture
def sample_history() -> pd.DataFrame:
    """테스트용 당첨번호 샘플 DataFrame (10회차)."""
    records = [
        {"round_no": 1, "date": "2002-12-07", "num1": 10, "num2": 23, "num3": 29,
         "num4": 33, "num5": 37, "num6": 40, "bonus": 16},
        {"round_no": 2, "date": "2002-12-14", "num1": 9, "num2": 13, "num3": 21,
         "num4": 25, "num5": 32, "num6": 42, "bonus": 2},
        {"round_no": 3, "date": "2002-12-21", "num1": 11, "num2": 16, "num3": 19,
         "num4": 21, "num5": 27, "num6": 31, "bonus": 30},
        {"round_no": 4, "date": "2002-12-28", "num1": 14, "num2": 27, "num3": 30,
         "num4": 31, "num5": 40, "num6": 42, "bonus": 2},
        {"round_no": 5, "date": "2003-01-04", "num1": 16, "num2": 24, "num3": 29,
         "num4": 40, "num5": 41, "num6": 42, "bonus": 3},
        {"round_no": 6, "date": "2003-01-11", "num1": 14, "num2": 15, "num3": 26,
         "num4": 27, "num5": 40, "num6": 42, "bonus": 34},
        {"round_no": 7, "date": "2003-01-18", "num1": 2, "num2": 9, "num3": 16,
         "num4": 25, "num5": 26, "num6": 40, "bonus": 42},
        {"round_no": 8, "date": "2003-01-25", "num1": 8, "num2": 19, "num3": 25,
         "num4": 34, "num5": 37, "num6": 39, "bonus": 20},
        {"round_no": 9, "date": "2003-02-01", "num1": 2, "num2": 4, "num3": 7,
         "num4": 10, "num5": 20, "num6": 23, "bonus": 45},
        {"round_no": 10, "date": "2003-02-08", "num1": 9, "num2": 25, "num3": 30,
         "num4": 33, "num5": 41, "num6": 44, "bonus": 6},
    ]
    return pd.DataFrame(records)


@pytest.fixture
def larger_history() -> pd.DataFrame:
    """테스트용 당첨번호 샘플 DataFrame (50회차 — feature/backtest 테스트용)."""
    rng = _random.Random(42)
    records = []
    for i in range(1, 51):
        nums = sorted(rng.sample(range(1, 46), 6))
        remaining = [n for n in range(1, 46) if n not in nums]
        bonus = rng.choice(remaining)
        records.append({
            "round_no": i,
            "date": "2002-12-07",
            "num1": nums[0], "num2": nums[1], "num3": nums[2],
            "num4": nums[3], "num5": nums[4], "num6": nums[5],
            "bonus": bonus,
        })
    return pd.DataFrame(records)


@pytest.fixture
def mock_api_response() -> dict:
    """동행복권 API 응답 샘플."""
    return {
        "returnValue": "success",
        "drwNo": 1100,
        "drwNoDate": "2024-01-06",
        "drwtNo1": 1,
        "drwtNo2": 12,
        "drwtNo3": 23,
        "drwtNo4": 34,
        "drwtNo5": 45,
        "drwtNo6": 7,
        "bnusNo": 8,
        "firstWinamnt": 2_000_000_000,
        "firstPrzwnerCo": 3,
        "totSellamnt": 100_000_000_000,
    }
