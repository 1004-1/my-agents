"""동행복권(dhlottery.co.kr) 당첨번호 수집기.

Features:
    - 날짜 기반 이진탐색으로 최신 회차 자동 탐색
    - requests 재시도 (지수 백오프, 최대 3회)
    - 커스텀 User-Agent / Referer 헤더
    - 회차별 수집 결과를 FetchResult 로 캡슐화
    - 데이터 유효성 검증 (범위·중복·보너스·날짜 형식)
    - 수집 통계를 CollectStats 로 반환
    - 진행 상황 콜백(on_progress) 지원
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

from .base import BaseCollector
from .validator import validate_draw

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────
_BASE_URL = "https://www.dhlottery.co.kr/common.do"
_DEFAULT_TIMEOUT = 10           # 단일 요청 타임아웃 (초)
_RATE_LIMIT_SLEEP = 0.35        # API 과부하 방지 딜레이 (초)
_MAX_RETRIES = 3                # 재시도 횟수
_BACKOFF_FACTOR = 0.5           # 재시도 대기 배율 (0.5→ 0.5s, 1s, 2s)
_RETRY_STATUS = (429, 500, 502, 503, 504)
_FIRST_DRAW_DATE = date(2002, 12, 7)   # 1회차 추첨일

_USER_AGENT = (
    "Mozilla/5.0 (compatible; mylotto-agent/0.1; "
    "+https://github.com/your-org/mylotto-agent)"
)
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.dhlottery.co.kr/gameResult.do?method=byWin",
    "X-Requested-With": "XMLHttpRequest",
}

# API 응답 키 → 내부 컬럼명 매핑
_API_KEY_MAP: dict[str, str] = {
    "drwNo":    "round_no",
    "drwNoDate": "date",
    "drwtNo1":  "num1",
    "drwtNo2":  "num2",
    "drwtNo3":  "num3",
    "drwtNo4":  "num4",
    "drwtNo5":  "num5",
    "drwtNo6":  "num6",
    "bnusNo":   "bonus",
}

COLUMNS = ["round_no", "date", "num1", "num2", "num3", "num4", "num5", "num6", "bonus"]


# ── 결과 모델 ──────────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    """단일 회차 수집 결과를 담는 값 객체."""

    round_no: int
    success: bool                               # API 호출 성공 여부
    data: dict | None = None                    # 파싱된 회차 데이터
    error: str | None = None                    # 오류 메시지 (실패 시)
    validation_errors: list[str] = field(default_factory=list)  # 검증 오류

    @property
    def is_valid(self) -> bool:
        """API 성공 + 검증 통과 모두 만족하는지 확인한다."""
        return self.success and len(self.validation_errors) == 0


@dataclass
class CollectStats:
    """수집 전체 통계를 담는 값 객체."""

    start_round: int
    end_round: int
    collected: int = 0      # API 성공 + 검증 통과
    skipped: int = 0        # API 오류로 스킵
    invalid: int = 0        # 검증 실패
    elapsed: float = 0.0    # 소요 시간(초)
    failed_rounds: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.end_round - self.start_round + 1

    @property
    def success_rate(self) -> float:
        return self.collected / self.total * 100 if self.total > 0 else 0.0


# ── 세션 팩토리 ────────────────────────────────────────────────────────────

def _build_session(
    max_retries: int = _MAX_RETRIES,
    backoff_factor: float = _BACKOFF_FACTOR,
) -> requests.Session:
    """지수 백오프 재시도 로직이 내장된 requests.Session을 생성한다.

    재시도 조건:
        - 상태 코드: 429, 500, 502, 503, 504
        - 메서드:   GET 전용
        - 대기:     0.5s → 1s → 2s (backoff_factor=0.5)
    """
    session = requests.Session()

    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(_RETRY_STATUS),
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(_HEADERS)
    return session


# ── Collector ──────────────────────────────────────────────────────────────

class DhCollector(BaseCollector):
    """동행복권 REST API를 통해 회차별 당첨번호를 수집한다.

    Args:
        base_url:          동행복권 API 기본 URL
        timeout:           단일 HTTP 요청 타임아웃(초)
        max_retries:       실패 시 재시도 횟수
        rate_limit_sleep:  회차 간 딜레이(초) — API 과부하 방지
    """

    def __init__(
        self,
        base_url: str = _BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
        rate_limit_sleep: float = _RATE_LIMIT_SLEEP,
    ):
        self._base_url = base_url
        self._timeout = timeout
        self._rate_limit_sleep = rate_limit_sleep
        self._session = _build_session(max_retries=max_retries)

    # ── Public API ─────────────────────────────────────────────────────────

    def fetch_latest_round(self) -> int:
        """현재 최신 회차를 이진탐색으로 찾는다.

        동행복권 API는 최신 회차를 직접 제공하지 않으므로
        날짜 기반 추정값을 상한으로 삼아 이진탐색으로 탐색한다.

        알고리즘:
            1. 오늘 날짜와 1회차(2002-12-07) 간격을 주(week) 단위로 환산 → hi 추정
            2. lo=1, hi=추정값+30 범위에서 이진탐색
            3. API returnValue == "success"인 마지막 회차를 반환

        Returns:
            최신 회차 번호 (예: 1150)
        """
        hi = max(100, (date.today() - _FIRST_DRAW_DATE).days // 7 + 30)
        lo, result = 1, 1

        logger.debug("최신 회차 탐색: lo=%d, hi=%d", lo, hi)
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                raw = self._call_api(mid)
                if raw.get("returnValue") == "success":
                    result = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            except Exception as exc:
                logger.debug("탐색 중 오류 (회차 %d): %s", mid, exc)
                hi = mid - 1

        logger.info("최신 회차: %d", result)
        return result

    def fetch_draw(self, round_no: int) -> dict:
        """단일 회차의 당첨번호를 딕셔너리로 반환한다.

        Returns:
            {"round_no": int, "date": str, "num1"~"num6": int, "bonus": int}

        Raises:
            ValueError: API가 실패 응답을 반환한 경우
            requests.HTTPError: HTTP 오류 상태 코드
            requests.Timeout: 타임아웃 초과
        """
        raw = self._call_api(round_no)
        if raw.get("returnValue") != "success":
            raise ValueError(
                f"회차 {round_no} API 응답 실패 "
                f"(returnValue={raw.get('returnValue')!r})"
            )
        return self._parse_row(raw)

    def fetch_range(self, start: int, end: int) -> pd.DataFrame:
        """BaseCollector 호환 인터페이스 — DataFrame만 반환.

        내부적으로 collect_range()를 사용하며, 통계는 무시된다.
        """
        df, _ = self.collect_range(start, end)
        return df

    def collect_range(
        self,
        start: int,
        end: int,
        on_progress: Callable[[FetchResult], None] | None = None,
    ) -> tuple[pd.DataFrame, CollectStats]:
        """start~end 회차를 수집하고 통계를 함께 반환한다.

        각 회차는 다음 순서로 처리된다:
            1. API 호출 (_fetch_one_safe — 예외를 FetchResult로 변환)
            2. 데이터 유효성 검증 (validator.validate_draw)
            3. 성공 시 rows에 추가, 실패 시 통계 카운트
            4. on_progress 콜백 호출 (있을 경우)
            5. rate_limit_sleep 대기

        Args:
            start:       시작 회차 (포함)
            end:         종료 회차 (포함)
            on_progress: 회차 처리 후 호출되는 콜백 (FetchResult 전달)

        Returns:
            (DataFrame, CollectStats)
            DataFrame 컬럼: ["round_no", "date", "num1"~"num6", "bonus"]
        """
        stats = CollectStats(start_round=start, end_round=end)
        t_start = time.monotonic()
        rows: list[dict] = []

        for round_no in range(start, end + 1):
            result = self._fetch_one_safe(round_no)

            if result.success and result.data is not None:
                ok, errs = validate_draw(result.data)
                if ok:
                    rows.append(result.data)
                    stats.collected += 1
                else:
                    result.success = False   # is_valid → False
                    result.validation_errors = errs
                    stats.invalid += 1
                    stats.failed_rounds.append(round_no)
                    logger.warning(
                        "회차 %d 검증 실패: %s", round_no, "; ".join(errs)
                    )
            else:
                stats.skipped += 1
                stats.failed_rounds.append(round_no)

            if on_progress is not None:
                on_progress(result)

            time.sleep(self._rate_limit_sleep)

        stats.elapsed = time.monotonic() - t_start

        if rows:
            df = pd.DataFrame(rows, columns=COLUMNS)
            df["round_no"] = df["round_no"].astype(int)
        else:
            df = pd.DataFrame(columns=COLUMNS)

        logger.info(
            "수집 완료: 성공=%d, 실패(API)=%d, 검증오류=%d, 소요=%.1fs",
            stats.collected, stats.skipped, stats.invalid, stats.elapsed,
        )
        return df, stats

    # ── Private helpers ────────────────────────────────────────────────────

    def _fetch_one_safe(self, round_no: int) -> FetchResult:
        """단일 회차를 수집하고 예외를 FetchResult로 래핑해 반환한다.

        어떤 예외가 발생해도 FetchResult(success=False)를 반환하므로
        호출부에서 try/except 없이 사용할 수 있다.
        """
        try:
            data = self.fetch_draw(round_no)
            return FetchResult(round_no=round_no, success=True, data=data)

        except requests.exceptions.Timeout:
            msg = f"타임아웃 (>{self._timeout}s)"
        except requests.exceptions.ConnectionError as exc:
            msg = f"연결 오류: {exc}"
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            msg = f"HTTP {code} 오류"
        except requests.exceptions.RequestException as exc:
            msg = f"요청 오류: {exc}"
        except ValueError as exc:
            msg = str(exc)
        except Exception as exc:
            msg = f"예기치 않은 오류: {exc}"
            logger.exception("회차 %d 수집 중 예외 발생", round_no)
            return FetchResult(round_no=round_no, success=False, error=msg)

        logger.warning("회차 %d 스킵: %s", round_no, msg)
        return FetchResult(round_no=round_no, success=False, error=msg)

    def _call_api(self, round_no: int) -> dict:
        """동행복권 API를 호출하고 원본 JSON을 반환한다."""
        params = {"method": "getLottoNumber", "drwNo": round_no}
        resp = self._session.get(
            self._base_url,
            params=params,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_row(raw: dict) -> dict:
        """API 응답 JSON을 내부 컬럼명 딕셔너리로 변환한다."""
        return {internal: raw[api_key] for api_key, internal in _API_KEY_MAP.items()}
