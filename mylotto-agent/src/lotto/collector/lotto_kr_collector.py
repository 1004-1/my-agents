"""lotto.co.kr HTML 스크래핑 기반 수집기.

동행복권(dhlottery.co.kr) API가 해외 IP에서 차단되는 경우 사용한다.
lotto.co.kr 의 /lotto_info/list_ajax 엔드포인트를 통해
이미지 파일명에서 당첨번호를 파싱한다.

엔드포인트:
    POST /lotto_info/list_cnt_ajax  → {"count": <최신회차>}
    POST /lotto_info/list_ajax      → 페이지별 당첨번호 HTML
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

from .base import BaseCollector
from .dh_collector import (
    COLUMNS,
    CollectStats,
    FetchResult,
    _BACKOFF_FACTOR,
    _MAX_RETRIES,
    _RATE_LIMIT_SLEEP,
    _RETRY_STATUS,
)
from .validator import validate_draw

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.lotto.co.kr"
_CATEGORY = "AC01"
_PAGE_SIZE = 10

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": f"{_BASE_URL}/article/list/AC01",
    "X-Requested-With": "XMLHttpRequest",
}

# HTML 파싱 패턴
_RE_LI        = re.compile(r"<li>(.*?)</li>",        re.DOTALL)
_RE_ROUND     = re.compile(r"<span>(\d+)회</span>")
_RE_DATE      = re.compile(r"\d+회</span>\s*<span>(\d{4}-\d{2}-\d{2})</span>")
_RE_BALL_ON   = re.compile(r"lottoball_92/on/(\d+)\.png")    # 당첨번호
_RE_BALL_BONUS = re.compile(r"lottoball_92/bonus/(\d+)\.png") # 보너스번호


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=list(_RETRY_STATUS),
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(_HEADERS)
    return session


class LottoKrCollector(BaseCollector):
    """lotto.co.kr HTML을 스크래핑해 회차별 당첨번호를 수집한다.

    동행복권 공식 API와 동일한 BaseCollector 인터페이스를 구현하므로
    DhCollector 와 교체해서 사용할 수 있다.

    수집 방식:
        1. /lotto_info/list_cnt_ajax → 최신 회차(== count) 확인
        2. /lotto_info/list_ajax     → 페이지 단위 HTML 스크래핑
        3. 이미지 파일명(lottoball_92/on/{N}.png)에서 번호 추출

    Args:
        base_url:          lotto.co.kr 기본 URL
        timeout:           단일 HTTP 요청 타임아웃(초)
        rate_limit_sleep:  요청 간 딜레이(초)
    """

    def __init__(
        self,
        base_url: str = _BASE_URL,
        timeout: int = 15,
        rate_limit_sleep: float = _RATE_LIMIT_SLEEP,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._rate_limit_sleep = rate_limit_sleep
        self._session = _build_session()

    # ── Public API ─────────────────────────────────────────────────────────

    def fetch_latest_round(self) -> int:
        """최신 회차를 /lotto_info/list_cnt_ajax 에서 가져온다.

        lotto.co.kr의 count 값은 현재 최신 회차 번호와 동일하다.
        (2002년 12월 7일 1회차부터 매주 1개씩 증가)
        """
        r = self._session.post(
            f"{self._base_url}/lotto_info/list_cnt_ajax",
            data={
                "category": _CATEGORY,
                "code_manage_id": "",
                "search_target": "",
                "search_word": "",
            },
            timeout=self._timeout,
        )
        r.raise_for_status()
        count = r.json().get("count")
        if count is None:
            raise ValueError(f"count 필드를 찾을 수 없습니다: {r.text[:200]}")
        logger.info("최신 회차: %d", count)
        return int(count)

    def fetch_draw(self, round_no: int) -> dict:
        """단일 회차 데이터를 페이지 스크래핑으로 반환한다.

        한 회차씩 낱건으로 가져오는 건 비효율적이므로
        내부적으로 해당 페이지를 통째로 가져온 뒤 필터링한다.
        """
        latest = self.fetch_latest_round()
        page = (latest - round_no) // _PAGE_SIZE + 1
        rows = self._fetch_page(page, latest)
        for row in rows:
            if row["round_no"] == round_no:
                return row
        raise ValueError(f"회차 {round_no} 데이터를 찾을 수 없습니다 (page={page})")

    def fetch_range(self, start: int, end: int) -> pd.DataFrame:
        """BaseCollector 호환 인터페이스 — DataFrame만 반환."""
        df, _ = self.collect_range(start, end)
        return df

    def collect_range(
        self,
        start: int,
        end: int,
        on_progress: Callable[[FetchResult], None] | None = None,
    ) -> tuple[pd.DataFrame, CollectStats]:
        """start~end 회차를 페이지 단위로 수집하고 통계를 반환한다.

        페이지 순서: 최신 회차부터 내림차순으로 페이지가 구성된다.
            - 최신=count:  page 1 → (count, count-1, ..., count-9)
            - page P     → (count-10*(P-1), ..., count-10*P+1)

        Args:
            start:       시작 회차 (포함)
            end:         종료 회차 (포함)
            on_progress: 회차 처리 후 호출 콜백 (FetchResult 전달)

        Returns:
            (DataFrame, CollectStats)
        """
        stats = CollectStats(start_round=start, end_round=end)
        t0 = time.monotonic()

        latest = self.fetch_latest_round()

        # 필요한 페이지 범위 계산
        first_page = (latest - end) // _PAGE_SIZE + 1
        last_page  = (latest - start) // _PAGE_SIZE + 1
        first_page = max(1, first_page)

        rows: list[dict] = []
        needed = set(range(start, end + 1))

        for page in range(first_page, last_page + 1):
            try:
                page_rows = self._fetch_page(page, latest)
                for row in page_rows:
                    rn = row["round_no"]
                    if rn not in needed:
                        continue
                    ok, errs = validate_draw(row)
                    if ok:
                        rows.append(row)
                        needed.discard(rn)
                        stats.collected += 1
                        result = FetchResult(round_no=rn, success=True, data=row)
                    else:
                        stats.invalid += 1
                        stats.failed_rounds.append(rn)
                        result = FetchResult(
                            round_no=rn, success=False,
                            validation_errors=errs,
                        )
                    if on_progress:
                        on_progress(result)
            except Exception as exc:
                logger.warning("페이지 %d 수집 실패: %s", page, exc)
            time.sleep(self._rate_limit_sleep)

        # needed 에 남은 회차 = 수집 실패
        for rn in sorted(needed):
            stats.skipped += 1
            stats.failed_rounds.append(rn)
            if on_progress:
                on_progress(FetchResult(
                    round_no=rn, success=False,
                    error="페이지에서 찾지 못함",
                ))

        stats.elapsed = time.monotonic() - t0

        if rows:
            df = pd.DataFrame(rows, columns=COLUMNS)
            df["round_no"] = df["round_no"].astype(int)
            df = df.sort_values("round_no").reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=COLUMNS)

        logger.info(
            "수집 완료: 성공=%d, 실패=%d, 검증오류=%d, 소요=%.1fs",
            stats.collected, stats.skipped, stats.invalid, stats.elapsed,
        )
        return df, stats

    # ── Private helpers ────────────────────────────────────────────────────

    def _fetch_page(self, page: int, total: int) -> list[dict]:
        """단일 페이지를 요청하고 당첨번호 목록을 파싱해 반환한다."""
        start_pos = (page - 1) * _PAGE_SIZE
        r = self._session.post(
            f"{self._base_url}/lotto_info/list_ajax",
            data={
                "category": _CATEGORY,
                "code_manage_id": "",
                "startPos": start_pos,
                "endPos": start_pos + _PAGE_SIZE,
                "pageSize": _PAGE_SIZE,
                "total": total,
                "page": page,
                "prev_page": "",
                "search_target": "",
                "search_word": "",
            },
            timeout=self._timeout,
        )
        r.raise_for_status()
        return self._parse_list_html(r.text)

    @staticmethod
    def _parse_list_html(html: str) -> list[dict]:
        """list_ajax HTML에서 당첨번호 목록을 추출한다.

        HTML 구조 (핵심만):
            <li>
              <p>
                <span>1225회</span>
                <span>2026-05-23</span>
                <span>
                  <img src=".../lottoball_92/on/8.png" .../>   ← 당첨번호
                  ...  (6개)
                  <img src=".../lottoball_92/bonus/33.png" .../> ← 보너스
                </span>
              </p>
            </li>
        """
        rows: list[dict] = []
        for li_html in _RE_LI.findall(html):
            m_round = _RE_ROUND.search(li_html)
            m_date  = _RE_DATE.search(li_html)
            if not m_round or not m_date:
                continue

            round_no = int(m_round.group(1))
            date     = m_date.group(1)
            nums     = [int(n) for n in _RE_BALL_ON.findall(li_html)]
            bonuses  = [int(n) for n in _RE_BALL_BONUS.findall(li_html)]

            if len(nums) != 6 or len(bonuses) != 1:
                logger.warning(
                    "회차 %d 파싱 이상: nums=%s, bonus=%s", round_no, nums, bonuses
                )
                continue

            rows.append({
                "round_no": round_no,
                "date":     date,
                "num1":     nums[0],
                "num2":     nums[1],
                "num3":     nums[2],
                "num4":     nums[3],
                "num5":     nums[4],
                "num6":     nums[5],
                "bonus":    bonuses[0],
            })
        return rows
