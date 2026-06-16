"""동행복권 구매 보조 자동화 — Playwright 기반.

목적: generated_games.csv의 번호를 자동 입력하고 구매 버튼 직전에서 대기한다.
     구매 최종 확인 버튼은 절대 자동 클릭하지 않는다.

세션: data/browser-profile  (persistent context — 로그인 쿠키 유지)
스크린샷: screenshots/

주의:
    playwright install chromium  이 필요합니다.
    과도한 자동 구매는 동행복권 이용 약관에 위반될 수 있습니다.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

_LOGIN_URL = "https://www.dhlottery.co.kr/login"
_BUY_URL   = "https://ol.dhlottery.co.kr/olotto/game/game645.do"

# 로그아웃 버튼 선택자 — 하나라도 보이면 로그인 상태로 판정
_LOGOUT_SELECTORS = ", ".join([
    "a[href*='logout']",
    ".btn_logout",
    "#idBtn_logout",
    "a[onclick*='logout']",
    "a[title='로그아웃']",
    "a:text('로그아웃')",
    ".member_wrap",
    ".user_info",
])

# 로그인 폼 선택자
_ID_SELECTORS = [
    "#userId",
    "input[name='userId']",
    "input[placeholder*='아이디']",
    "input[type='text']",
]
_PW_SELECTORS = [
    "#userPw",
    "input[name='userPw']",
    "input[placeholder*='비밀번호']",
    "input[type='password']",
]
_LOGIN_BTN_SELECTORS = [
    "#btnLogin",
    "button[type='submit']",
    "button:text('로그인')",
    "input[type='submit']",
    ".btn_login",
    "a:text('로그인')",
]

# 수동 탭 선택자 (우선순위 순)
# game645 페이지: a#num1 = "혼합선택" (수동 번호 입력 모드)
_MANUAL_TAB_SELECTORS = [
    "#num1",
    "a#num1",
    "#mType2",
    "label[for='mType2']",
    "a:text('수동')",
    "a:text('혼합선택')",
    "li.manual_tab a",
    "input[value='수동']",
]

# 번호 초기화 버튼 선택자
_CLEAR_SELECTORS = [
    "#btnNumClear",
    ".btn_number_clear",
    "a:text('초기화')",
    "button:text('초기화')",
    ".btn-clear",
]

# 추첨번호 추가(카트 담기) 버튼 선택자
# ⚠ #btnBuy 는 "구매하기" (최종 구매) — 절대 포함 금지
# game645 페이지: 적용수량 옆 "확인" 버튼이 게임 추가 버튼
_ADD_SELECTORS = [
    "a:text-is('확인')",
    "input[value='확인']",
    "button:text-is('확인')",
    "a:text('추첨번호추가')",
    "button:text('추첨번호추가')",
    "a:text('번호추가')",
    "#addBtn",
    ".btn_add",
]


class LottoBuyer:
    """동행복권 번호 자동 입력 보조기.

    사용 흐름:
        buyer = LottoBuyer()
        games = buyer.load_latest_games()
        await buyer.open_browser()
        await buyer.wait_for_manual_login()   # 사용자 직접 로그인
        screenshot = await buyer.run(games)
        await buyer.close()                   # 엔터 후 닫기
    """

    def __init__(
        self,
        games_csv:      Path = Path("data/generated_games.csv"),
        profile_dir:    Path = Path("data/browser-profile"),
        screenshot_dir: Path = Path("screenshots"),
        max_games:      int  = 5,
        headless:       bool = False,
        login_id:       str | None = None,
        login_pw:       str | None = None,
    ):
        self.games_csv      = Path(games_csv)
        self.profile_dir    = Path(profile_dir)
        self.screenshot_dir = Path(screenshot_dir)
        self.max_games      = max(1, min(max_games, 5))  # 동행복권 1회 최대 5게임
        self.headless       = headless
        self.login_id       = login_id or os.getenv("DH_LOGIN_ID", "")
        self.login_pw       = login_pw or os.getenv("DH_LOGIN_PW", "")
        self._pw            = None
        self._context       = None
        self._page          = None
        self._game_frame    = None  # 구매 페이지 iframe (없으면 _page 와 동일)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def load_latest_games(self) -> list[list[int]]:
        """generated_games.csv에서 가장 최근 생성 배치의 미구매 게임을 읽는다.

        - 이미 purchased=True 인 게임은 제외한다.
        - 최신 배치가 모두 구매됐으면 ValueError를 발생시킨다.

        Returns:
            [[n1, n2, n3, n4, n5, n6], ...]  오름차순 정렬, 최대 max_games 개
        """
        if not self.games_csv.exists():
            raise FileNotFoundError(
                f"게임 파일이 없습니다: {self.games_csv}\n"
                "  먼저 `python main.py run` 을 실행하세요."
            )

        from ..storage.local_storage import LocalStorage
        storage = LocalStorage(
            results_path=self.games_csv.parent / "lotto_draw_results.csv",
            games_path=self.games_csv,
        )
        df = storage.load_games()  # dtype 보정 포함
        if df.empty:
            raise ValueError("게임 파일이 비어 있습니다.")

        latest_ts = df["generated_at"].max()
        latest_df = df[df["generated_at"] == latest_ts]

        # 이미 구매 완료된 게임 제외
        unpurchased = latest_df[~latest_df["purchased"].astype(bool)]
        if unpurchased.empty:
            date_str = str(latest_ts)[:16]
            raise ValueError(
                f"최신 배치({date_str})의 모든 게임이 이미 구매됐습니다.\n"
                "  `python main.py run` 으로 새 번호를 생성하세요."
            )

        games: list[list[int]] = []
        for _, row in unpurchased.head(self.max_games).iterrows():
            nums = sorted(int(row[f"num{i}"]) for i in range(1, 7))
            games.append(nums)
        return games

    async def open_browser(self) -> None:
        """Chromium을 persistent context로 실행하고 로그인 페이지로 이동한다.

        data/browser-profile 에 쿠키·세션이 유지되어 재로그인을 최소화한다.
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "playwright가 설치되지 않았습니다.\n"
                "  pip install playwright && playwright install chromium"
            )

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._pw = await async_playwright().start()
        self._context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            viewport={"width": 1280, "height": 900},
            locale="ko-KR",
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        pages = self._context.pages
        self._page = pages[0] if pages else await self._context.new_page()
        # 메인 페이지로 이동해서 세션 유효 여부를 확인
        await self._page.goto("https://www.dhlottery.co.kr", wait_until="domcontentloaded")
        await self._page.wait_for_timeout(1500)

    async def login(self, timeout_seconds: int = 180) -> None:
        """로그인을 처리한다.

        1) 기존 세션이 유효하면 즉시 통과.
        2) 환경변수 DH_LOGIN_ID / DH_LOGIN_PW 가 설정돼 있으면 자동 로그인 시도.
        3) 자동 로그인 실패 시 사용자가 직접 로그인할 때까지 대기.
        """
        if await self._is_logged_in():
            console.print("[green]✓ 기존 세션 유효 — 로그인 건너뜀[/green]")
            return

        if self.login_id and self.login_pw:
            console.print("[cyan]⚙  자동 로그인 시도 중...[/cyan]")
            if await self._do_auto_login():
                console.print("[green]✓ 자동 로그인 완료[/green]")
                return
            console.print("[yellow]⚠ 자동 로그인 실패 — 수동 로그인으로 전환[/yellow]")

        # 수동 로그인 대기
        console.print(
            f"\n[bold yellow]⏳  브라우저에서 동행복권에 직접 로그인하세요.[/bold yellow]\n"
            f"   최대 [bold]{timeout_seconds}[/bold]초 대기 중..."
        )
        try:
            await self._page.wait_for_url(
                lambda url: "/login" not in url,
                timeout=timeout_seconds * 1000,
            )
        except Exception:
            raise TimeoutError(
                f"{timeout_seconds}초 내에 로그인이 완료되지 않았습니다."
            )
        console.print("[green]✓ 로그인 완료[/green]")

    # 하위 호환 별칭
    async def wait_for_manual_login(self, timeout_seconds: int = 180) -> None:
        await self.login(timeout_seconds)

    async def navigate_to_buy_page(self) -> None:
        """팝업 닫기 → 추첨식복권 바로구매 클릭 → 구매 페이지 이동."""
        page = self._page

        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(1500)

        # 로그인 직후 화면 캡처 (팝업 구조 확인용)
        dbg2 = self.screenshot_dir / "debug_after_login.png"
        await page.screenshot(path=str(dbg2))
        console.print(f"  [dim]로그인 후 캡처: {dbg2}[/dim]")

        # 팝업 HTML 구조 출력
        popup_info = await page.evaluate("""
            () => {
                const candidates = Array.from(document.querySelectorAll(
                    '[class*=popup],[class*=modal],[class*=layer],[class*=dim]'
                )).filter(el => el.offsetParent !== null)
                  .map(el => ({tag: el.tagName, id: el.id, cls: el.className.slice(0,80)}));
                return candidates.slice(0, 10);
            }
        """)
        console.print(f"  [dim]팝업 후보: {popup_info}[/dim]")

        # 팝업 닫기 (최대 3개)
        await self._close_popups()

        # "바로구매" 버튼 클릭 — "추첨식복권" 영역 우선, 전체 페이지 폴백
        _BUY_NOW_SELECTORS = [
            "a:text('바로구매')",
            "button:text('바로구매')",
            "a:has-text('바로구매')",
        ]
        clicked = False
        for sel in _BUY_NOW_SELECTORS:
            try:
                # 추첨식복권 섹션 내 바로구매 우선
                in_section = page.locator(f"*:has-text('추첨식복권') {sel}")
                if await in_section.count() > 0:
                    await in_section.first.click()
                    console.print("  [dim]추첨식복권 > 바로구매 클릭[/dim]")
                    clicked = True
                    break
                # 전체 페이지 폴백
                el = page.locator(sel)
                if await el.count() > 0:
                    await el.first.click()
                    console.print(f"  [dim]바로구매 클릭 ({sel})[/dim]")
                    clicked = True
                    break
            except Exception as e:
                logger.debug("바로구매 sel %s 실패: %s", sel, e)

        if not clicked:
            console.print("  [yellow]'바로구매' 버튼 미발견 → 직접 URL 이동[/yellow]")
            await page.goto(_BUY_URL, wait_until="domcontentloaded")

        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(2500)

        # 구매 페이지 캡처
        dbg_buy = self.screenshot_dir / "debug_buy_page.png"
        await page.screenshot(path=str(dbg_buy))
        console.print(f"  [dim]구매 페이지 캡처: {dbg_buy}[/dim]")

        # 번호 피커가 들어 있는 iframe 프레임 감지 (없으면 메인 페이지 사용)
        self._game_frame = await self._find_game_frame()

    async def _diagnose_page(self) -> None:
        """구매 페이지 DOM 구조를 진단해 콘솔에 출력한다."""
        frame = self._game_frame or self._page
        info = await frame.evaluate("""
            () => {
                // 1. check645num 체크박스 총 개수 및 샘플
                const cbs = Array.from(document.querySelectorAll("input[name='check645num']"));
                const cbSample = cbs.slice(0, 3).map(el => {
                    const lbl = document.querySelector("label[for='" + el.id + "']");
                    return {
                        id: el.id, val: el.value, chk: el.checked,
                        lblTxt: lbl ? lbl.textContent.trim().slice(0,10) : null,
                        lblCls: lbl ? lbl.className.slice(0,40) : null
                    };
                });

                // 2. 모든 버튼/링크 텍스트 (클릭 가능 요소 파악)
                const btns = Array.from(document.querySelectorAll('a,button'))
                    .filter(el => el.offsetParent !== null && el.textContent.trim().length > 0)
                    .slice(0, 15)
                    .map(el => ({tag: el.tagName, id: el.id, cls: el.className.slice(0,30),
                                 txt: el.textContent.trim().slice(0,20)}));

                return {cbTotal: cbs.length, cbSample, btns};
            }
        """)
        console.print(f"  [dim][진단] check645num 총 {info['cbTotal']}개, 샘플: {info['cbSample']}[/dim]")
        console.print(f"  [dim][진단] 클릭 가능 버튼/링크: {info['btns']}[/dim]")

    async def fill_numbers(self, games: list[list[int]]) -> int:
        """구매 페이지에서 각 게임 번호를 클릭하고 카트에 추가한다.

        Returns:
            카트에 추가 성공한 게임 수
        """
        await self._diagnose_page()
        added_count = 0

        for idx, numbers in enumerate(games, 1):
            console.print(f"  [cyan]게임 {idx}[/cyan]: {numbers}")

            # 혼합선택 모드 탭 클릭 (a#num1)
            await self._click_manual_tab()
            await self._page.wait_for_timeout(500)

            # 기존 선택 초기화
            await self._clear_selection()
            await self._page.wait_for_timeout(300)

            hit = 0
            for num in numbers:
                if await self._click_number(num):
                    hit += 1
                else:
                    console.print(f"    [yellow]⚠ 번호 {num} 클릭 실패[/yellow]")
                await self._page.wait_for_timeout(180)

            # JS로 실제 체크된 번호 수 확인
            checked = await self._count_checked_numbers()
            console.print(f"    [dim]체크 확인: {checked}/6[/dim]")

            # 경고 팝업 닫기 (최대 6개 초과 등)
            frame = self._game_frame or self._page
            try:
                warn = frame.locator(".ui-dialog button, .msgBox button, .alertPopup button")
                if await warn.count() > 0 and await warn.first.is_visible(timeout=500):
                    await warn.first.click()
                    await self._page.wait_for_timeout(400)
                    console.print("    [dim]경고 팝업 닫음[/dim]")
            except Exception:
                pass

            if hit == 6:
                # "확인" 클릭 전 스크린샷
                snap = self.screenshot_dir / f"game{idx}_before_add.png"
                await self._page.screenshot(path=str(snap))

                added, which_sel = await self._click_add_button()
                await self._page.wait_for_timeout(800)

                # "확인" 클릭 후 스크린샷
                snap2 = self.screenshot_dir / f"game{idx}_after_add.png"
                await self._page.screenshot(path=str(snap2))

                if added:
                    status = f"[green]✓ 카트 추가[/green] [dim]({which_sel})[/dim]"
                    added_count += 1
                else:
                    status = "[yellow]⚠ '확인' 버튼 미발견 — 카트 추가 실패[/yellow]"
            else:
                status = f"[red]✗ {hit}/6개 입력 — 카트 추가 건너뜀[/red]"
            console.print(f"    {status}")
            await self._page.wait_for_timeout(500)

        return added_count

    async def save_screenshot(self, label: str = "numbers_filled") -> Path:
        """현재 화면을 screenshots/<timestamp>_<label>.png 에 저장한다."""
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.screenshot_dir / f"{ts}_{label}.png"
        await self._page.screenshot(path=str(path), full_page=False)
        logger.info("스크린샷 저장: %s", path)
        return path

    async def purchase_games(self) -> bool:
        """구매하기 버튼을 클릭하고 확인 팝업까지 처리한다.

        ⚠ 이 메서드 호출 시 실제 금전 거래가 완료됩니다.

        동행복권은 window.confirm() 네이티브 dialog 또는 DOM 팝업 두 가지를 사용한다.
        Playwright 기본 동작은 confirm()을 false(취소)로 자동 처리하므로
        반드시 핸들러를 등록해 accept() 해야 구매가 완료된다.

        Returns:
            True  — 구매 완료
            False — 구매하기 버튼 미발견 / 비활성화 / 오류
        """
        frame = self._game_frame or self._page
        page  = self._page

        # 구매 전 스크린샷
        await self.save_screenshot("before_purchase")

        # 구매하기 버튼 찾기
        buy_btn = frame.locator("button#btnBuy")
        if await buy_btn.count() == 0:
            buy_btn = page.locator("button#btnBuy, button:text('구매하기')")
        if await buy_btn.count() == 0:
            console.print("[red]✗ 구매하기 버튼 미발견[/red]")
            return False

        # 버튼 활성화 여부 확인 (카트 비어있으면 disabled)
        disabled = await buy_btn.first.get_attribute("disabled")
        cls      = await buy_btn.first.get_attribute("class") or ""
        if disabled is not None or "disabled" in cls:
            console.print("[red]✗ 구매하기 버튼 비활성화 — 카트가 비어 있습니다[/red]")
            return False

        # ── 네이티브 dialog 핸들러 등록 ──────────────────────────────────────
        # Playwright 기본값: confirm() → false (취소). accept()로 재정의해야 구매 진행.
        _dialog_accepted: list[str] = []

        async def _on_dialog(dialog) -> None:
            msg = dialog.message
            _dialog_accepted.append(msg)
            console.print(f"  [dim]브라우저 dialog 수락: {msg[:80]}[/dim]")
            await dialog.accept()

        page.on("dialog", _on_dialog)

        try:
            await buy_btn.first.click()
            console.print("  [dim]구매하기 버튼 클릭[/dim]")
            await page.wait_for_timeout(2000)

            # 팝업 상태 캡처
            await self.save_screenshot("buy_confirm_popup")

            confirmed_via_dialog = bool(_dialog_accepted)

            # DOM 기반 팝업 처리 (네이티브 dialog가 없었던 경우)
            if not confirmed_via_dialog:
                confirmed_via_dom = await self._handle_purchase_popup()
            else:
                confirmed_via_dom = False

            await page.wait_for_timeout(2500)

            # 구매 완료 후 캡처
            await self.save_screenshot("after_purchase")

            # 성공 판정: dialog 처리 OR DOM 팝업 처리 OR 페이지 성공 감지
            success_on_page = await self._detect_purchase_success()
            if success_on_page:
                console.print("[green]✓ 구매 완료 확인 (페이지 메시지 감지)[/green]")
                return True

            if confirmed_via_dialog:
                console.print("[green]✓ 구매 완료 (브라우저 dialog 수락)[/green]")
                return True

            if confirmed_via_dom:
                console.print("[green]✓ 구매 완료 (DOM 팝업 처리)[/green]")
                return True

            console.print("[yellow]⚠ 구매 완료 여부 불명확 — 스크린샷을 확인하세요[/yellow]")
            return False

        finally:
            page.remove_listener("dialog", _on_dialog)

    async def _handle_purchase_popup(self) -> bool:
        """DOM 기반 구매 확인 팝업의 '확인' 버튼을 클릭한다.

        '구매하시겠습니까?' 텍스트를 기준으로 JS로 직접 탐색한다.
        (사이트가 .ui-dialog 등 표준 클래스를 쓰지 않으므로 CSS 셀렉터 방식은 신뢰 불가)
        """
        page  = self._page
        frame = self._game_frame or self._page

        # ── 방법 1: JS — "구매하시겠습니까" 텍스트 기준으로 확인 버튼 찾기 ──
        for ctx_name, ctx in [("frame", frame), ("page", page)]:
            try:
                result = await ctx.evaluate("""() => {
                    const keyword = '구매하시겠습니까';
                    // "구매하시겠습니까" 를 포함하면서 button 자식을 가진 DOM 요소 중 가장 작은 것
                    const candidates = Array.from(document.querySelectorAll('*'))
                        .filter(el =>
                            el.textContent.includes(keyword) &&
                            el.querySelectorAll('button').length >= 1
                        )
                        .sort((a, b) => a.textContent.length - b.textContent.length);
                    if (!candidates.length) return 'no_popup';
                    const popup = candidates[0];
                    const confirmBtn = Array.from(popup.querySelectorAll('button'))
                        .find(b => (b.textContent.trim() === '확인' || b.innerText.trim() === '확인'));
                    if (confirmBtn) {
                        confirmBtn.click();
                        return 'clicked:' + popup.className;
                    }
                    return 'no_button';
                }""")
                console.print(f"  [dim]JS 팝업 탐색 ({ctx_name}): {result}[/dim]")
                if isinstance(result, str) and result.startswith("clicked"):
                    return True
            except Exception as e:
                console.print(f"  [dim]JS 팝업 탐색 오류 ({ctx_name}): {e}[/dim]")

        # ── 방법 2: Playwright 셀렉터 (클래스 무관, 텍스트 기반) ──────────────
        broad_sels = [
            "button:text-is('확인')",
            "input[value='확인']",
            "button:text-is('예')",
        ]
        for ctx_name, ctx in [("frame", frame), ("page", page)]:
            for sel in broad_sels:
                try:
                    els = ctx.locator(sel)
                    cnt = await els.count()
                    # 여러 "확인" 버튼 중 마지막(팝업 버튼이 DOM 후위에 있을 가능성)
                    for i in range(cnt - 1, -1, -1):
                        el = els.nth(i)
                        if await el.is_visible(timeout=500):
                            await el.click()
                            console.print(f"  [dim]팝업 버튼 클릭 ({ctx_name}: {sel} #{i})[/dim]")
                            return True
                except Exception:
                    continue

        console.print("  [dim]구매 확인 팝업 처리 실패[/dim]")
        return False

    async def _detect_purchase_success(self) -> bool:
        """구매 완료 메시지/페이지를 감지한다."""
        page  = self._page
        frame = self._game_frame or self._page

        # 동행복권 구매 완료 후 표시 가능한 텍스트
        success_texts = [
            "text=구매가 완료",
            "text=구매 완료",
            "text=구매완료",
            "text=정상적으로 구매",
            "text=복권을 구매하였습니다",
            "text=구매하였습니다",
            ".buy_complete",
            "#buyComplete",
        ]
        for ctx in [page, frame]:
            for sel in success_texts:
                try:
                    if await ctx.locator(sel).count() > 0:
                        return True
                except Exception:
                    continue
        return False

    def mark_purchased(self, games: list[list[int]]) -> None:
        """구매 완료된 게임을 generated_games.csv에 purchased=True 로 표시한다."""
        from ..storage.local_storage import LocalStorage
        storage = LocalStorage(
            results_path=self.games_csv.parent / "lotto_draw_results.csv",
            games_path=self.games_csv,
        )
        storage.mark_purchased_games(games)
        console.print(f"  [green]✓ {len(games)}게임 구매 기록 저장[/green]")

    async def run(self, games: list[list[int]]) -> tuple[Path, bool]:
        """로그인 → 구매 페이지 이동 → 번호 입력 → 구매 → 스크린샷 순서로 실행한다.

        open_browser() 호출 이후에 실행한다.

        Returns:
            (screenshot_path, purchase_success)
        """
        await self.login()
        await self.navigate_to_buy_page()
        added = await self.fill_numbers(games)
        if added == 0:
            console.print("[red]✗ 카트에 추가된 게임 없음 — 구매 중단[/red]")
            return await self.save_screenshot("cart_empty"), False
        purchased = await self.purchase_games()
        if purchased:
            self.mark_purchased(games)
        label = "purchase_complete" if purchased else "numbers_filled"
        return await self.save_screenshot(label), purchased

    async def close(self) -> None:
        """브라우저와 Playwright 인스턴스를 닫는다."""
        if self._context:
            await self._context.close()
        if self._pw:
            await self._pw.stop()

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    async def _do_auto_login(self) -> bool:
        """ID/PW로 로그인 폼을 채우고 제출한다. 성공 여부를 반환한다."""
        page = self._page

        if _LOGIN_URL not in page.url:
            await page.goto(_LOGIN_URL, wait_until="domcontentloaded")

        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(1500)

        # 디버그 스크린샷
        dbg_path = self.screenshot_dir / "debug_login.png"
        await page.screenshot(path=str(dbg_path))
        console.print(f"  [dim]로그인 페이지 캡처: {dbg_path}[/dim]")

        # ── ID 입력: #inpUserId ────────────────────────────────────────────
        try:
            id_el = page.locator("#inpUserId")
            await id_el.click()
            await page.wait_for_timeout(200)
            await id_el.fill("")
            await id_el.type(self.login_id)   # 실제 키 입력 — 사이트 이벤트 핸들러 동작
            console.print("  [dim]ID 입력 완료 (#inpUserId)[/dim]")
        except Exception as e:
            console.print(f"  [red]ID 입력 실패: {e}[/red]")
            return False

        await page.wait_for_timeout(300)

        # ── PW 입력: #inpUserPswdEncn ─────────────────────────────────────
        # 비밀번호는 클라이언트 암호화 후 숨김 필드(userPswdEncn)에 저장됨
        # → type() 으로 실제 키 입력을 보내야 암호화 로직이 작동함
        try:
            pw_el = page.locator("#inpUserPswdEncn")
            await pw_el.click()
            await page.wait_for_timeout(200)
            await pw_el.fill("")
            await pw_el.type(self.login_pw)
            console.print("  [dim]PW 입력 완료 (#inpUserPswdEncn)[/dim]")
        except Exception as e:
            console.print(f"  [red]PW 입력 실패: {e}[/red]")
            return False

        await page.wait_for_timeout(300)

        # ── 로그인 버튼 클릭 ───────────────────────────────────────────────
        try:
            await page.locator("#btnLogin").click()
            console.print("  [dim]버튼 클릭 (#btnLogin)[/dim]")
        except Exception:
            await page.keyboard.press("Enter")
            console.print("  [dim]Enter 키 제출[/dim]")

        # 로그인 성공 감지: 로그인 페이지에서 벗어나면 성공
        try:
            await page.wait_for_url(
                lambda url: "/login" not in url,
                timeout=12_000,
            )
            return True
        except Exception:
            return False

    async def _is_logged_in(self) -> bool:
        """현재 페이지가 로그인된 상태인지 확인한다."""
        # 로그인 페이지에 있으면 무조건 미로그인
        if "/login" in self._page.url:
            return False
        # 메인/기타 페이지 → 로그아웃 링크·사용자 정보 영역 확인
        for sel in _LOGOUT_SELECTORS.split(", "):
            try:
                if await self._page.locator(sel.strip()).count() > 0:
                    return True
            except Exception:
                continue
        return False

    async def _close_popups(self, max_count: int = 3) -> None:
        """화면에 뜬 팝업/경고 대화상자의 닫기·확인 버튼을 클릭한다."""
        page = self._page
        _CLOSE_SELECTORS = [
            "button:text('닫기')",
            "a:text('닫기')",
            ".btn_close",
            ".popup_close",
            ".layer_close",
            "[aria-label='닫기']",
            "button.close",
            ".modal_close",
        ]
        # 경고 대화상자용: "최대 6개의 숫자를 선택할 수 있습니다." 등
        _DIALOG_SELECTORS = [
            ".ui-dialog button:text('확인')",
            ".ui-dialog button",
            ".alert button",
            ".msgBox button",
            ".layerWrap button:text('확인')",
        ]
        closed = 0
        for _ in range(max_count):
            found = False
            for sel in _CLOSE_SELECTORS + _DIALOG_SELECTORS:
                try:
                    el = page.locator(sel)
                    if await el.count() > 0:
                        await el.first.click(timeout=2000)
                        await page.wait_for_timeout(400)
                        closed += 1
                        found = True
                        break
                except Exception:
                    continue
            if not found:
                break
        if closed:
            console.print(f"  [dim]팝업 {closed}개 닫음[/dim]")

    async def _find_game_frame(self):
        """번호 피커(span.ball)가 있는 iframe 프레임을 찾아 반환한다.

        찾지 못하면 메인 페이지를 반환한다.
        """
        page = self._page
        await page.wait_for_timeout(1000)

        frames = page.frames
        frame_urls = [f.url for f in frames]
        console.print(f"  [dim]프레임 목록({len(frames)}개): {frame_urls}[/dim]")

        # span.ball 요소가 있는 프레임 우선
        for frame in frames:
            try:
                cnt = await frame.locator("span[class*='ball']").count()
                if cnt > 0:
                    console.print(f"  [dim]게임 프레임 발견: {frame.url} (ball×{cnt})[/dim]")
                    return frame
            except Exception:
                continue

        # URL 패턴으로 fallback
        for frame in frames:
            url = frame.url or ""
            if frame is not page.main_frame and ("olotto" in url or "game" in url.lower()):
                console.print(f"  [dim]URL 기반 프레임 사용: {url}[/dim]")
                return frame

        console.print("  [yellow]게임 iframe 미발견 — 메인 프레임 사용[/yellow]")
        return page

    async def _click_manual_tab(self) -> None:
        frame = self._game_frame or self._page
        for sel in _MANUAL_TAB_SELECTORS:
            try:
                el = frame.locator(sel)
                if await el.count() > 0:
                    await el.first.click()
                    await self._page.wait_for_timeout(400)
                    return
            except Exception:
                continue

    async def _clear_selection(self) -> None:
        frame = self._game_frame or self._page
        for sel in _CLEAR_SELECTORS:
            try:
                el = frame.locator(sel)
                if await el.count() > 0:
                    await el.first.click()
                    await self._page.wait_for_timeout(300)
                    return
            except Exception:
                continue

    async def _click_number(self, num: int) -> bool:
        """번호를 클릭하고 성공 여부를 반환한다.

        실제 페이지 구조: input#check645num{N} + label[for='check645num{N}']
        """
        frame = self._game_frame or self._page

        # 1순위: label 클릭 (체크박스가 CSS로 숨겨진 경우 레이블을 클릭해야 함)
        label_sel = f"label[for='check645num{num}']"
        try:
            el = frame.locator(label_sel)
            if await el.count() > 0:
                await el.first.click(timeout=2000)
                return True
        except Exception:
            pass

        # 2순위: input 직접 클릭 (force=True — CSS hidden 무시)
        try:
            el = frame.locator(f"#check645num{num}")
            if await el.count() > 0:
                await el.first.click(force=True, timeout=2000)
                return True
        except Exception:
            pass

        # 3순위: JS 직접 클릭 (Playwright 가시성 제한 완전 우회)
        try:
            clicked = await frame.evaluate(f"""() => {{
                const cb = document.getElementById('check645num{num}');
                if (cb) {{ cb.click(); return true; }}
                const lbl = document.querySelector("label[for='check645num{num}']");
                if (lbl) {{ lbl.click(); return true; }}
                return false;
            }}""")
            if clicked:
                return True
        except Exception:
            pass

        # 4순위: 기존 span.ball 계열 선택자 (폴백)
        for sel in [
            f"span[class*='ball']:text-is('{num}')",
            f"#btnNo{str(num).zfill(2)}",
            f"[data-num='{num}']",
        ]:
            try:
                el = frame.locator(sel)
                cnt = await el.count()
                if cnt == 0:
                    continue
                for i in range(min(cnt, 3)):
                    item = el.nth(i)
                    try:
                        if await item.is_visible():
                            await item.click()
                            return True
                    except Exception:
                        continue
            except Exception:
                continue

        return False

    async def _click_add_button(self) -> tuple[bool, str]:
        """추첨번호추가(카트 담기) 버튼을 클릭한다.

        Returns:
            (성공 여부, 매칭된 셀렉터 문자열)
        """
        frame = self._game_frame or self._page
        for sel in _ADD_SELECTORS:
            try:
                el = frame.locator(sel)
                cnt = await el.count()
                if cnt == 0:
                    continue
                visible = await el.first.is_visible(timeout=500)
                if not visible:
                    continue
                await el.first.click()
                await self._page.wait_for_timeout(600)
                return True, sel
            except Exception:
                continue
        return False, ""

    async def _count_checked_numbers(self) -> int:
        """현재 체크된 check645num 체크박스 수를 JS로 확인한다."""
        frame = self._game_frame or self._page
        try:
            return await frame.evaluate("""() =>
                Array.from(document.querySelectorAll("input[name='check645num']"))
                     .filter(el => el.checked).length
            """)
        except Exception:
            return -1
