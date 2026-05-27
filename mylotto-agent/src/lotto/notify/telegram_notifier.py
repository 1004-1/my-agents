"""Telegram Bot 알림 (skeleton).

TODO:
    - send_games(): 생성된 게임 번호를 텔레그램으로 전송
    - send_results(): 최신 당첨번호를 텔레그램으로 전송
    - send_text(): 임의 텍스트 메시지 전송

환경변수:
    TELEGRAM_BOT_TOKEN: BotFather에서 발급받은 토큰
    TELEGRAM_CHAT_ID: 메시지를 받을 채팅/채널 ID
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Telegram Bot API를 통해 메시지를 전송한다."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    # ──────────────────────────────────────────────────────────────────────
    # Public API (stubs)
    # ──────────────────────────────────────────────────────────────────────

    def send_text(self, text: str, parse_mode: str = "Markdown") -> bool:
        """텍스트 메시지를 Telegram으로 전송한다.

        Returns:
            성공 여부
        """
        raise NotImplementedError("Telegram 전송은 아직 구현되지 않았습니다.")

    def send_games(self, games: list[list[int]], strategy: str, round_no: int | None = None) -> bool:
        """생성된 게임 번호를 포맷팅해서 전송한다.

        Args:
            games:      [[num, ...], ...]
            strategy:   전략 이름 (random / balanced)
            round_no:   다음 회차 번호 (optional)

        Returns:
            성공 여부
        """
        raise NotImplementedError("Telegram 게임 전송은 아직 구현되지 않았습니다.")

    def send_latest_result(self, round_no: int, numbers: list[int], bonus: int) -> bool:
        """최신 당첨번호를 전송한다.

        Args:
            round_no:   회차 번호
            numbers:    당첨번호 6개
            bonus:      보너스 번호

        Returns:
            성공 여부
        """
        raise NotImplementedError("Telegram 당첨번호 전송은 아직 구현되지 않았습니다.")

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _check_config(self) -> None:
        """필수 설정이 있는지 확인한다."""
        if not self.bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN 환경변수가 설정되지 않았습니다.")
        if not self.chat_id:
            raise RuntimeError("TELEGRAM_CHAT_ID 환경변수가 설정되지 않았습니다.")

    @staticmethod
    def _format_game(game_no: int, numbers: list[int]) -> str:
        """게임 한 줄을 사람이 읽기 좋은 형태로 포맷한다."""
        balls = " ".join(f"*{n:02d}*" for n in sorted(numbers))
        return f"게임 {game_no}: {balls}"
