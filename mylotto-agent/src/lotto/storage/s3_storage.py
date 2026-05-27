"""AWS S3 스토리지 — CSV 업로드/다운로드 (skeleton).

TODO:
    - upload_results(): lotto_draw_results.csv → S3 업로드
    - download_results(): S3 → 로컬 동기화
    - upload_games(): generated_games.csv → S3 업로드
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class S3Storage:
    """로컬 CSV를 S3에 백업하거나 S3에서 복원한다.

    사용 전 환경변수 설정 필요:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
        AWS_DEFAULT_REGION, S3_BUCKET_NAME, S3_PREFIX
    """

    def __init__(self, bucket: str, prefix: str = "lotto/"):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self._client = self._build_client()

    # ──────────────────────────────────────────────────────────────────────
    # Public API (stubs)
    # ──────────────────────────────────────────────────────────────────────

    def upload_results(self, local_path: str | Path) -> str:
        """당첨번호 CSV를 S3에 업로드한다.

        Returns:
            업로드된 S3 URI (s3://bucket/prefix/filename)
        """
        raise NotImplementedError("S3 업로드는 아직 구현되지 않았습니다.")

    def download_results(self, local_path: str | Path) -> None:
        """S3에서 당첨번호 CSV를 로컬에 다운로드한다."""
        raise NotImplementedError("S3 다운로드는 아직 구현되지 않았습니다.")

    def upload_games(self, local_path: str | Path) -> str:
        """생성 게임 CSV를 S3에 업로드한다."""
        raise NotImplementedError("S3 업로드는 아직 구현되지 않았습니다.")

    def list_objects(self) -> list[str]:
        """버킷의 prefix 아래 파일 목록을 반환한다."""
        raise NotImplementedError("S3 목록 조회는 아직 구현되지 않았습니다.")

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_client():
        """boto3 S3 클라이언트를 생성한다."""
        try:
            import boto3  # noqa: PLC0415

            return boto3.client("s3")
        except ImportError:
            logger.warning("boto3가 설치되어 있지 않습니다. pip install boto3")
            return None
