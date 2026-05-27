"""AWS EC2 원격 실행 (skeleton).

TODO:
    - start_instance(): EC2 인스턴스 시작
    - stop_instance(): EC2 인스턴스 중지
    - run_agent_on_ec2(): SSM SendCommand로 lotto agent 실행
    - get_instance_status(): 인스턴스 상태 확인

환경변수:
    EC2_INSTANCE_ID: 실행할 EC2 인스턴스 ID
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class Ec2Runner:
    """EC2 인스턴스에서 lotto agent를 원격 실행한다."""

    def __init__(self, instance_id: str | None = None, region: str | None = None):
        self.instance_id = instance_id or os.getenv("EC2_INSTANCE_ID", "")
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
        self._ec2 = None
        self._ssm = None

    # ──────────────────────────────────────────────────────────────────────
    # Public API (stubs)
    # ──────────────────────────────────────────────────────────────────────

    def start_instance(self) -> None:
        """EC2 인스턴스를 시작한다."""
        raise NotImplementedError("EC2 시작은 아직 구현되지 않았습니다.")

    def stop_instance(self) -> None:
        """EC2 인스턴스를 중지(stop)한다."""
        raise NotImplementedError("EC2 중지는 아직 구현되지 않았습니다.")

    def get_instance_status(self) -> str:
        """인스턴스 상태(running/stopped 등)를 반환한다."""
        raise NotImplementedError("EC2 상태 조회는 아직 구현되지 않았습니다.")

    def run_agent_on_ec2(self, command: str) -> str:
        """SSM SendCommand로 EC2에서 lotto agent 커맨드를 실행한다.

        Args:
            command: 실행할 shell 커맨드
                     예: "cd /home/ec2-user/mylotto-agent && python main.py run"

        Returns:
            SSM Command ID
        """
        raise NotImplementedError("EC2 원격 실행은 아직 구현되지 않았습니다.")

    def wait_for_command(self, command_id: str, timeout: int = 300) -> dict:
        """SSM 커맨드 실행이 완료될 때까지 폴링한다.

        Args:
            command_id: run_agent_on_ec2에서 반환된 ID
            timeout:    최대 대기 시간(초)

        Returns:
            {"status": str, "stdout": str, "stderr": str}
        """
        raise NotImplementedError("EC2 커맨드 대기는 아직 구현되지 않았습니다.")

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _init_clients(self) -> None:
        """boto3 클라이언트를 초기화한다."""
        try:
            import boto3  # noqa: PLC0415

            self._ec2 = boto3.client("ec2", region_name=self.region)
            self._ssm = boto3.client("ssm", region_name=self.region)
        except ImportError:
            raise RuntimeError("boto3가 설치되어 있지 않습니다. pip install boto3")
