from datetime import datetime

from .config import TIMEZONE


def log(message: str) -> None:
    now = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)
