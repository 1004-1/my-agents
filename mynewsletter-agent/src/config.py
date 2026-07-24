import os
from zoneinfo import ZoneInfo

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

LABEL_NAME = "Meco_3631eff2-09a3-465b-8569-8d6e623ef8f3"

GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_MODEL = "llama3.1:8b"

CHUNK_SIZE = 10000
CHUNK_OVERLAP = 500
MAX_REPORT_CHARS = 90000
TIMEZONE = ZoneInfo("Asia/Seoul")

DRY_RUN_DELETE = False

LANG_KOREAN = "ko"
LANG_FOREIGN = "foreign"

NORMAL_FINISH_REASONS = {"STOP"}
