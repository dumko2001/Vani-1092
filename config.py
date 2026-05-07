"""Configuration — reads from .env, provides typed access."""

import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE_URL = "https://api.sarvam.ai"

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

# Model choices
STT_MODEL = "saarika:v2.5"         # Sarvam speech-to-text
LLM_MODEL = "sarvam-m"             # Sarvam-M (FREE)
TTS_MODEL = "bulbul:v2"            # Sarvam text-to-speech
TTS_SPEAKER = "anushka"            # Female voice, works for hi/en

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vani1092.db")


def _sqlite_path_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "sqlite":
        raise ValueError("Only sqlite DATABASE_URL is supported for this prototype.")
    if parsed.path.startswith("/./"):
        return parsed.path[3:]
    if parsed.path.startswith("/"):
        return parsed.path[1:]
    return parsed.path or "vani1092.db"


DATABASE_PATH = _sqlite_path_from_url(DATABASE_URL)

# Confirmation threshold
CONFIDENCE_THRESHOLD = 0.80

# Panic keywords that trigger CRITICAL urgency
# NOTE: Be careful not to include common words like "help" which appear in non-critical contexts
PANIC_KEYWORDS = [
    "bachaao", "bachao", "police", "maro", "maar", "dying",
    "dead", "kill", "dying", "emergency", "fire", "accident",
    "bachaav", "kaapat", "chidru", "haal",   # Kannada approximations
    "bachaao", "mujhe bachao", "following me", "threatening",
    "attacking", "attack", "stabbed", "bleeding", "dying",
]
