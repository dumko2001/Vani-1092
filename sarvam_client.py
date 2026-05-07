"""Sarvam API client — thin wrapper around their REST APIs.

Tested endpoints (as of May 2026):
- POST /speech-to-text         → transcript (multipart/form-data)
- POST /text-to-speech         → audio bytes (JSON, returns base64)
- POST /v1/chat/completions    → LLM response (JSON)

NOTE: These are BLOCKING synchronous calls. Use asyncio.to_thread() when calling from async code.
"""

import json
import base64
import requests
from config import SARVAM_API_KEY, SARVAM_BASE_URL, STT_MODEL, LLM_MODEL, TTS_MODEL, TTS_SPEAKER

HEADERS_JSON = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}
HEADERS_FILE = {"api-subscription-key": SARVAM_API_KEY}  # requests sets multipart boundary


def stt_with_meta(audio_bytes: bytes, language_code: str = "unknown") -> dict:
    """Speech to text. Returns transcript and language metadata."""
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"model": STT_MODEL, "language_code": language_code}
    resp = requests.post(
        f"{SARVAM_BASE_URL}/speech-to-text",
        headers=HEADERS_FILE,
        files=files,
        data=data,
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    return {
        "transcript": result.get("transcript", ""),
        "language_code": result.get("language_code"),
        "language_probability": result.get("language_probability"),
        "raw": result,
    }


def stt(audio_bytes: bytes, language_code: str = "unknown") -> str:
    """Backward-compatible STT helper returning transcript only."""
    return stt_with_meta(audio_bytes, language_code=language_code).get("transcript", "")


def llm_chat(messages: list, temperature: float = 0.3) -> str:
    """Chat completion. messages = [{"role": "...", "content": "..."}]"""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    resp = requests.post(
        f"{SARVAM_BASE_URL}/v1/chat/completions",
        headers=HEADERS_JSON,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    raw = result["choices"][0]["message"]["content"]
    # Strip <think> tags (Sarvam-M reasoning)
    import re
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    return raw.strip()


def tts(text: str, language_code: str = "hi") -> bytes:
    """Text to speech. Returns audio bytes (WAV)."""
    payload = {
        "model": TTS_MODEL,
        "inputs": [text],
        "target_language_code": language_code,
        "speaker": TTS_SPEAKER,
    }
    resp = requests.post(
        f"{SARVAM_BASE_URL}/text-to-speech",
        headers=HEADERS_JSON,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    audio_b64 = result["audios"][0]
    return base64.b64decode(audio_b64)


def llm_json(messages: list, temperature: float = 0.2) -> dict:
    """Call LLM and parse response as JSON."""
    # Add JSON-only instruction
    json_messages = messages.copy()
    has_system = any(m["role"] == "system" for m in json_messages)
    if has_system:
        for m in json_messages:
            if m["role"] == "system":
                m["content"] += "\n\nRespond ONLY with valid JSON. No markdown, no explanations, no <think> tags."
                break
    else:
        json_messages.insert(0, {"role": "system", "content": "Respond ONLY with valid JSON. No markdown, no explanations, no <think> tags."})

    raw = llm_chat(json_messages, temperature)
    # Strip <think> tags (Sarvam-M reasoning)
    import re
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    # Strip markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw.rsplit("\n", 1)[0] if "\n" in raw else raw[:-3]
    raw = raw.strip()
    return json.loads(raw)
