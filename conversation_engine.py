"""Conversation engine — intent, urgency, confirmation, handoff.

This is the brain. It takes a transcript and decides:
- What is the citizen's problem? (intent)
- How urgent is it? (urgency)
- Where is it happening? (location)
- Should we ask for confirmation or route immediately?

All LLM calls run in threads to avoid blocking the async event loop.
"""

import asyncio
import json
import re
from sarvam_client import llm_chat, llm_json
from config import PANIC_KEYWORDS, CONFIDENCE_THRESHOLD


SYSTEM_UNDERSTANDING = """You are an AI assistant for a government helpline (1092).
Analyze the citizen's message and output ONLY a JSON object with these fields:

- "intent": one of ["harassment", "complaint", "help_request", "emergency", "general"]
- "urgency": one of ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
- "location": any place mentioned, or ""
- "summary": one sentence summarizing the issue
- "confidence": number 0.0 to 1.0 (how sure you are)
- "panic_words_found": list of panic/danger words found in the text

Rules:
- "emergency" or "harassment" with immediate danger = CRITICAL
- "complaint" about infrastructure = LOW
- "help_request" for lost items = MEDIUM
- If citizen says "right now", "currently", "immediately" → raise urgency by one level
"""


SYSTEM_CONFIRMATION = """You are Vani-1092, a helpline assistant.
The citizen said something. You understood their issue.

Your job: ask for confirmation in a SHORT, CLEAR sentence.
Speak in the SAME LANGUAGE as the citizen.

Example outputs:
- "Did I understand correctly that someone is following you near the bus stand?"
- "Mujhe samajhna hai — aapka phone chori ho gaya hai?"
- "Nimage correct aagide — nimage bus stand hathira yaraadaru follow maadta idaara?"

Keep it under 15 words. Ask ONE thing.
"""


SYSTEM_REASK = """You are Vani-1092. You did NOT understand the citizen clearly.
Politely ask them to repeat or clarify in simpler words.
Speak in the SAME LANGUAGE as the citizen.
Keep it under 10 words.
"""


SYSTEM_ROUTE_MESSAGE = """You are Vani-1092. You have understood the citizen's issue and they confirmed it.
Politely tell them you are connecting them to a human operator.
Speak in the SAME LANGUAGE as the citizen.
Keep it under 10 words.
"""


def _run_in_thread(func, *args, **kwargs):
    """Run a blocking function in a thread pool."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop, call directly
        return func(*args, **kwargs)
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _detect_emotion_signals(transcript: str) -> dict:
    """Low-cost heuristic emotion signals to support operator context."""
    text = transcript.lower()
    buckets = {
        "distress": ["help", "save me", "bachao", "bachaao", "please", "cry", "panic"],
        "urgency": ["now", "immediately", "jaldi", "abhi", "right now", "urgent"],
        "anger": ["angry", "furious", "mad", "shouting", "threaten", "gussa"],
        "fear": ["afraid", "scared", "darr", "dar", "threat", "following me"],
        "confusion": ["don't understand", "samajh", "confused", "what to do", "kya karu"],
    }
    scores = {k: 0 for k in buckets}
    for label, words in buckets.items():
        scores[label] = sum(1 for w in words if w in text)
    if max(scores.values()) == 0:
        return {"emotion": "neutral_or_calm", "emotion_scores": scores}
    winner = max(scores, key=scores.get)
    return {"emotion": winner, "emotion_scores": scores}


def _detect_language_hint(transcript: str) -> str:
    """Simple hint only; real language handling remains model-driven."""
    text = transcript.lower()
    kn_words = ["nimage", "nanna", "illa", "yenu", "beku", "houdu"]
    hi_words = ["mujhe", "mera", "nahi", "haan", "kya", "kripya", "madad"]
    if any(w in text for w in kn_words):
        return "kn"
    if any(w in text for w in hi_words):
        return "hi"
    return "en_or_mixed"


def _normalize_language_code(code: str) -> str:
    if not code:
        return "en_or_mixed"
    code = code.lower()
    if code.startswith("kn"):
        return "kn"
    if code.startswith("hi"):
        return "hi"
    if code.startswith("en"):
        return "en_or_mixed"
    return "en_or_mixed"


async def analyze_transcript(
    transcript: str,
    stt_language_code: str = "",
    stt_language_probability: float = 0.0,
) -> dict:
    """Extract intent, urgency, location, summary from transcript."""
    messages = [
        {"role": "system", "content": SYSTEM_UNDERSTANDING},
        {"role": "user", "content": transcript},
    ]
    result = await _run_in_thread(llm_json, messages, temperature=0.2)

    # Keyword override for panic words
    lower = transcript.lower()
    panic_found = [w for w in PANIC_KEYWORDS if w in lower]
    if panic_found:
        result["panic_words_found"] = panic_found
        result["urgency"] = "CRITICAL"
        result["confidence"] = min(1.0, result.get("confidence", 0.5) + 0.2)

    # Ensure all fields exist
    result.setdefault("intent", "general")
    result.setdefault("urgency", "LOW")
    result.setdefault("location", "")
    result.setdefault("summary", transcript[:100])
    result.setdefault("confidence", 0.5)
    result.setdefault("panic_words_found", [])
    # Prefer STT language metadata when confidence is reasonable.
    stt_hint = _normalize_language_code(stt_language_code)
    try:
        stt_prob = float(stt_language_probability or 0.0)
    except (TypeError, ValueError):
        stt_prob = 0.0
    if stt_prob >= 0.6:
        result.setdefault("language_hint", stt_hint)
    else:
        result.setdefault("language_hint", _detect_language_hint(transcript))
    result.setdefault("stt_language_code", stt_language_code or "")
    result.setdefault("stt_language_probability", stt_prob)
    result.update(_detect_emotion_signals(transcript))

    return result


def should_bypass_confirmation(analysis: dict) -> bool:
    """If CRITICAL urgency, skip confirmation and route immediately."""
    return analysis["urgency"] == "CRITICAL"


async def build_confirmation_prompt(analysis: dict, transcript: str) -> str:
    """Ask citizen to confirm our understanding."""
    messages = [
        {"role": "system", "content": SYSTEM_CONFIRMATION},
        {"role": "user", "content": f"Citizen said: {transcript}\nMy understanding: {analysis['summary']}\nLocation: {analysis['location']}"},
    ]
    prompt = await _run_in_thread(llm_chat, messages, temperature=0.5)
    prompt = prompt.strip()
    if prompt and "?" not in prompt:
        prompt = f"{prompt}?"
    return prompt


async def build_reask_prompt(transcript: str) -> str:
    """Ask citizen to clarify."""
    messages = [
        {"role": "system", "content": SYSTEM_REASK},
        {"role": "user", "content": f"Citizen said: {transcript}\nI did not understand."},
    ]
    return await _run_in_thread(llm_chat, messages, temperature=0.5)


async def build_route_message(analysis: dict, transcript: str) -> str:
    """Tell citizen we are connecting them."""
    messages = [
        {"role": "system", "content": SYSTEM_ROUTE_MESSAGE},
        {"role": "user", "content": f"Citizen said: {transcript}\nConfirmed."},
    ]
    return await _run_in_thread(llm_chat, messages, temperature=0.5)


def check_confirmation_response(citizen_text: str) -> str:
    """Parse citizen's response to a confirmation question.
    Returns: 'yes', 'no', or 'unclear'
    
    This is fast (no LLM call needed for most cases), so it's synchronous.
    """
    lower = citizen_text.lower().strip()
    tokens = re.findall(r"[a-zA-Z']+", lower)
    token_set = set(tokens)

    yes_words = ["yes", "yeah", "yep", "correct", "right", "true", "exactly",
                 "haan", "haa", "han", "sahi", "sahihai", "theek", "thik",
                 "houdu", "hoo", "sari", "sariyagi"]

    no_words = ["no", "nope", "wrong", "incorrect", "false",
                "nahi", "na", "galat", "tappu",
                "illa", "thappu", "alla"]

    for w in yes_words:
        if w in token_set or w in lower:
            return "yes"
    for w in no_words:
        if w in token_set:
            return "no"

    if len(lower.split()) <= 2:
        return "unclear"

    # Keep confirmation parsing deterministic and low-latency.
    return "unclear"


def build_handoff_payload(call_id: str, analysis: dict, transcript: list) -> dict:
    """Structured payload for the operator dashboard."""
    return {
        "call_id": call_id,
        "language": "auto-detected",
        "urgency": analysis["urgency"],
        "intent": analysis["intent"],
        "location": analysis["location"],
        "confidence": analysis["confidence"],
        "emotion": analysis.get("emotion", "neutral_or_calm"),
        "emotion_scores": analysis.get("emotion_scores", {}),
        "language_hint": analysis.get("language_hint", "en_or_mixed"),
        "stt_language_code": analysis.get("stt_language_code", ""),
        "stt_language_probability": analysis.get("stt_language_probability", 0.0),
        "summary": analysis["summary"],
        "transcript": transcript,
        "confirmation_given": not should_bypass_confirmation(analysis),
        "recommended_action": _recommend_action(analysis),
    }


def _recommend_action(analysis: dict) -> str:
    if analysis["urgency"] == "CRITICAL":
        return "Connect immediately"
    if analysis["intent"] == "harassment":
        return "Connect to female operator if available"
    if analysis["intent"] == "emergency":
        return "Connect to emergency response team"
    return "Standard queue"
