"""Quick API key test — uses Sarvam-M LLM which is FREE.
Run this to verify your key works before starting the agent."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY", "")
if not API_KEY or API_KEY == "your-actual-key-here":
    print("ERROR: Set SARVAM_API_KEY in .env first")
    exit(1)

print(f"Testing Sarvam API key: {API_KEY[:10]}...")

# Test 1: Free LLM (Sarvam-M) — costs ₹0
print("\n[1/2] Testing Sarvam-M LLM (FREE)...")
try:
    resp = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={"api-subscription-key": API_KEY, "Content-Type": "application/json"},
        json={
            "model": "sarvam-m",
            "messages": [{"role": "user", "content": "Say 'API key works' in Kannada"}],
            "temperature": 0.3,
        },
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    text = result["choices"][0]["message"]["content"]
    print(f"✅ LLM OK: {text}")
except Exception as e:
    print(f"❌ LLM FAILED: {e}")
    exit(1)

# Test 2: TTS — cheap, one short sentence (~₹0.03)
print("\n[2/2] Testing Sarvam TTS (Bulbul v2)...")
try:
    resp = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": API_KEY, "Content-Type": "application/json"},
        json={
            "model": "bulbul:v2",
            "inputs": ["Hello, this is a test."],
            "target_language_code": "en-IN",
            "speaker": "anushka",
        },
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    audio_b64 = result["audios"][0]
    audio_bytes = __import__("base64").b64decode(audio_b64)
    print(f"✅ TTS OK: {len(audio_bytes)} bytes of audio received")
except Exception as e:
    print(f"❌ TTS FAILED: {e}")
    exit(1)

print("\n🎉 All tests passed. Your API key works.")
print("   LLM:  FREE (Sarvam-M)")
print("   TTS:  ~₹0.03 for this test")
print("   STT:  Not tested (will test when agent runs)")
