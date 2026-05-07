"""Test harness — automated testing without clicking around.

Usage:
    python test_harness.py

This feeds text directly to the conversation engine (bypassing STT/TTS)
to test the core logic: intent extraction, urgency detection, confirmation flow.
"""

import asyncio
from dataclasses import dataclass

from vani1092.conversation_engine import (
    analyze_transcript,
    should_bypass_confirmation,
    build_confirmation_prompt,
    check_confirmation_response,
    build_handoff_payload,
)


@dataclass
class TestResult:
    passed: bool
    test_id: str
    errors: list


TEST_CASES = [
    {
        "id": "TC-01",
        "input": "I want to file a complaint about the water supply in my area",
        "language": "en",
        "expected_intent": "complaint",
        "expected_urgency": "LOW",
        "must_confirm": True,
    },
    {
        "id": "TC-02",
        "input": "Someone is following me near the bus stand, I am very scared",
        "language": "en",
        "expected_intent": "emergency",  # LLM classifies following as emergency/harassment
        "expected_urgency": "CRITICAL",
        "must_confirm": False,
    },
    {
        "id": "TC-03",
        "input": "Help me, someone is trying to kill me right now",
        "language": "en",
        "expected_intent": "emergency",  # LLM classifies as emergency
        "expected_urgency": "CRITICAL",
        "must_confirm": False,  # Bypasses confirmation
    },
    {
        "id": "TC-04",
        "input": "Mera phone chori ho gaya",
        "language": "hi",
        "expected_intent": "help_request",
        "expected_urgency": "MEDIUM",
        "must_confirm": True,
    },
    {
        "id": "TC-05",
        "input": "Street light is not working near my house for 3 days",
        "language": "en",
        "expected_intent": "complaint",
        "expected_urgency": "LOW",
        "must_confirm": True,
    },
]


async def run_test_case(tc: dict) -> TestResult:
    """Run a single test case."""
    errors = []
    
    # Step 1: Analyze transcript
    analysis = await analyze_transcript(tc["input"])
    
    # Check intent (allow harassment/emergency for safety cases)
    is_safety_case = tc["id"] in ["TC-02", "TC-03"]
    if is_safety_case:
        valid_intents = ["harassment", "emergency"]
        if analysis["intent"] not in valid_intents:
            errors.append(
                f"Intent mismatch: expected one of {valid_intents}, got '{analysis['intent']}'"
            )
    elif analysis["intent"] != tc["expected_intent"]:
        errors.append(
            f"Intent mismatch: expected '{tc['expected_intent']}', got '{analysis['intent']}'"
        )
    
    # Check urgency
    if analysis["urgency"] != tc["expected_urgency"]:
        errors.append(
            f"Urgency mismatch: expected '{tc['expected_urgency']}', got '{analysis['urgency']}'"
        )
    
    # Check bypass
    bypass = should_bypass_confirmation(analysis)
    if bypass != (not tc["must_confirm"]):
        errors.append(
            f"Bypass mismatch: expected bypass={not tc['must_confirm']}, got {bypass}"
        )
    
    # Check summary exists and is non-empty
    if not analysis.get("summary") or len(analysis["summary"]) < 5:
        errors.append(
            f"Summary too short or missing: '{analysis.get('summary', '')}'"
        )
    
    # Check confidence is a valid number
    if not isinstance(analysis.get("confidence"), (int, float)):
        errors.append(
            f"Confidence not a number: {analysis.get('confidence')}"
        )
    elif analysis["confidence"] < 0 or analysis["confidence"] > 1:
        errors.append(
            f"Confidence out of range: {analysis['confidence']}"
        )
    
    # If confirmation is expected, verify the prompt makes sense
    if tc["must_confirm"]:
        confirm_prompt = await build_confirmation_prompt(analysis, tc["input"])
        if len(confirm_prompt) < 10:
            errors.append(
                f"Confirmation prompt too short: '{confirm_prompt}'"
            )
        # Should contain a question mark (basic check)
        if "?" not in confirm_prompt and "sahee" not in confirm_prompt.lower():
            errors.append(
                f"Confirmation prompt doesn't seem like a question: '{confirm_prompt[:50]}...'"
            )
    
    # Check handoff payload
    payload = build_handoff_payload(tc["id"], analysis, [])
    required_keys = ["call_id", "urgency", "intent", "summary", "recommended_action"]
    for key in required_keys:
        if key not in payload:
            errors.append(f"Handoff payload missing key: {key}")
    
    passed = len(errors) == 0
    return TestResult(passed=passed, test_id=tc["id"], errors=errors)


def run_confirmation_tests():
    """Test the confirmation response parser."""
    tests = [
        ("yes", "yes"),
        ("haan", "yes"),
        ("houdu", "yes"),
        ("sahi hai", "yes"),
        ("no", "no"),
        ("nahi", "no"),
        ("illa", "no"),
        ("maybe", "unclear"),
        ("", "unclear"),
    ]
    
    results = []
    for text, expected in tests:
        actual = check_confirmation_response(text)
        passed = actual == expected
        if not passed:
            results.append(f"  Confirm parser: '{text}' → expected '{expected}', got '{actual}'")
    
    return results


async def main():
    print("=" * 60)
    print("VANI-1092 TEST HARNESS")
    print("=" * 60)
    
    results = []
    
    # Run main test cases
    for tc in TEST_CASES:
        result = await run_test_case(tc)
        results.append(result)
        
        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {result.test_id}: {tc['input'][:50]}...")
        if result.errors:
            for err in result.errors:
                print(f"       → {err}")
    
    # Run confirmation parser tests
    print("\n" + "-" * 60)
    print("Confirmation Response Parser Tests")
    print("-" * 60)
    confirm_errors = run_confirmation_tests()
    if confirm_errors:
        for err in confirm_errors:
            print(f"  {err}")
    else:
        print("  All confirmation parser tests passed")
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} test cases passed")
    if passed == total and not confirm_errors:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — see above")
    print("=" * 60)
    
    return passed == total and not confirm_errors


if __name__ == "__main__":
    import sys
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
