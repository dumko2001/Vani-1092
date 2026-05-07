"""LiveKit Agent — handles real-time voice with citizens.

This agent connects to a LiveKit room, listens to citizen audio,
runs it through STT → LLM → TTS, and speaks back.

Run:
    python agent.py dev

Requirements:
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env
"""

import os
import asyncio
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli

import vani1092.database as db
from vani1092.audio_utils import livekit_to_wav, wav_to_livekit_frames, is_speech_present
from vani1092.sarvam_client import stt_with_meta as sarvam_stt_with_meta, tts as sarvam_tts
from vani1092.conversation_engine import (
    analyze_transcript,
    should_bypass_confirmation,
    build_confirmation_prompt,
    build_reask_prompt,
    build_route_message,
    check_confirmation_response,
    build_handoff_payload,
    CONFIDENCE_THRESHOLD,
)

load_dotenv()

participant_states = {}


def _tts_language_from_analysis(analysis: dict) -> str:
    hint = (analysis or {}).get("language_hint", "en_or_mixed")
    if hint == "kn":
        return "kn-IN"
    if hint == "hi":
        return "hi-IN"
    return "en-IN"


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    room = ctx.room
    print(f"[Agent] Connected to room: {room.name}")
    
    # Create audio source for publishing TTS
    audio_source = rtc.AudioSource(48000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-voice", audio_source)
    await room.local_participant.publish_track(track)
    
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, pub, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"[Agent] Subscribed to {participant.identity}'s audio")
            asyncio.create_task(handle_citizen(track, participant, room, audio_source))
    
    @room.on("track_unsubscribed")
    def on_track_unsubscribed(track: rtc.Track, pub, participant: rtc.RemoteParticipant):
        print(f"[Agent] {participant.identity} left")
        if participant.identity in participant_states:
            call_id = participant_states[participant.identity]["call_id"]
            db.update_call(call_id, status="ended", ended_at=datetime.now(timezone.utc).isoformat())
            del participant_states[participant.identity]
    
    while True:
        await asyncio.sleep(1)


async def handle_citizen(track: rtc.Track, participant, room, audio_source):
    """Handle one citizen's audio stream."""
    call_id = str(uuid.uuid4())[:8]
    db.create_call(call_id, caller_number=participant.identity)
    
    participant_states[participant.identity] = {
        "call_id": call_id,
        "analysis": None,
        "awaiting_confirmation": False,
        "confirmed": False,
        "routed": False,
    }
    
    # Greet
    await speak(audio_source, "Hello, I am Vani. Please tell me how I can help you today.", language_code="en-IN")
    
    audio_stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1)
    buffer = bytearray()
    last_speech_time = asyncio.get_event_loop().time()
    
    async for frame in audio_stream:
        state = participant_states.get(participant.identity)
        if not state or state["routed"]:
            break
        
        buffer.extend(frame.frame.data.tobytes())
        now = asyncio.get_event_loop().time()
        
        # Process every ~2.5 seconds of audio
        if len(buffer) >= 48000 * 2 * 2.5:  # 2.5s, 48kHz, mono, int16
            data = bytes(buffer)
            buffer = bytearray()
            
            if not is_speech_present(data):
                continue
            
            # STT
            try:
                wav = livekit_to_wav(data, sample_rate=48000, channels=1)
                stt_result = sarvam_stt_with_meta(wav, language_code="unknown")
            except Exception as e:
                print(f"[STT Error] {e}")
                continue
            transcript = stt_result.get("transcript", "")
            
            if not transcript or not transcript.strip():
                continue
            
            print(f"[Citizen] {transcript}")
            db.add_transcript_turn(call_id, "user", transcript)
            
            # Process
            await process_turn(
                participant.identity,
                call_id,
                transcript,
                audio_source,
                stt_result.get("language_code", ""),
                stt_result.get("language_probability", 0.0),
            )
    
    if participant.identity in participant_states:
        del participant_states[participant.identity]


async def process_turn(identity, call_id, transcript, audio_source, stt_language_code="", stt_language_probability=0.0):
    state = participant_states.get(identity)
    if not state:
        return
    
    # Waiting for confirmation response
    if state["awaiting_confirmation"]:
        result = check_confirmation_response(transcript)
        
        if result == "yes":
            state["confirmed"] = True
            state["awaiting_confirmation"] = False
            msg = await build_route_message(state["analysis"], transcript)
            await speak(audio_source, msg, language_code=_tts_language_from_analysis(state.get("analysis")))
            await do_handoff(call_id, state)
            return
            
        elif result == "no":
            state["awaiting_confirmation"] = False
            state["analysis"] = None
            msg = await build_reask_prompt(transcript)
            await speak(audio_source, msg, language_code=_tts_language_from_analysis(state.get("analysis")))
            return
            
        else:
            await speak(audio_source, "I did not catch that. Please say yes or no.", language_code="en-IN")
            return
    
    # Fresh analysis
    try:
        analysis = await analyze_transcript(
            transcript,
            stt_language_code=stt_language_code,
            stt_language_probability=stt_language_probability,
        )
    except Exception as e:
        print(f"[Analysis Error] {e}")
        await speak(audio_source, "I did not understand. Can you please repeat that?", language_code="en-IN")
        return
    
    state["analysis"] = analysis
    print(f"[Analysis] {analysis['intent']} | {analysis['urgency']} | conf={analysis['confidence']:.2f}")
    
    # Critical bypass
    if should_bypass_confirmation(analysis):
        msg = await build_route_message(analysis, transcript)
        await speak(audio_source, msg, language_code=_tts_language_from_analysis(analysis))
        await do_handoff(call_id, state)
        return
    
    # Low confidence
    if analysis["confidence"] < CONFIDENCE_THRESHOLD:
        msg = await build_reask_prompt(transcript)
        await speak(audio_source, msg, language_code=_tts_language_from_analysis(analysis))
        return
    
    # Ask for confirmation
    state["awaiting_confirmation"] = True
    msg = await build_confirmation_prompt(analysis, transcript)
    await speak(audio_source, msg, language_code=_tts_language_from_analysis(analysis))


async def speak(audio_source: rtc.AudioSource, text: str, language_code: str = "en-IN"):
    """TTS and publish audio frames."""
    print(f"[Agent] {text}")
    try:
        audio_bytes = sarvam_tts(text, language_code=language_code)
        frames = wav_to_livekit_frames(audio_bytes)
        for frame_data in frames:
            frame = rtc.AudioFrame(frame_data, 48000, 1, 480)
            await audio_source.capture_frame(frame)
    except Exception as e:
        print(f"[TTS Error] {e}")


async def do_handoff(call_id: str, state: dict):
    state["routed"] = True
    analysis = state["analysis"]
    call = db.get_call(call_id)
    transcript = []
    if call:
        import json
        try:
            transcript = json.loads(call.get("transcript", "[]"))
        except:
            pass
    
    payload = build_handoff_payload(call_id, analysis, transcript)
    import json
    db.update_call(
        call_id,
        status="waiting_operator",
        summary=analysis["summary"],
        intent=analysis["intent"],
        urgency=analysis["urgency"],
        location=analysis["location"],
        confidence=analysis["confidence"],
        confirmation_given=0 if should_bypass_confirmation(analysis) else int(state.get("confirmed", False)),
        handoff_payload=json.dumps(payload),
    )
    print(f"[Handoff] Call {call_id} → operator queue")


if __name__ == "__main__":
    db.init_db()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
