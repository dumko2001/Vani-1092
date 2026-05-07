"""Audio format conversion helpers.

This project keeps the LiveKit path mono at 48kHz to avoid channel-shape
bugs while still matching the agent SDK's native frame sizes.

LiveKit sends/receives: 48kHz, mono, int16, 10ms frames
Sarvam STT expects:     16kHz, mono, WAV
Sarvam TTS returns:     WAV (various rates, usually 22kHz mono)
"""

import io
import wave
import numpy as np
from scipy import signal


def livekit_to_wav(audio_bytes: bytes, sample_rate: int = 48000, channels: int = 1) -> bytes:
    """Convert LiveKit PCM buffer → 16kHz mono WAV for Sarvam STT."""
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # Deinterleave stereo
    if channels == 2:
        arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
    
    # Resample 48kHz → 16kHz
    if sample_rate != 16000:
        arr_float = arr.astype(np.float32) / 32768.0
        num_samples = int(len(arr_float) * 16000 / sample_rate)
        arr_resampled = signal.resample(arr_float, num_samples)
        arr = (arr_resampled * 32767).astype(np.int16)
    
    # Write WAV
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(arr.tobytes())
    return buf.getvalue()


def wav_to_livekit_frames(wav_bytes: bytes, target_sr: int = 48000) -> list[bytes]:
    """Convert Sarvam TTS WAV → list of 10ms mono PCM chunks for LiveKit.

    Each chunk is 960 bytes = 480 samples x 2 bytes at 48kHz mono.
    """
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        nchannels = wf.getnchannels()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        data = wf.readframes(nframes)

    arr = np.frombuffer(data, dtype=np.int16)

    # Collapse multi-channel input to mono.
    if nchannels > 1:
        arr = arr.reshape(-1, nchannels).mean(axis=1).astype(np.int16)

    # Resample to 48kHz if needed
    if framerate != target_sr:
        arr_float = arr.astype(np.float32) / 32768.0
        num_samples = int(len(arr_float) * target_sr / framerate)
        arr_resampled = signal.resample(arr_float, num_samples)
        arr = (arr_resampled * 32767).astype(np.int16)
    
    # Split into 10ms mono frames.
    frame_bytes = 960
    raw = arr.tobytes()
    frames = []
    for i in range(0, len(raw), frame_bytes):
        chunk = raw[i:i+frame_bytes]
        if len(chunk) < frame_bytes:
            chunk += b'\x00' * (frame_bytes - len(chunk))
        frames.append(chunk)
    
    return frames


def is_speech_present(audio_bytes: bytes, threshold: int = 500) -> bool:
    """Simple energy-based VAD. Returns True if audio has significant energy."""
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    energy = np.abs(arr).mean()
    return energy > threshold
