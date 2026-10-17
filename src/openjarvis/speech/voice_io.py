"""Microphone recording with silence detection and audio playback helpers."""

from __future__ import annotations

import io
import os
import wave

_SAMPLE_RATE = 16000
_CHANNELS = 1
_CHUNK = 1024
_DEFAULT_SILENCE_THRESHOLD = 500  # RMS below this → silence
_SILENCE_THRESHOLD = _DEFAULT_SILENCE_THRESHOLD  # backward-compat alias
_DEFAULT_SILENCE_SECONDS = 1.5  # default seconds of silence before auto-stop
_SILENCE_SECONDS = _DEFAULT_SILENCE_SECONDS  # backward-compat alias
_STARTUP_SILENCE_SECONDS = 5.0  # give up early if speech never begins
_MAX_RECORD_SECONDS = 30  # safety ceiling


def _get_silence_threshold() -> int:
    """Read silence threshold from JARVIS_SILENCE_THRESHOLD env var, defaulting to 500."""
    raw = os.environ.get("JARVIS_SILENCE_THRESHOLD", "").strip()
    if raw:
        try:
            val = int(float(raw))
            if val > 0:
                return val
        except (ValueError, OverflowError):
            pass
    return _DEFAULT_SILENCE_THRESHOLD


def _get_silence_seconds() -> float:
    """Read silence duration from JARVIS_SILENCE_SECONDS env var, defaulting to 1.5s."""
    raw = os.environ.get("JARVIS_SILENCE_SECONDS", "").strip()
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return _DEFAULT_SILENCE_SECONDS


def _rms(data: bytes) -> float:
    """Compute RMS amplitude of 16-bit PCM bytes."""
    import struct

    n = len(data) // 2
    if n == 0:
        return 0.0
    shorts = struct.unpack(f"{n}h", data[: n * 2])
    return (sum(s * s for s in shorts) / n) ** 0.5


def record_until_silence(
    *,
    sample_rate: int = _SAMPLE_RATE,
    silence_threshold: int | None = None,
    silence_seconds: float | None = None,
    startup_silence_seconds: float = _STARTUP_SILENCE_SECONDS,
    max_seconds: float = _MAX_RECORD_SECONDS,
) -> bytes:
    """Record from the default microphone until silence is detected.

    Returns raw WAV bytes (16-bit mono).
    Raises RuntimeError if sounddevice is not installed.
    """
    if silence_threshold is None:
        silence_threshold = _get_silence_threshold()
    if silence_seconds is None:
        silence_seconds = _get_silence_seconds()

    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is required for voice input. "
            "Install with: pip install sounddevice"
        )

    chunks_per_second = sample_rate / _CHUNK
    silence_chunks = int(silence_seconds * chunks_per_second)
    startup_silence_chunks = max(1, int(startup_silence_seconds * chunks_per_second))
    max_chunks = int(max_seconds * chunks_per_second)

    frames: list[bytes] = []
    silence_count = 0
    has_speech = False
    try:
        from openjarvis.speech.wake import is_playback_active

        while is_playback_active():
            import time

            time.sleep(0.02)
    except Exception:
        pass

    with sd.RawInputStream(
        samplerate=sample_rate,
        channels=_CHANNELS,
        dtype="int16",
        blocksize=_CHUNK,
    ) as stream:
        for _ in range(max_chunks):
            raw, _ = stream.read(_CHUNK)
            data = bytes(raw)
            frames.append(data)

            amplitude = _rms(data)
            if amplitude > silence_threshold:
                has_speech = True
                silence_count = 0
            elif not has_speech and len(frames) >= startup_silence_chunks:
                break
            elif has_speech:
                silence_count += 1
                if silence_count >= silence_chunks:
                    break

    return _frames_to_wav(frames, sample_rate)


def _frames_to_wav(frames: list[bytes], sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def play_wav(audio: bytes, sample_rate: int = 24000) -> None:
    """Play raw WAV bytes through the default output device.

    If the bytes are a valid WAV file, sample rate is read from the header;
    otherwise ``sample_rate`` is used as a fallback.
    """
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "sounddevice, numpy, and soundfile are required for voice output. "
            "Install with: pip install sounddevice numpy soundfile"
        )

    buf = io.BytesIO(audio)
    try:
        data, sr = sf.read(buf, dtype="float32")
    except Exception:
        # Fall back: treat as raw PCM
        import struct

        n = len(audio) // 2
        data = (
            np.array(struct.unpack(f"{n}h", audio[: n * 2]), dtype="float32") / 32768.0
        )
        sr = sample_rate

    from openjarvis.speech.wake import playback_guard

    with playback_guard():
        sd.play(data, sr)
        sd.wait()


__all__ = ["play_wav", "record_until_silence"]
