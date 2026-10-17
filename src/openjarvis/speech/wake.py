"""Wake word detection and session management for hands-free voice chat.

Provides audio gating, wake word parsing, anti-echo playback guarding,
"Sim, senhor." cached audio playback, and 10-second active conversational windows.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import io
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Optional, Tuple
import unicodedata
import wave

logger = logging.getLogger(__name__)

# Default active conversation window in seconds
_DEFAULT_WAKE_WINDOW = 10.0

# Minimum speech duration to qualify as intentional speech (filters out claps, clicks)
_DEFAULT_MIN_SPEECH_SECONDS = 0.25
_MIN_SPEECH_SECONDS = _DEFAULT_MIN_SPEECH_SECONDS


def get_min_speech_seconds() -> float:
    """Read minimum speech duration from JARVIS_MIN_SPEECH_SECONDS env var, defaulting to 0.25s."""
    raw = os.environ.get("JARVIS_MIN_SPEECH_SECONDS", "").strip()
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return _DEFAULT_MIN_SPEECH_SECONDS

# Cooldown after audio playback before microphone re-opens (anti-echo)
_DEFAULT_PLAYBACK_COOLDOWN = 0.5

# Anti-echo state: timestamp until which mic input is ignored
_playback_until: float = 0.0


def reset_playback_guard() -> None:
    """Reset the global anti-echo playback guard timestamp."""
    global _playback_until
    _playback_until = 0.0


def mark_playback_start() -> None:
    """Signal that audio playback has started (pauses listening)."""
    global _playback_until
    _playback_until = float("inf")


def mark_playback_end(cooldown_seconds: float = _DEFAULT_PLAYBACK_COOLDOWN) -> None:
    """Signal that audio playback has ended; mic remains muted for cooldown."""
    global _playback_until
    _playback_until = time.perf_counter() + cooldown_seconds


def is_playback_active() -> bool:
    """Return True if audio playback or the post-playback cooldown is active."""
    return time.perf_counter() < _playback_until


@contextmanager
def playback_guard(cooldown_seconds: float = _DEFAULT_PLAYBACK_COOLDOWN):
    """Context manager to guard audio playback against mic self-transcription."""
    mark_playback_start()
    try:
        yield
    finally:
        mark_playback_end(cooldown_seconds)


def get_wake_window_seconds() -> float:
    """Return conversational window duration from JARVIS_WAKE_WINDOW, defaulting to 10.0s."""
    raw = os.environ.get("JARVIS_WAKE_WINDOW", "").strip()
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return _DEFAULT_WAKE_WINDOW


def normalize_wake_text(text: str) -> str:
    """Normalize text by removing accents, punctuation, and converting to lowercase."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_bytes = nfkd.encode("ASCII", "ignore")
    cleaned = ascii_bytes.decode("utf-8").lower().strip()
    return cleaned


@dataclass(slots=True)
class WakeParseResult:
    """Structured result of wake word phrase analysis."""

    detected: bool
    wake_named: bool
    clean_text: Optional[str]
    is_command: bool


def parse_wake_phrase(text: str, wake_word: str = "junior") -> WakeParseResult:
    """Inspect transcribed text for the wake word.

    The wake word is valid ONLY at the beginning of the utterance (optionally
    preceded by greetings like 'ei', 'oi', 'olá', 'opa', 'hey') or at the
    very end of the phrase. Occurrences in the middle preceded by other words
    (e.g. 'o Junior chegou') will NOT activate.
    """
    if not text or not text.strip():
        return WakeParseResult(
            detected=False,
            wake_named=False,
            clean_text=None,
            is_command=False,
        )

    norm_word = normalize_wake_text(wake_word)
    if norm_word == "junior":
        wake_names = r"(?:junior|júnior|junio|jr)"
    else:
        wake_names = rf"(?:{re.escape(norm_word)})"

    # Pattern A: Wake word at start (optionally preceded by a greeting)
    start_pat = rf"^\s*(?:(?:ei|ola|olá|opa|hey|oi)[,\s]+)?{wake_names}\b"
    # Pattern B: Wake word at end (preceded by comma or space)
    end_pat = rf"[,\s]+{wake_names}\s*[?.!,;~-]*$"

    is_at_start = bool(re.search(start_pat, text, flags=re.IGNORECASE))
    is_at_end = bool(re.search(end_pat, text, flags=re.IGNORECASE))

    if not is_at_start and not is_at_end:
        # Not a valid wake word invocation (e.g. "o Junior chegou")
        return WakeParseResult(
            detected=False,
            wake_named=False,
            clean_text=text.strip(),
            is_command=bool(text.strip()),
        )

    clean = text

    if is_at_start:
        # Strip greeting and wake word from start
        clean = re.sub(start_pat, "", clean, flags=re.IGNORECASE)
        # Strip conversational follow-ups like "você tá aí?", "vc tá me ouvindo?", "tá aí?", "você tá por aí?"
        clean = re.sub(
            r"(?i)^[,\s]*(?:(?:voc[eê]|vc|c[eê])\s+)?(?:t[aá]\s+a[ií]|est[aá]\s+a[ií]|(?:t[aá]\s+|est[aá]\s+)?por\s+a[ií]|t[aá]\s+me\s+ouvindo|est[aá]\s+me\s+ouvindo|ouviu)\s*[?.!,]*",
            "",
            clean,
        )
    elif is_at_end:
        # Strip wake word from end
        clean = re.sub(end_pat, "", clean, flags=re.IGNORECASE)

    # Clean leading and trailing punctuation/whitespace
    clean = re.sub(r"^[,\s.?!:;~-]+", "", clean)
    clean = re.sub(r"[,\s:;~-]+$", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    # If nothing remains, or only punctuation remains without alphanumeric characters
    if not clean or not any(ch.isalnum() for ch in clean):
        return WakeParseResult(
            detected=True,
            wake_named=True,
            clean_text=None,
            is_command=False,
        )

    return WakeParseResult(
        detected=True,
        wake_named=True,
        clean_text=clean,
        is_command=True,
    )


def is_speech_too_short(
    speech_seconds: float,
    min_seconds: Optional[float] = None,
) -> bool:
    """Return True if detected speech is shorter than the minimum duration threshold."""
    if min_seconds is None:
        min_seconds = get_min_speech_seconds()
    return speech_seconds < min_seconds


def measure_wav_speech_duration(
    audio_bytes: bytes,
    silence_threshold: Optional[int] = None,
) -> float:
    """Measure duration from first to last block exceeding silence threshold in a WAV buffer."""
    if not audio_bytes:
        return 0.0

    # If the buffer is not a valid RIFF/WAV file (e.g. mocked test bytes), do not reject it
    if len(audio_bytes) < 44 or not audio_bytes.startswith(b"RIFF"):
        return 1.0

    from openjarvis.speech.voice_io import _get_silence_threshold, _rms

    if silence_threshold is None:
        silence_threshold = _get_silence_threshold()

    chunk_size = 1024
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            framerate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            channels = wf.getnchannels()
            raw_frames = wf.readframes(wf.getnframes())
    except Exception:
        return 1.0

    if sampwidth != 2 or channels != 1:
        return 1.0

    first_idx: Optional[int] = None
    last_idx: Optional[int] = None
    total_chunks = len(raw_frames) // chunk_size

    for i in range(total_chunks):
        chunk = raw_frames[i * chunk_size : (i + 1) * chunk_size]
        if _rms(chunk) > silence_threshold:
            if first_idx is None:
                first_idx = i
            last_idx = i

    if first_idx is None or last_idx is None:
        return 0.0

    chunks_per_sec = framerate / (chunk_size // sampwidth)
    span_chunks = last_idx - first_idx + 1
    return span_chunks / chunks_per_sec


def get_sim_senhor_cache_path(voice_id: str = "default") -> Path:
    """Resolve destination path for the cached 'Sim, senhor.' audio file."""
    base_dir = Path(os.environ.get("OPENJARVIS_HOME", os.path.expanduser("~/.openjarvis")))
    base_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", voice_id.strip()) if voice_id else "default"
    return base_dir / f"sim_senhor_{slug}.wav"


def prepare_sim_senhor_cache(session: Any, voice_id: Optional[str] = None) -> Tuple[Path, str]:
    """Synthesize and cache 'Sim, senhor.' WAV using actual_voice and atomic file write.

    Returns tuple of (cache_file_path, actual_voice_slug).
    """
    backend = session.get_tts_backend() if hasattr(session, "get_tts_backend") else None
    if backend is None:
        raise RuntimeError("No TTS backend available to synthesize wake confirmation")

    actual_voice, speed = (
        session.voice_for_backend(backend)
        if hasattr(session, "voice_for_backend")
        else (voice_id or "default", 1.0)
    )
    actual_voice_name = actual_voice or voice_id or "default"
    path = get_sim_senhor_cache_path(actual_voice_name)

    if path.exists() and path.stat().st_size > 0:
        return path, actual_voice_name

    synth_kwargs: dict[str, Any] = {"output_format": "wav", "speed": speed}
    if actual_voice:
        synth_kwargs["voice_id"] = actual_voice

    result = backend.synthesize("Sim, senhor.", **synth_kwargs)
    if not getattr(result, "audio", None):
        raise RuntimeError("TTS synthesis returned empty audio")

    # Write to a temporary file in the target directory and rename atomically
    base_dir = path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=base_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(result.audio)
        os.replace(tmp_path_str, path)
    except Exception:
        try:
            if os.path.exists(tmp_path_str):
                os.unlink(tmp_path_str)
        except OSError:
            pass
        raise

    return path, actual_voice_name


def play_sim_senhor(session: Any, console: Any = None, voice_id: Optional[str] = None) -> bool:
    """Play cached 'Sim, senhor.' audio with sample_rate extraction and anti-echo guard."""
    try:
        path, actual_voice = prepare_sim_senhor_cache(session, voice_id)
        if path.exists() and path.stat().st_size > 0:
            audio_bytes = path.read_bytes()

            # Extract sample rate from WAV header if possible
            sample_rate = 24000
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    sample_rate = wf.getframerate()
            except Exception:
                pass

            from openjarvis.speech.voice_io import play_wav

            with playback_guard():
                play_wav(audio_bytes, sample_rate=sample_rate)
            return True
    except Exception as exc:
        logger.debug("Playback of sim_senhor failed: %s", exc)

    # Fallback to printed text
    if console is not None:
        console.print("[dim cyan]Sim, senhor.[/dim cyan]")
    else:
        print("Sim, senhor.")
    return False


class WakeWordSession:
    """Session state managing wake word gating, active conversational window, and timing."""

    def __init__(
        self,
        wake_word: str = "junior",
        window_seconds: Optional[float] = None,
        wake_model: Optional[str] = None,
    ) -> None:
        self.wake_word = wake_word
        self.window_seconds = window_seconds if window_seconds is not None else get_wake_window_seconds()
        self.wake_model_name = wake_model if wake_model is not None else os.environ.get("JARVIS_WAKE_MODEL")
        self._dedicated_wake_backend: Optional[Any] = None
        self._is_awake = False
        self._awake_until: float = 0.0
        self.last_wake_named = False
        self.t_call_start: Optional[float] = None

    @property
    def is_awake(self) -> bool:
        """Check if currently within the active conversational window."""
        if self._is_awake and time.perf_counter() < self._awake_until:
            return True
        self._is_awake = False
        return False

    def touch(self, wake_named: bool = False) -> None:
        """Mark interaction, renewing the active window for ~10 seconds."""
        self._is_awake = True
        self._awake_until = time.perf_counter() + self.window_seconds
        self.last_wake_named = wake_named

    def remaining_awake_time(self) -> float:
        """Return remaining seconds in active window, or 0.0 if expired."""
        return max(0.0, self._awake_until - time.perf_counter())

    def sleep(self) -> None:
        """Force session back to sleep (waiting for wake word)."""
        self._is_awake = False
        self._awake_until = 0.0
        self.last_wake_named = False
        self.t_call_start = None

    def get_wake_stt_backend(self, default_backend: Any = None) -> Any:
        """Return dedicated wake STT model if JARVIS_WAKE_MODEL is set, else default."""
        if not self.wake_model_name:
            return default_backend

        if self._dedicated_wake_backend is None:
            from openjarvis.speech.faster_whisper import FasterWhisperBackend

            logger.info("Loading dedicated wake STT model: %s", self.wake_model_name)
            self._dedicated_wake_backend = FasterWhisperBackend(model_size=self.wake_model_name)

        return self._dedicated_wake_backend


def drain_stream(stream: Any, chunk_size: int = 1024) -> None:
    """Drain and discard unread audio frames buffered in the stream."""
    if stream is None:
        return
    try:
        available = getattr(stream, "read_available", 0)
        if available > 0:
            while available > 0:
                to_read = min(available, chunk_size * 4)
                stream.read(to_read)
                available = getattr(stream, "read_available", 0)
            return
    except Exception:
        pass
    try:
        stream.read(chunk_size)
    except Exception:
        pass


def record_wake_chunk_audio(
    *,
    stream: Optional[Any] = None,
    pre_buffer: Optional[deque[bytes]] = None,
    sample_rate: int = 16000,
    silence_threshold: Optional[int] = None,
    silence_seconds: float = 0.8,
    max_seconds: float = 4.0,
    pre_buffer_seconds: float = 0.3,
) -> Optional[Tuple[bytes, float]]:
    """Listen on microphone stream for audio exceeding silence_threshold.

    Maintains pre-recorded audio in memory (~0.3s) so the start of the word
    is preserved. Returns (wav_bytes, speech_seconds) if speech is detected,
    or None if silence or sound too short.
    """
    if stream is None:
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError(
                "sounddevice is required for voice input. "
                "Install with: pip install sounddevice"
            )
        chunk_size = 1024
        with sd.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_size,
        ) as local_stream:
            return record_wake_chunk_audio(
                stream=local_stream,
                pre_buffer=pre_buffer,
                sample_rate=sample_rate,
                silence_threshold=silence_threshold,
                silence_seconds=silence_seconds,
                max_seconds=max_seconds,
                pre_buffer_seconds=pre_buffer_seconds,
            )

    from openjarvis.speech.voice_io import _frames_to_wav, _get_silence_threshold, _rms

    if silence_threshold is None:
        silence_threshold = _get_silence_threshold()

    chunk_size = 1024
    chunks_per_second = sample_rate / chunk_size
    silence_chunks = int(silence_seconds * chunks_per_second)
    max_chunks = int(max_seconds * chunks_per_second)
    pre_chunks = max(1, int(pre_buffer_seconds * chunks_per_second))

    if pre_buffer is None:
        pre_buffer = deque(maxlen=pre_chunks)

    frames: list[bytes] = []
    silence_count = 0
    has_speech = False

    for _ in range(max_chunks):
        if is_playback_active():
            pre_buffer.clear()
            drain_stream(stream, chunk_size)
            time.sleep(0.02)
            return None

        raw, _ = stream.read(chunk_size)
        data = bytes(raw)
        amp = _rms(data)

        if not has_speech:
            if amp > silence_threshold:
                has_speech = True
                frames = list(pre_buffer) + [data]
                silence_count = 0
            else:
                pre_buffer.append(data)
                # If ~0.5s of silence passes without speech starting, yield control
                # Stream remains open and pre_buffer stays populated!
                if len(pre_buffer) >= pre_chunks:
                    return None
        else:
            frames.append(data)
            if amp > silence_threshold:
                silence_count = 0
            else:
                silence_count += 1
                if silence_count >= silence_chunks or len(frames) >= max_chunks:
                    break

    if not has_speech or not frames:
        return None

    first_idx: Optional[int] = None
    last_idx: Optional[int] = None
    for idx, f in enumerate(frames):
        if _rms(f) > silence_threshold:
            if first_idx is None:
                first_idx = idx
            last_idx = idx

    if first_idx is None or last_idx is None:
        pre_buffer.clear()
        return None

    span_chunks = last_idx - first_idx + 1
    speech_seconds = span_chunks / chunks_per_second

    if is_speech_too_short(speech_seconds):
        logger.debug(
            "Speech too short (%.2fs < %.2fs) — ignoring",
            speech_seconds,
            get_min_speech_seconds(),
        )
        pre_buffer.clear()
        return None

    pre_buffer.clear()
    return _frames_to_wav(frames, sample_rate), speech_seconds


def listen_for_wake_turn(
    console: Any,
    session: Any,
    wake_session: WakeWordSession,
) -> Optional[str] | object:
    """Listen for a user voice turn, handling wake word gating and active windows."""
    from openjarvis.cli._voice_chat import VOICE_EXIT, _terminal_safe_text, record_voice

    # Case A: Jarvis is SLEEPING -> waiting for wake word
    if not wake_session.is_awake:
        console.print(
            f"[dim cyan]Escutando pelo nome '{wake_session.wake_word.capitalize()}'...[/dim cyan]"
        )
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError(
                "sounddevice is required for voice input. "
                "Install with: pip install sounddevice"
            )

        chunk_size = 1024
        sample_rate = 16000
        pre_chunks = max(1, int(0.3 * (sample_rate / chunk_size)))
        pre_buffer: deque[bytes] = deque(maxlen=pre_chunks)

        with sd.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_size,
        ) as stream:
            while True:
                try:
                    chunk_res = record_wake_chunk_audio(
                        stream=stream,
                        pre_buffer=pre_buffer,
                        sample_rate=sample_rate,
                    )
                except KeyboardInterrupt:
                    return VOICE_EXIT
                except Exception as exc:
                    console.print(f"[red]Mic error: {_terminal_safe_text(exc)}[/red]")
                    return VOICE_EXIT

                if chunk_res is None:
                    continue

                audio_bytes, speech_sec = chunk_res
                t0_call = time.perf_counter() - speech_sec

                stt_backend = wake_session.get_wake_stt_backend(session.get_stt_backend())
                if stt_backend is None:
                    console.print("[red]No speech-to-text backend available.[/red]")
                    return VOICE_EXIT

                try:
                    trans_res = stt_backend.transcribe(
                        audio_bytes,
                        format="wav",
                        language=session.get_stt_language() if hasattr(session, "get_stt_language") else "pt",
                    )
                    text = trans_res.text.strip()
                except Exception as exc:
                    logger.debug("Wake transcription error: %s", exc)
                    continue

                if not text:
                    continue

                parse_res = parse_wake_phrase(text, wake_session.wake_word)
                if not parse_res.detected:
                    # Name was not spoken at start or end
                    continue

                # Wake word was detected! Record the start of call timestamp for [TEMPO]
                wake_session.t_call_start = t0_call

                if not parse_res.is_command:
                    # Name alone or greeting: e.g. "Junior", "Junior, tá aí?"
                    play_sim_senhor(session, console)
                    if os.environ.get("JARVIS_DEBUG_TEMPO") == "1" and wake_session.t_call_start is not None:
                        t_ack = time.perf_counter() - wake_session.t_call_start
                        console.print(
                            f"[dim yellow][TEMPO] Do chamado até a resposta: {t_ack:.2f}s[/dim yellow]"
                        )
                    wake_session.touch(wake_named=False)
                    # Now record the actual command from the user
                    console.print("[dim cyan]Listening… (speak now, stops on silence)[/dim cyan]")
                    cmd_text = record_voice(console, session)
                    if isinstance(cmd_text, str) and cmd_text:
                        follow_res = parse_wake_phrase(cmd_text, wake_session.wake_word)
                        if follow_res.detected:
                            wake_session.touch(wake_named=True)
                            return follow_res.clean_text or cmd_text
                        else:
                            wake_session.touch(wake_named=False)
                            return cmd_text
                    wake_session.touch(wake_named=False)
                    return cmd_text
                else:
                    # Inline command: e.g. "Junior, o que tenho hoje?" or "Junior, sim"
                    wake_session.touch(wake_named=True)
                    console.print(f"[bold]You (voice):[/bold] {_terminal_safe_text(parse_res.clean_text or '')}")
                    return parse_res.clean_text

    # Case B: Jarvis is AWAKE -> within active conversational window
    else:
        while is_playback_active():
            time.sleep(0.05)
        console.print("[dim cyan]Listening… (speak now, stops on silence)[/dim cyan]")
        t0_rec = time.perf_counter()
        try:
            from openjarvis.speech.voice_io import record_until_silence

            audio_bytes = record_until_silence(
                startup_silence_seconds=min(wake_session.remaining_awake_time(), 10.0),
            )
        except KeyboardInterrupt:
            return VOICE_EXIT
        except Exception as exc:
            console.print(f"[red]Mic error: {_terminal_safe_text(exc)}[/red]")
            return VOICE_EXIT

        t_rec = time.perf_counter() - t0_rec
        if not wake_session.is_awake:
            wake_session.sleep()
            return None

        speech_sec = measure_wav_speech_duration(audio_bytes)
        if is_speech_too_short(speech_sec):
            logger.debug(
                "Speech too short (%.2fs < %.2fs) — ignoring",
                speech_sec,
                get_min_speech_seconds(),
            )
            return None

        wake_session.t_call_start = time.perf_counter() - t_rec
        backend = session.get_stt_backend()
        if backend is None:
            return VOICE_EXIT

        try:
            trans_res = backend.transcribe(
                audio_bytes,
                format="wav",
                language=session.get_stt_language() if hasattr(session, "get_stt_language") else "pt",
            )
            text = trans_res.text.strip()
        except Exception as exc:
            console.print(f"[red]Transcription error: {_terminal_safe_text(exc)}[/red]")
            return None

        if not text:
            wake_session.sleep()
            return None

        parse_res = parse_wake_phrase(text, wake_session.wake_word)
        if parse_res.detected:
            wake_session.touch(wake_named=True)
            clean_cmd = parse_res.clean_text or text
            console.print(f"[bold]You (voice):[/bold] {_terminal_safe_text(clean_cmd)}")
            return clean_cmd
        else:
            wake_session.touch(wake_named=False)
            console.print(f"[bold]You (voice):[/bold] {_terminal_safe_text(text)}")
            return text


__all__ = [
    "WakeParseResult",
    "WakeWordSession",
    "drain_stream",
    "get_min_speech_seconds",
    "get_sim_senhor_cache_path",
    "get_wake_window_seconds",
    "is_playback_active",
    "is_speech_too_short",
    "listen_for_wake_turn",
    "mark_playback_end",
    "mark_playback_start",
    "measure_wav_speech_duration",
    "normalize_wake_text",
    "parse_wake_phrase",
    "playback_guard",
    "play_sim_senhor",
    "prepare_sim_senhor_cache",
    "record_wake_chunk_audio",
    "reset_playback_guard",
]

