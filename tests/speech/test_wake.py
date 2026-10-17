"""Unit tests for wake word detection, audio gating, anti-echo, and cache."""

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from openjarvis.speech.wake import (
    WakeParseResult,
    WakeWordSession,
    get_min_speech_seconds,
    get_sim_senhor_cache_path,
    get_wake_window_seconds,
    is_playback_active,
    is_speech_too_short,
    mark_playback_end,
    mark_playback_start,
    measure_wav_speech_duration,
    normalize_wake_text,
    parse_wake_phrase,
    playback_guard,
    play_sim_senhor,
    prepare_sim_senhor_cache,
    reset_playback_guard,
)


class TestWakePhraseParsing(unittest.TestCase):
    """Test wake word detection and command separation."""

    def test_wake_variants_alone_trigger_wake_without_command(self):
        cases = [
            "Junior",
            "júnior",
            "JÚNIOR",
            "junio",
            "Junio",
            "JR",
            "jr",
            "Ei Junior",
            "Olá, júnior!",
            "Junior, tá aí?",
            "Junior, está aí?",
            "Opa junior!",
            "Junior, você tá aí?",
            "Junior, vc tá aí?",
            "Junior, cê tá aí?",
            "Junior, você está aí?",
            "Junior, você tá me ouvindo?",
            "Junior, cê tá me ouvindo?",
            "Junior, você tá por aí?",
        ]
        for phrase in cases:
            with self.subTest(phrase=phrase):
                res = parse_wake_phrase(phrase)
                self.assertTrue(res.detected)
                self.assertTrue(res.wake_named)
                self.assertIsNone(res.clean_text)
                self.assertFalse(res.is_command)

    def test_wake_with_inline_command(self):
        cases = [
            ("Junior, o que tenho hoje?", "o que tenho hoje?"),
            ("júnior o que tenho hoje", "o que tenho hoje"),
            ("Junior, marca dentista amanhã às 15h", "marca dentista amanhã às 15h"),
            ("Ei, junior, como está o tempo?", "como está o tempo?"),
            ("Junior, sim", "sim"),
            ("Junior, confirma", "confirma"),
            ("Júnior, pode marcar", "pode marcar"),
            ("O que tenho hoje, Junior?", "O que tenho hoje"),
            ("Como está o tempo, júnior?", "Como está o tempo"),
            ("Sim, Junior", "Sim"),
        ]
        for input_phrase, expected_command in cases:
            with self.subTest(input_phrase=input_phrase):
                res = parse_wake_phrase(input_phrase)
                self.assertTrue(res.detected)
                self.assertTrue(res.wake_named)
                self.assertEqual(res.clean_text, expected_command)
                self.assertTrue(res.is_command)

    def test_middle_name_does_not_activate_wake(self):
        # Adjustment 1: "o Junior chegou" NÃO pode ativar nem virar comando
        invalid_cases = [
            "o Junior chegou",
            "o júnior chegou",
            "falei com o junior ontem",
            "meu amigo junior vai lá",
            "porque o junior disse isso",
        ]
        for phrase in invalid_cases:
            with self.subTest(phrase=phrase):
                res = parse_wake_phrase(phrase)
                self.assertFalse(res.detected)
                self.assertFalse(res.wake_named)
                self.assertEqual(res.clean_text, phrase)
                self.assertTrue(res.is_command)

    def test_phrase_without_wake_word(self):
        cases = [
            "o que tenho hoje?",
            "marca dentista amanhã às 15h",
            "sim",
            "confirma",
            "boa tarde",
        ]
        for phrase in cases:
            with self.subTest(phrase=phrase):
                res = parse_wake_phrase(phrase)
                self.assertFalse(res.detected)
                self.assertFalse(res.wake_named)
                self.assertEqual(res.clean_text, phrase)
                self.assertTrue(res.is_command)

    def test_empty_phrase(self):
        res = parse_wake_phrase("")
        self.assertFalse(res.detected)
        self.assertFalse(res.wake_named)
        self.assertIsNone(res.clean_text)
        self.assertFalse(res.is_command)


class TestShortSpeechFilter(unittest.TestCase):
    """Test short sound duration threshold (default 0.25s, configurable)."""

    def setUp(self):
        os.environ.pop("JARVIS_MIN_SPEECH_SECONDS", None)

    def tearDown(self):
        os.environ.pop("JARVIS_MIN_SPEECH_SECONDS", None)

    def test_default_min_speech_seconds(self):
        self.assertEqual(get_min_speech_seconds(), 0.25)

    def test_env_override_min_speech_seconds(self):
        os.environ["JARVIS_MIN_SPEECH_SECONDS"] = "0.35"
        self.assertEqual(get_min_speech_seconds(), 0.35)

    def test_invalid_env_falls_back_to_default(self):
        os.environ["JARVIS_MIN_SPEECH_SECONDS"] = "abc"
        self.assertEqual(get_min_speech_seconds(), 0.25)
        os.environ["JARVIS_MIN_SPEECH_SECONDS"] = "-1.0"
        self.assertEqual(get_min_speech_seconds(), 0.25)

    def test_speech_vs_snap(self):
        # Fala de ~0.3s passa
        self.assertFalse(is_speech_too_short(0.30))
        self.assertFalse(is_speech_too_short(0.50))
        # Estalo de ~0.1s não passa
        self.assertTrue(is_speech_too_short(0.10))
        self.assertTrue(is_speech_too_short(0.24))

    def test_measure_wav_speech_duration(self):
        # Empty audio
        self.assertEqual(measure_wav_speech_duration(b""), 0.0)

        # 1-second WAV of pure silence (all 0s)
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)
        self.assertEqual(measure_wav_speech_duration(buf.getvalue()), 0.0)

        # Estalo: 2 blocos de 512 amostras = 1024 bytes = 0.064s cercados de silêncio
        buf_snap = io.BytesIO()
        with wave.open(buf_snap, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1024)
            wf.writeframes(b"\x00\x20" * 1024)
            wf.writeframes(b"\x00\x00" * 1024)
        dur_snap = measure_wav_speech_duration(buf_snap.getvalue())
        self.assertAlmostEqual(dur_snap, 0.064, delta=0.02)
        self.assertTrue(is_speech_too_short(dur_snap))

        # Fala de ~0.32s: 10 blocos de 512 amostras = 5120 amostras = 0.32s cercados de silêncio
        buf_speech = io.BytesIO()
        with wave.open(buf_speech, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1024)
            wf.writeframes(b"\x00\x20" * 5120)
            wf.writeframes(b"\x00\x00" * 1024)
        dur_speech = measure_wav_speech_duration(buf_speech.getvalue())
        self.assertAlmostEqual(dur_speech, 0.32, delta=0.05)
        self.assertFalse(is_speech_too_short(dur_speech))


class TestAntiEchoPlaybackGuard(unittest.TestCase):
    """Test anti-echo muting while Jarvis speaks and for ~0.5s after."""

    def setUp(self):
        reset_playback_guard()

    def tearDown(self):
        reset_playback_guard()

    def test_guard_pauses_during_playback_and_cooldown(self):
        self.assertFalse(is_playback_active())

        with playback_guard(cooldown_seconds=0.1):
            self.assertTrue(is_playback_active())

        # Immediately after exit, cooldown is active
        self.assertTrue(is_playback_active())

        # After cooldown expires, inactive
        time.sleep(0.15)
        self.assertFalse(is_playback_active())


class TestWakeWindowAndSession(unittest.TestCase):
    """Test 10-second active conversational window and session state."""

    def test_default_wake_window_is_10s(self):
        os.environ.pop("JARVIS_WAKE_WINDOW", None)
        self.assertEqual(get_wake_window_seconds(), 10.0)

    def test_wake_window_env_override(self):
        os.environ["JARVIS_WAKE_WINDOW"] = "15.5"
        try:
            self.assertEqual(get_wake_window_seconds(), 15.5)
        finally:
            os.environ.pop("JARVIS_WAKE_WINDOW", None)

    def test_session_state_transitions(self):
        session = WakeWordSession(window_seconds=0.1)
        self.assertFalse(session.is_awake)
        self.assertFalse(session.last_wake_named)

        # Touch wakes it up
        session.touch(wake_named=True)
        self.assertTrue(session.is_awake)
        self.assertTrue(session.last_wake_named)

        # Wait for expiration
        time.sleep(0.15)
        self.assertFalse(session.is_awake)

        # Touch again and sleep immediately
        session.touch(wake_named=True)
        self.assertTrue(session.is_awake)
        session.sleep()
        self.assertFalse(session.is_awake)

    def test_remaining_awake_time(self):
        session = WakeWordSession(window_seconds=2.0)
        self.assertEqual(session.remaining_awake_time(), 0.0)

        session.touch(wake_named=True)
        rem = session.remaining_awake_time()
        self.assertTrue(1.5 < rem <= 2.0)

        session.sleep()
        self.assertEqual(session.remaining_awake_time(), 0.0)

    def test_touch_without_arguments_preserves_wake_named(self):
        session = WakeWordSession(window_seconds=10.0)
        session.touch(wake_named=True)
        self.assertTrue(session.last_wake_named)

        # Renew after assistant responds
        session.touch()
        self.assertTrue(session.last_wake_named)
        self.assertTrue(session.is_awake)

    def test_wake_stt_model_selection(self):
        session_default = WakeWordSession()
        mock_default = MagicMock()
        self.assertEqual(session_default.get_wake_stt_backend(mock_default), mock_default)

        # Dedicated model
        with patch("openjarvis.speech.faster_whisper.FasterWhisperBackend") as mock_fw_cls:
            mock_dedicated = MagicMock()
            mock_fw_cls.return_value = mock_dedicated

            session_dedicated = WakeWordSession(wake_model="base")
            backend = session_dedicated.get_wake_stt_backend(mock_default)
            mock_fw_cls.assert_called_once_with(model_size="base")
            self.assertEqual(backend, mock_dedicated)


class TestSimSenhorCache(unittest.TestCase):
    """Test 'Sim, senhor.' audio caching and fallback."""

    def setUp(self):
        reset_playback_guard()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.orig_home = os.environ.get("OPENJARVIS_HOME")
        os.environ["OPENJARVIS_HOME"] = self.temp_dir.name

    def tearDown(self):
        reset_playback_guard()
        if self.orig_home is not None:
            os.environ["OPENJARVIS_HOME"] = self.orig_home
        else:
            os.environ.pop("OPENJARVIS_HOME", None)

    def test_cache_path_generation(self):
        path = get_sim_senhor_cache_path("pt_BR-fabio")
        self.assertTrue(str(path).endswith("sim_senhor_pt_BR-fabio.wav"))

    def test_prepare_sim_senhor_cache_synthesizes_once_with_actual_voice(self):
        # Generates a valid dummy WAV header
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * 100)
        dummy_wav = buf.getvalue()

        mock_session = MagicMock()
        mock_backend = MagicMock()
        mock_result = MagicMock()
        mock_result.audio = dummy_wav
        mock_backend.synthesize.return_value = mock_result
        mock_session.get_tts_backend.return_value = mock_backend
        # Configured was 'want-voice', but actual backend substituted 'actual-voice'
        mock_session.voice_for_backend.return_value = ("actual-voice", 1.0)

        path, actual_voice = prepare_sim_senhor_cache(mock_session, "want-voice")
        self.assertEqual(actual_voice, "actual-voice")
        self.assertTrue(str(path).endswith("sim_senhor_actual-voice.wav"))
        self.assertTrue(path.exists())
        self.assertEqual(path.read_bytes(), dummy_wav)
        mock_backend.synthesize.assert_called_once()

        # Second call does not re-synthesize
        mock_backend.synthesize.reset_mock()
        path2, voice2 = prepare_sim_senhor_cache(mock_session, "want-voice")
        self.assertEqual(path2, path)
        self.assertEqual(voice2, "actual-voice")
        mock_backend.synthesize.assert_not_called()

    def test_play_sim_senhor_extracts_sample_rate_and_plays(self):
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * 100)
        dummy_wav = buf.getvalue()

        mock_session = MagicMock()
        mock_backend = MagicMock()
        mock_result = MagicMock()
        mock_result.audio = dummy_wav
        mock_backend.synthesize.return_value = mock_result
        mock_session.get_tts_backend.return_value = mock_backend
        mock_session.voice_for_backend.return_value = ("test-voice", 1.0)

        with patch("openjarvis.speech.voice_io.play_wav") as mock_play:
            played = play_sim_senhor(mock_session, voice_id="test-voice")
            self.assertTrue(played)
            # Sample rate extracted from header (22050) passed to play_wav
            mock_play.assert_called_once_with(dummy_wav, sample_rate=22050)

    def test_play_sim_senhor_fallback_to_text(self):
        mock_session = MagicMock()
        mock_session.get_tts_backend.return_value = None
        mock_console = MagicMock()

        played = play_sim_senhor(mock_session, console=mock_console, voice_id="none")
        self.assertFalse(played)
        mock_console.print.assert_called_once_with("[dim cyan]Sim, senhor.[/dim cyan]")


class TestListenForWakeTurn(unittest.TestCase):
    """Test listen_for_wake_turn flow for both sleeping and awake states."""

    def setUp(self):
        reset_playback_guard()

    def tearDown(self):
        reset_playback_guard()

    @patch("openjarvis.speech.wake.record_wake_chunk_audio")
    def test_inline_command_while_sleeping(self, mock_record):
        from openjarvis.speech.wake import WakeWordSession, listen_for_wake_turn

        mock_record.return_value = (b"dummy_wav", 1.2)
        mock_session = MagicMock()
        mock_stt = MagicMock()
        mock_res = MagicMock()
        mock_res.text = "Junior, que horas são?"
        mock_stt.transcribe.return_value = mock_res
        mock_session.get_stt_backend.return_value = mock_stt

        console = MagicMock()
        wake_session = WakeWordSession(wake_word="junior")
        self.assertFalse(wake_session.is_awake)

        result = listen_for_wake_turn(console, mock_session, wake_session)
        self.assertEqual(result, "que horas são?")
        self.assertTrue(wake_session.is_awake)
        self.assertTrue(wake_session.last_wake_named)

    @patch("openjarvis.speech.wake.record_wake_chunk_audio")
    @patch("openjarvis.speech.wake.play_sim_senhor")
    @patch("openjarvis.cli._voice_chat.record_voice")
    def test_call_alone_triggers_sim_senhor_then_record(self, mock_rec_voice, mock_play, mock_chunk):
        from openjarvis.speech.wake import WakeWordSession, listen_for_wake_turn

        mock_chunk.return_value = (b"dummy_wav", 0.8)
        mock_session = MagicMock()
        mock_stt = MagicMock()
        mock_res = MagicMock()
        mock_res.text = "Junior, você tá aí?"
        mock_stt.transcribe.return_value = mock_res
        mock_session.get_stt_backend.return_value = mock_stt
        mock_rec_voice.return_value = "criar evento"

        console = MagicMock()
        wake_session = WakeWordSession(wake_word="junior")
        result = listen_for_wake_turn(console, mock_session, wake_session)

        mock_play.assert_called_once_with(mock_session, console)
        mock_rec_voice.assert_called_once_with(console, mock_session)
        self.assertEqual(result, "criar evento")
        self.assertTrue(wake_session.is_awake)
        self.assertTrue(wake_session.last_wake_named)


if __name__ == "__main__":
    unittest.main()
