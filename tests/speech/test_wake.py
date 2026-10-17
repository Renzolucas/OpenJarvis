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
    get_sim_senhor_cache_path,
    get_wake_window_seconds,
    is_playback_active,
    is_speech_too_short,
    mark_playback_end,
    mark_playback_start,
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
    """Test short sound duration threshold (< 0.4s)."""

    def test_under_0_4s_is_too_short(self):
        self.assertTrue(is_speech_too_short(0.1))
        self.assertTrue(is_speech_too_short(0.25))
        self.assertTrue(is_speech_too_short(0.39))

    def test_at_or_above_0_4s_is_not_too_short(self):
        self.assertFalse(is_speech_too_short(0.40))
        self.assertFalse(is_speech_too_short(1.0))
        self.assertFalse(is_speech_too_short(3.5))


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


if __name__ == "__main__":
    unittest.main()
