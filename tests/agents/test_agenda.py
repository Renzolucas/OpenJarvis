"""Unit tests for AgendaAgent with mocked Google Calendar."""

from __future__ import annotations

from datetime import date, datetime, timezone
import os
import unittest
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import httpx

from openjarvis.agents._stubs import AgentContext, AgentResult
from openjarvis.agents.agenda import (
    AgendaAgent,
    format_agenda_phrase,
    is_agenda_request,
    is_ignored_calendar,
    is_ignored_event,
    is_tomorrow_request,
)
from openjarvis.core.registry import AgentRegistry

_TZ = ZoneInfo("America/Fortaleza")


def _make_mock_engine(content: str = "Olá! Como posso ajudar?") -> MagicMock:
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {
        "content": content,
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        "model": "gemma4:e2b",
        "finish_reason": "stop",
    }
    return engine


class TestAgendaIntent(unittest.TestCase):
    """Test intent detection and keywords."""

    def test_is_agenda_request_positive(self):
        phrases = [
            "resuma meu dia",
            "resuma o meu dia",
            "resumir o dia de hoje",
            "o que tenho hoje",
            "o que eu tenho hoje?",
            "o que temos para hoje",
            "o que há pra hoje",
            "o que tenho agendado",
            "o que tenho programado",
            "minha agenda de amanhã",
            "minha agenda hoje",
            "qual a minha agenda",
            "agenda de hoje",
            "agenda de amanhã",
            "meus compromissos",
            "compromissos de hoje",
            "meus eventos",
            "eventos de hoje",
            "programação de hoje",
            "programacao de amanhã",
            "agenda",
        ]
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertTrue(is_agenda_request(phrase), f"Failed for: {phrase}")

    def test_is_agenda_request_negative(self):
        phrases = [
            "como está o tempo hoje?",
            "qual a capital da França?",
            "toque uma música",
            "olá jarvis, tudo bem?",
            "conte uma piada",
            "quem foi Santos Dumont?",
        ]
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertFalse(is_agenda_request(phrase), f"Should not match: {phrase}")

    def test_is_tomorrow_request(self):
        self.assertTrue(is_tomorrow_request("minha agenda de amanhã"))
        self.assertTrue(is_tomorrow_request("o que tenho para amanhã?"))
        self.assertTrue(is_tomorrow_request("agenda de amanha"))
        self.assertFalse(is_tomorrow_request("resuma meu dia de hoje"))
        self.assertFalse(is_tomorrow_request("o que tenho hoje"))


class TestAgendaCalendarFiltering(unittest.TestCase):
    """Test birthday and holiday calendar / event filters strictly by origin."""

    def test_birthday_calendars_ignored_by_id(self):
        self.assertTrue(is_ignored_calendar("addressbook#contacts@group.v.calendar.google.com"))
        # Summary/description alone must NOT ignore normal calendars
        self.assertFalse(is_ignored_calendar("user@gmail.com", summary="Aniversários"))
        self.assertFalse(is_ignored_calendar("cal123", description="Lista de aniversários"))

    def test_holiday_calendars_ignored_by_id(self):
        self.assertTrue(is_ignored_calendar("pt.brazilian#holiday@group.v.calendar.google.com"))
        self.assertTrue(is_ignored_calendar("en.usa#holiday@group.v.calendar.google.com"))
        # Summary alone must NOT ignore normal calendars
        self.assertFalse(is_ignored_calendar("cal456", summary="Feriados no Brasil"))

    def test_normal_calendar_not_ignored(self):
        self.assertFalse(is_ignored_calendar("primary", summary="Principal"))
        self.assertFalse(is_ignored_calendar("user@gmail.com", summary="Trabalho"))

    def test_ignored_events_by_event_type_only(self):
        # Only eventType "birthday" is ignored
        self.assertTrue(is_ignored_event("Qualquer título", event_type="birthday"))
        self.assertTrue(is_ignored_event(event_type="birthday"))

        # User events in normal calendars must NOT be ignored even if title contains aniversário or feriado
        self.assertFalse(is_ignored_event("Festa de aniversário"))
        self.assertFalse(is_ignored_event("Festa de aniversário da Ana"))
        self.assertFalse(is_ignored_event("Feriado com a família"))
        self.assertFalse(is_ignored_event("Almoço com cliente"))



class TestAgendaPhraseFormatting(unittest.TestCase):
    """Test phrase formatting in Brazilian Portuguese."""

    def test_no_events_today(self):
        phrase = format_agenda_phrase([], is_tomorrow=False)
        self.assertEqual(phrase, "Você não tem nenhum compromisso na agenda hoje.")

    def test_no_events_tomorrow(self):
        phrase = format_agenda_phrase([], is_tomorrow=True)
        self.assertEqual(phrase, "Você não tem nenhum compromisso na agenda amanhã.")

    def test_one_timed_event(self):
        events = [{"title": "Dentista", "is_all_day": False, "time_str": "10:00"}]
        phrase = format_agenda_phrase(events, is_tomorrow=False)
        self.assertEqual(phrase, "Para hoje, você tem 1 compromisso: às 10:00, Dentista.")

    def test_two_timed_events(self):
        events = [
            {"title": "Daily", "is_all_day": False, "time_str": "09:30"},
            {"title": "Almoço", "is_all_day": False, "time_str": "12:30"},
        ]
        phrase = format_agenda_phrase(events, is_tomorrow=False)
        self.assertEqual(
            phrase,
            "Para hoje, você tem 2 compromissos: às 09:30, Daily e às 12:30, Almoço.",
        )

    def test_three_events(self):
        events = [
            {"title": "Daily", "is_all_day": False, "time_str": "09:30"},
            {"title": "Reunião de Projeto", "is_all_day": False, "time_str": "14:00"},
            {"title": "Academia", "is_all_day": False, "time_str": "18:00"},
        ]
        phrase = format_agenda_phrase(events, is_tomorrow=False)
        self.assertEqual(
            phrase,
            "Para hoje, você tem 3 compromissos: às 09:30, Daily; às 14:00, Reunião de Projeto; e às 18:00, Academia.",
        )

    def test_all_day_event(self):
        events = [{"title": "Viagem", "is_all_day": True, "time_str": ""}]
        phrase = format_agenda_phrase(events, is_tomorrow=True)
        self.assertEqual(phrase, "Para amanhã, você tem 1 compromisso: o dia todo, Viagem.")


class TestAgendaAgentExecution(unittest.TestCase):
    """Test full agent execution with mocked GCalendar calls."""

    def setUp(self):
        self.engine = _make_mock_engine()
        self.agent = AgendaAgent(self.engine, "gemma4:e2b", timezone="America/Fortaleza")

    def test_agent_registered(self):
        self.assertTrue(AgentRegistry.contains("agenda"))
        cls = AgentRegistry.get("agenda")
        self.assertIs(cls, AgendaAgent)
        self.assertEqual(self.agent.agent_id, "agenda")

    @patch("openjarvis.connectors.google_auth.call_with_refresh")
    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=True)
    def test_fetch_events_day_window_and_birthday_filter(self, mock_is_conn, mock_call):
        # Target date: 2026-10-14
        target_date = date(2026, 10, 14)

        calendars_resp = {
            "items": [
                {"id": "primary", "summary": "Principal"},
                {"id": "pt.brazilian#holiday@group.v.calendar.google.com", "summary": "Feriados no Brasil"},
                {"id": "addressbook#contacts@group.v.calendar.google.com", "summary": "Aniversários"},
            ]
        }

        # Events in the primary calendar:
        # 1. Event on today (10:00 local America/Fortaleza) -> should be included
        # 2. "Festa de aniversário da Ana" on today (14:00) -> should be included
        # 3. System birthday event with eventType="birthday" -> should be excluded
        # 4. All day event on today (2026-10-14) -> should be included
        # 5. Event on yesterday -> should be excluded
        # 6. Event on tomorrow -> should be excluded
        events_resp = {
            "items": [
                {
                    "id": "evt_yesterday",
                    "summary": "Compromisso de ontem",
                    "start": {"dateTime": "2026-10-13T14:00:00-03:00"},
                    "end": {"dateTime": "2026-10-13T15:00:00-03:00"},
                },
                {
                    "id": "evt_today_1",
                    "summary": "Reunião de Alinhamento",
                    "start": {"dateTime": "2026-10-14T10:00:00-03:00"},
                    "end": {"dateTime": "2026-10-14T11:00:00-03:00"},
                },
                {
                    "id": "evt_festa",
                    "summary": "Festa de aniversário da Ana",
                    "start": {"dateTime": "2026-10-14T14:00:00-03:00"},
                    "end": {"dateTime": "2026-10-14T18:00:00-03:00"},
                },
                {
                    "id": "evt_birthday_type",
                    "summary": "Lembrete do Sistema",
                    "eventType": "birthday",
                    "start": {"dateTime": "2026-10-14T11:00:00-03:00"},
                    "end": {"dateTime": "2026-10-14T12:00:00-03:00"},
                },
                {
                    "id": "evt_all_day",
                    "summary": "Planejamento Trimestral",
                    "start": {"date": "2026-10-14"},
                    "end": {"date": "2026-10-15"},
                },
                {
                    "id": "evt_tomorrow",
                    "summary": "Compromisso de amanhã",
                    "start": {"dateTime": "2026-10-15T09:00:00-03:00"},
                    "end": {"dateTime": "2026-10-15T10:00:00-03:00"},
                },
            ]
        }

        # Return calendars_resp for first call, then events_resp for primary calendar
        mock_call.side_effect = [calendars_resp, events_resp]

        events = self.agent._fetch_events_for_date(target_date, _TZ)

        # Expected: evt_all_day (all day), evt_today_1 (10:00), and evt_festa (14:00)
        self.assertEqual(len(events), 3)
        titles = [e["title"] for e in events]
        self.assertIn("Planejamento Trimestral", titles)
        self.assertIn("Reunião de Alinhamento", titles)
        self.assertIn("Festa de aniversário da Ana", titles)
        self.assertNotIn("Lembrete do Sistema", titles)
        self.assertNotIn("Compromisso de ontem", titles)
        self.assertNotIn("Compromisso de amanhã", titles)

    @patch("openjarvis.connectors.google_auth.call_with_refresh")
    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=True)
    def test_festa_de_aniversario_in_primary_calendar_is_included(self, mock_is_conn, mock_call):
        """User events with 'Festa de aniversário' in primary calendar must always appear."""
        target_date = date(2026, 10, 14)
        calendars_resp = {"items": [{"id": "primary", "summary": "Principal"}]}
        events_resp = {
            "items": [
                {
                    "id": "evt_festa",
                    "summary": "Festa de aniversário",
                    "start": {"dateTime": "2026-10-14T19:00:00-03:00"},
                    "end": {"dateTime": "2026-10-14T23:00:00-03:00"},
                }
            ]
        }
        mock_call.side_effect = [calendars_resp, events_resp]

        events = self.agent._fetch_events_for_date(target_date, _TZ)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["title"], "Festa de aniversário")
        self.assertEqual(events[0]["time_str"], "19:00")
        self.assertFalse(events[0]["is_all_day"])


    @patch("openjarvis.connectors.google_auth.call_with_refresh")
    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=True)
    def test_max_five_events_limit(self, mock_is_conn, mock_call):
        target_date = date(2026, 10, 14)
        calendars_resp = {"items": [{"id": "primary", "summary": "Principal"}]}
        items = []
        for hour in range(8, 16):  # 8 events from 08:00 to 15:00
            items.append({
                "id": f"evt_{hour}",
                "summary": f"Evento {hour:02d}h",
                "start": {"dateTime": f"2026-10-14T{hour:02d}:00:00-03:00"},
                "end": {"dateTime": f"2026-10-14T{hour+1:02d}:00:00-03:00"},
            })
        events_resp = {"items": items}
        mock_call.side_effect = [calendars_resp, events_resp]

        events = self.agent._fetch_events_for_date(target_date, _TZ)
        self.assertEqual(len(events), 5)
        self.assertEqual(events[0]["title"], "Evento 08h")
        self.assertEqual(events[4]["title"], "Evento 12h")

    @patch("openjarvis.connectors.google_auth.call_with_refresh")
    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=True)
    def test_run_agenda_request_does_not_call_llm(self, mock_is_conn, mock_call):
        calendars_resp = {"items": [{"id": "primary", "summary": "Principal"}]}
        events_resp = {
            "items": [
                {
                    "id": "evt_1",
                    "summary": "Reunião de Alinhamento",
                    "start": {"dateTime": datetime.now(_TZ).strftime("%Y-%m-%dT10:00:00-03:00")},
                    "end": {"dateTime": datetime.now(_TZ).strftime("%Y-%m-%dT11:00:00-03:00")},
                }
            ]
        }
        mock_call.side_effect = [calendars_resp, events_resp]

        result = self.agent.run("resuma meu dia")
        self.assertIsInstance(result, AgentResult)
        self.assertIn("Para hoje", result.content)
        self.assertIn("Reunião de Alinhamento", result.content)
        # LLM generate should not be called when code formats safely
        self.engine.generate.assert_not_called()

    def test_run_non_agenda_request_calls_llm(self):
        result = self.agent.run("olá jarvis, conte uma curiosidade")
        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.content, "Olá! Como posso ajudar?")
        self.engine.generate.assert_called_once()

    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=False)
    def test_run_calendar_not_connected(self, mock_is_conn):
        result = self.agent.run("o que tenho hoje")
        self.assertIsInstance(result, AgentResult)
        self.assertIn("Google Calendar não está conectado", result.content)


class TestAgendaCreation(unittest.TestCase):
    """Test voice event creation, confirmation state machine, dry-run, and 403 handling."""

    def setUp(self):
        self.engine = _make_mock_engine()
        # Thursday, 2026-10-15 13:00 America/Fortaleza
        self.fixed_now = datetime(2026, 10, 15, 13, 0, tzinfo=_TZ)
        self.current_monotonic = 1000.0

        def clock_fn():
            return self.fixed_now

        def monotonic_fn():
            return self.current_monotonic

        self.agent = AgendaAgent(
            self.engine,
            "gemma4:e2b",
            timezone="America/Fortaleza",
            voice_mode=True,
            clock=clock_fn,
            monotonic=monotonic_fn,
        )

    def test_creation_in_ask_mode_warns_voice_chat_only(self):
        # Non-voice mode agent (like jarvis ask)
        ask_agent = AgendaAgent(
            self.engine,
            "gemma4:e2b",
            timezone="America/Fortaleza",
            voice_mode=False,
            clock=lambda: self.fixed_now,
        )
        res = ask_agent.run("marca dentista amanhã às 15h")
        self.assertEqual(res.content, "A criação de eventos só funciona no chat --voice.")

    def test_sim_without_pending_action_calls_llm(self):
        res = self.agent.run("sim")
        self.assertEqual(res.content, "Olá! Como posso ajudar?")
        self.engine.generate.assert_called_once()

    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=True)
    @patch("openjarvis.agents.agenda.check_duplicate_event", return_value=None)
    @patch("openjarvis.agents.agenda.create_calendar_event")
    def test_event_creation_full_flow_with_sim(self, mock_create, mock_dup, mock_is_conn):
        # Turn 1: create request
        res1 = self.agent.run("marca dentista amanhã às 15h")
        self.assertEqual(res1.content, "Marcar Dentista para sexta, 16/10, 15h? Confirma?")
        mock_create.assert_not_called()

        # Turn 2: confirm with "sim"
        res2 = self.agent.run("sim")
        self.assertEqual(res2.content, "Pronto: Dentista, sexta 16/10, 15h.")
        mock_create.assert_called_once()
        payload = mock_create.call_args[0][0]
        self.assertEqual(payload["summary"], "Dentista")
        self.assertEqual(payload["start"]["dateTime"], "2026-10-16T15:00:00-03:00")
        self.assertEqual(payload["end"]["dateTime"], "2026-10-16T16:00:00-03:00")
        self.assertIn("reminders", payload)

    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=True)
    @patch("openjarvis.agents.agenda.check_duplicate_event", return_value=None)
    @patch("openjarvis.agents.agenda.create_calendar_event")
    def test_affirmative_variants(self, mock_create, mock_dup, mock_is_conn):
        variants = ["confirma", "pode marcar", "isso", "Sim!"]
        for var in variants:
            with self.subTest(var=var):
                self.agent._pending_event = None
                self.agent.run("marca dentista amanhã às 15h")
                res = self.agent.run(var)
                self.assertEqual(res.content, "Pronto: Dentista, sexta 16/10, 15h.")

    def test_non_affirmative_cancels_and_warns(self):
        non_affirmatives = [
            "sim, mas queria às 16h",
            "não",
            "cancela",
            "espera um pouco",
            "sim, por favor",
            "vamos ver",
        ]
        for phr in non_affirmatives:
            with self.subTest(phr=phr):
                self.agent._pending_event = None
                res1 = self.agent.run("marca dentista amanhã às 15h")
                self.assertIn("Confirma?", res1.content)

                res2 = self.agent.run(phr)
                self.assertEqual(res2.content, "Marcação cancelada.")
                self.assertIsNone(self.agent._pending_event)

    def test_expiration_after_two_minutes(self):
        # Turn 1: create request
        self.agent.run("marca dentista amanhã às 15h")
        self.assertIsNotNone(self.agent._pending_event)

        # Advance monotonic clock by 125 seconds (> 2 minutes)
        self.current_monotonic += 125.0

        # Turn 2: user says "sim", but it expired!
        res = self.agent.run("sim")
        # With no pending action, "sim" does nothing and falls through to normal conversation
        self.assertEqual(res.content, "Olá! Como posso ajudar?")
        self.assertIsNone(self.agent._pending_event)

    def test_ambiguous_time_asks_and_resolves(self):
        # Turn 1: ambiguous hour 3 (no morning/afternoon)
        res1 = self.agent.run("marca dentista amanhã às 3")
        self.assertEqual(res1.content, "Você prefere às 3 da manhã ou às 3 da tarde?")

        # Turn 2: user clarifies "da tarde"
        res2 = self.agent.run("da tarde")
        self.assertEqual(res2.content, "Marcar Dentista para sexta, 16/10, 15h? Confirma?")

        # Turn 3: confirms
        with patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=True), \
             patch("openjarvis.agents.agenda.check_duplicate_event", return_value=None), \
             patch("openjarvis.agents.agenda.create_calendar_event") as mock_create:
            res3 = self.agent.run("sim")
            self.assertEqual(res3.content, "Pronto: Dentista, sexta 16/10, 15h.")
            payload = mock_create.call_args[0][0]
            self.assertEqual(payload["start"]["dateTime"], "2026-10-16T15:00:00-03:00")

    def test_past_time_today_asks_and_never_creates(self):
        # Today is 13:00. Time 10:00 da manhã has passed today.
        res = self.agent.run("marca reunião hoje às 10 da manhã")
        self.assertEqual(res.content, "Esse horário já passou hoje. Para qual horário você gostaria de marcar?")
        self.assertNotEqual(getattr(self.agent._pending_event, "awaiting", ""), "confirmation")

    def test_weekday_equal_to_today_uses_next_week(self):
        # Today is Thursday (quinta). User asks "na quinta às 15h" -> next week (22/10)
        res = self.agent.run("marca reunião na quinta às 15h")
        self.assertEqual(res.content, "Marcar Reunião para quinta, 22/10, 15h? Confirma?")

    def test_missing_title_asks_for_title(self):
        # User doesn't provide title
        res1 = self.agent.run("marca amanhã às 15h")
        self.assertEqual(res1.content, "Qual é o título do evento?")

        # Turn 2: provides title
        res2 = self.agent.run("Dentista")
        self.assertEqual(res2.content, "Marcar Dentista para sexta, 16/10, 15h? Confirma?")

    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=True)
    @patch("openjarvis.agents.agenda.check_duplicate_event")
    @patch("openjarvis.agents.agenda.create_calendar_event")
    def test_duplicate_event_detected_cancels_creation(self, mock_create, mock_dup, mock_is_conn):
        # Existing event on same day and time
        mock_dup.return_value = {"id": "dup_1", "summary": "Dentista"}

        self.agent.run("marca dentista amanhã às 15h")
        res = self.agent.run("sim")

        self.assertIn("Já existe um compromisso 'Dentista' agendado para", res.content)
        mock_create.assert_not_called()

    def test_dry_run_environment_variable(self):
        os.environ["JARVIS_AGENDA_DRY_RUN"] = "1"
        try:
            self.agent.run("marca dentista amanhã às 15h")
            with patch("openjarvis.agents.agenda.create_calendar_event") as mock_create:
                res = self.agent.run("sim")
                self.assertEqual(res.content, "Simulação: Dentista, sexta 16/10, 15h. Nada foi criado.")
                mock_create.assert_not_called()
        finally:
            os.environ.pop("JARVIS_AGENDA_DRY_RUN", None)

    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=True)
    @patch("openjarvis.agents.agenda.check_duplicate_event", return_value=None)
    @patch("openjarvis.agents.agenda.create_calendar_event")
    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.disconnect")
    def test_http_403_forbidden_handling_on_create(self, mock_disconnect, mock_create, mock_dup, mock_is_conn):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_create.side_effect = httpx.HTTPStatusError("Forbidden", request=MagicMock(), response=mock_response)

        self.agent.run("marca dentista amanhã às 15h")
        res = self.agent.run("sim")

        # Must say write permission is missing and how to redo login
        self.assertIn("Falta permissão de escrita no Google Calendar", res.content)
        self.assertIn("jarvis connect gcalendar", res.content)
        mock_disconnect.assert_not_called()

    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=True)
    @patch("openjarvis.agents.agenda.check_duplicate_event")
    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.disconnect")
    def test_http_403_forbidden_handling_on_check_duplicate(self, mock_disconnect, mock_dup, mock_is_conn):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_dup.side_effect = httpx.HTTPStatusError("Forbidden", request=MagicMock(), response=mock_response)

        self.agent.run("marca dentista amanhã às 15h")
        res = self.agent.run("sim")

        self.assertIn("Falta permissão de escrita no Google Calendar", res.content)
        self.assertIn("jarvis connect gcalendar", res.content)
        mock_disconnect.assert_not_called()

    def test_retry_on_unrecognized_time_asks_once_then_cancels(self):
        # Requirement 2: when pending state expects time and response not understood, ask once then cancel
        self.agent.run("marca dentista amanhã")
        # 1st unrecognized response
        res1 = self.agent.run("banana")
        self.assertEqual(res1.content, "Não entendi. Pode repetir o horário?")
        self.assertIsNotNone(self.agent._pending_event)

        # 2nd unrecognized response -> cancels
        res2 = self.agent.run("laranja")
        self.assertEqual(res2.content, "Marcação cancelada.")
        self.assertIsNone(self.agent._pending_event)

    def test_retry_on_unrecognized_time_succeeds_on_second_turn(self):
        self.agent.run("marca dentista amanhã")
        res1 = self.agent.run("não sei")
        self.assertEqual(res1.content, "Não entendi. Pode repetir o horário?")

        res2 = self.agent.run("às 15h")
        self.assertEqual(res2.content, "Marcar Dentista para sexta, 16/10, 15h? Confirma?")

    def test_explicit_cancel_words_in_pending_state(self):
        # Requirement 2: cancel immediately on "não", "cancela", "deixa pra lá"
        for cancel_word in ["não", "cancela", "deixa pra lá", "deixa pra la"]:
            with self.subTest(cancel_word=cancel_word):
                self.agent._pending_event = None
                self.agent.run("marca dentista amanhã")
                res = self.agent.run(cancel_word)
                self.assertEqual(res.content, "Marcação cancelada.")
                self.assertIsNone(self.agent._pending_event)

    def test_ambiguous_time_retry_then_resolves(self):
        # Requirement 1 & 2: "marca dentista amanhã às três" -> ambiguous
        res1 = self.agent.run("marca dentista amanhã às três")
        self.assertEqual(res1.content, "Você prefere às 3 da manhã ou às 3 da tarde?")

        # Unrecognized turn
        res2 = self.agent.run("como assim?")
        self.assertEqual(res2.content, "Não entendi. Pode repetir o horário?")

        # Resolves with just the period
        res3 = self.agent.run("da tarde")
        self.assertEqual(res3.content, "Marcar Dentista para sexta, 16/10, 15h? Confirma?")

    def test_ambiguous_time_period_only_variants(self):
        # Requirement 2: "Uma resposta só com o período ('da tarde', 'de manhã', 'à noite') resolve a ambiguidade pendente."
        self.agent._pending_event = None
        self.agent.run("marca reunião amanhã às 10")
        res = self.agent.run("de manhã")
        self.assertEqual(res.content, "Marcar Reunião para sexta, 16/10, 10h? Confirma?")

        self.agent._pending_event = None
        self.agent.run("marca reunião amanhã às 10")
        res = self.agent.run("à noite")
        self.assertEqual(res.content, "Marcar Reunião para sexta, 16/10, 22h? Confirma?")

    def test_missing_title_retry_and_cancellation(self):
        self.agent._pending_event = None
        res1 = self.agent.run("marca amanhã às 15h")
        self.assertEqual(res1.content, "Qual é o título do evento?")

        # Unrecognized / empty title
        res2 = self.agent.run("por favor")
        self.assertEqual(res2.content, "Não entendi. Pode repetir o título?")

        # Cancel on "cancela"
        res3 = self.agent.run("cancela")
        self.assertEqual(res3.content, "Marcação cancelada.")
        self.assertIsNone(self.agent._pending_event)

    def test_wake_mode_confirmation_without_wake_named_asks_for_security(self):
        self.agent._pending_event = None
        self.agent.run("marca dentista amanhã às 15h")
        self.assertIsNotNone(self.agent._pending_event)

        ctx_unnamed = AgentContext()
        ctx_unnamed.metadata["wake_mode"] = True
        ctx_unnamed.metadata["wake_named"] = False

        res = self.agent.run("sim", context=ctx_unnamed)
        self.assertEqual(res.content, "Por segurança, confirme dizendo 'Junior, sim'.")
        self.assertIsNotNone(self.agent._pending_event)
        self.assertEqual(self.agent._pending_event.awaiting, "confirmation")

    @patch("openjarvis.connectors.gcalendar.GCalendarConnector.is_connected", return_value=True)
    @patch("openjarvis.agents.agenda.check_duplicate_event", return_value=None)
    @patch("openjarvis.agents.agenda.create_calendar_event")
    def test_wake_mode_confirmation_with_wake_named_creates_event(self, mock_create, mock_dup, mock_is_conn):
        self.agent._pending_event = None
        self.agent.run("marca dentista amanhã às 15h")

        ctx_named = AgentContext()
        ctx_named.metadata["wake_mode"] = True
        ctx_named.metadata["wake_named"] = True

        res = self.agent.run("sim", context=ctx_named)
        self.assertEqual(res.content, "Pronto: Dentista, sexta 16/10, 15h.")
        mock_create.assert_called_once()
        self.assertIsNone(self.agent._pending_event)


if __name__ == "__main__":
    unittest.main()
