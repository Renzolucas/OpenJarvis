"""AgendaAgent — voice-friendly daily schedule summary and event creation agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import json
import logging
import os
import re
import time as time_mod
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.agents.agenda_parse import (
    WEEKDAYS_PT,
    build_event_payload,
    format_confirmation_prompt,
    is_create_intent,
    is_short_affirmative,
    parse_create_event,
    parse_time,
    parse_title,
)
from openjarvis.connectors.gcalendar_actions import (
    check_duplicate_event,
    create_calendar_event,
)
from openjarvis.core.registry import AgentRegistry

logger = logging.getLogger(__name__)

_DEFAULT_TIMEZONE = "America/Fortaleza"
_DEFAULT_SYSTEM_PROMPT = (
    "Você é um assistente pessoal prestativo e amigável. "
    "Responda sempre em português do Brasil, de forma concisa e natural para voz. "
    "Nunca use o termo 'Sir' nem formalidades excessivas."
)

_IGNORED_CALENDAR_ID_SUBSTRINGS = (
    "#contacts@group.v.calendar.google.com",
    "#holiday",
)


@dataclass
class PendingEventState:
    """In-memory state for an unconfirmed event creation request."""

    title: Optional[str] = None
    target_date: Optional[date] = None
    target_time: Optional[time] = None
    target_dt: Optional[datetime] = None
    ambiguous_hour: Optional[int] = None
    ambiguous_minute: int = 0
    awaiting: str = "confirmation"  # "confirmation", "title", "time", "ambiguous_time"
    created_at: float = field(default_factory=time_mod.monotonic)
    turns_alive: int = 0
    retry_count: int = 0


def is_explicit_cancel(text: str) -> bool:
    """Return True if text is an explicit cancellation command."""
    cleaned = re.sub(r"[.!?,;:~]+$", "", text.strip().lower()).strip()
    if cleaned in {
        "não",
        "nao",
        "cancela",
        "cancelar",
        "esquece",
        "deixa pra lá",
        "deixa pra la",
        "não quero mais",
        "nao quero mais",
        "deixa",
    }:
        return True
    cancel_words = [
        "cancela",
        "cancelar",
        "esquece",
        "deixa pra lá",
        "deixa pra la",
        "não quero mais",
        "nao quero mais",
    ]
    return any(w in cleaned for w in cancel_words)


def is_agenda_request(text: str) -> bool:
    """Return True if the text asks for a daily agenda/calendar summary."""
    t = text.lower()
    patterns = [
        r"\bresum(a|e|ir|o)\b.*\b(dia|hoje|amanh[aã]|agenda)\b",
        r"\b(o que|qual)\b.*(tenho|temos|tem|h[aá]).*\b(hoje|amanh[aã]|programado|agendado)\b",
        r"\bminha agenda\b",
        r"\bagenda\b.*\b(de hoje|de amanh[aã]|para hoje|para amanh[aã]|pra hoje|pra amanh[aã])\b",
        r"\b(meus )?compromissos\b",
        r"\b(meus )?eventos\b",
        r"\bprograma[cç][aã]o\b.*\b(de hoje|de amanh[aã]|hoje|amanh[aã])\b",
        r"^\s*agenda\s*$",
    ]
    return any(re.search(p, t) for p in patterns)


def is_tomorrow_request(text: str) -> bool:
    """Return True if the request specifically refers to tomorrow."""
    t = text.lower()
    return bool(re.search(r"\bamanh[aã]\b", t))


def is_ignored_calendar(cal_id: str, summary: str = "", description: str = "") -> bool:
    """Check if calendar is a birthday or holiday calendar by its ID."""
    cal_id_lower = cal_id.lower()
    return any(sub in cal_id_lower for sub in _IGNORED_CALENDAR_ID_SUBSTRINGS)


def is_ignored_event(title: str = "", event_type: str = "") -> bool:
    """Check if an individual event is a system birthday event by eventType."""
    return event_type.lower() == "birthday"


def format_agenda_phrase(events: List[Dict[str, Any]], is_tomorrow: bool) -> str:
    """Format the list of events into a concise spoken sentence in Portuguese."""
    day_label = "amanhã" if is_tomorrow else "hoje"
    if not events:
        return f"Você não tem nenhum compromisso na agenda {day_label}."

    items_str: List[str] = []
    for ev in events:
        title = ev.get("title", "(Sem título)")
        if ev.get("is_all_day"):
            items_str.append(f"o dia todo, {title}")
        else:
            time_str = ev.get("time_str", "")
            if time_str:
                items_str.append(f"às {time_str}, {title}")
            else:
                items_str.append(title)

    count = len(events)
    termo = "compromisso" if count == 1 else "compromissos"
    if count == 1:
        return f"Para {day_label}, você tem 1 {termo}: {items_str[0]}."
    elif count == 2:
        return f"Para {day_label}, você tem 2 {termo}: {items_str[0]} e {items_str[1]}."
    else:
        all_but_last = "; ".join(items_str[:-1])
        return f"Para {day_label}, você tem {count} {termo}: {all_but_last}; e {items_str[-1]}."


@AgentRegistry.register("agenda")
class AgendaAgent(BaseAgent):
    """Personal agenda agent that summarizes and creates Google Calendar events in Portuguese."""

    agent_id = "agenda"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._timezone_name = kwargs.pop("timezone", _DEFAULT_TIMEZONE)
        self._voice_mode: bool = kwargs.pop("voice_mode", False)
        self._clock_fn: Optional[Callable[[], datetime]] = kwargs.pop("clock", None)
        self._monotonic_fn: Optional[Callable[[], float]] = kwargs.pop("monotonic", None)
        self._pending_event: Optional[PendingEventState] = None
        super().__init__(*args, **kwargs)

    def _get_tz(self) -> ZoneInfo:
        try:
            return ZoneInfo(self._timezone_name)
        except Exception:
            return ZoneInfo(_DEFAULT_TIMEZONE)

    def _get_now(self) -> datetime:
        tz = self._get_tz()
        if self._clock_fn is not None:
            res = self._clock_fn()
            if res.tzinfo is None:
                return res.replace(tzinfo=tz)
            return res.astimezone(tz)
        return datetime.now(tz)

    def _get_monotonic(self) -> float:
        if self._monotonic_fn is not None:
            return self._monotonic_fn()
        return time_mod.monotonic()

    def _is_voice_mode(self, context: Optional[AgentContext] = None) -> bool:
        if self._voice_mode or getattr(self, "voice_mode", False):
            return True
        if context is not None and bool(context.metadata.get("voice_mode")):
            return True
        return False

    def _fetch_events_for_date(
        self,
        target_date: Any,
        tz: ZoneInfo,
    ) -> List[Dict[str, Any]]:
        """Fetch and filter events for a specific calendar date."""
        from openjarvis.connectors.gcalendar import (
            GCalendarConnector,
            _gcal_api_calendars_list,
            _gcal_api_events_list,
        )
        from openjarvis.connectors.google_auth import call_with_refresh

        connector = GCalendarConnector()
        if not connector.is_connected():
            raise RuntimeError("gcalendar_not_connected")

        try:
            calendars_resp = call_with_refresh(
                _gcal_api_calendars_list, connector._credentials_path
            )
        except Exception as exc:
            logger.error("Failed to fetch calendar list: %s", type(exc).__name__)
            raise

        calendars = calendars_resp.get("items", [])
        start_of_day = datetime(
            target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=tz
        )
        time_min_str = start_of_day.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        matching_events: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for cal in calendars:
            cal_id = cal.get("id", "")
            if not cal_id:
                continue
            if is_ignored_calendar(
                cal_id, cal.get("summary", ""), cal.get("description", "")
            ):
                continue

            try:
                events_resp = call_with_refresh(
                    _gcal_api_events_list,
                    connector._credentials_path,
                    cal_id,
                    time_min=time_min_str,
                )
            except Exception as exc:
                logger.warning("Failed to fetch events for calendar: %s", type(exc).__name__)
                continue

            for item in events_resp.get("items", []):
                evt_id = item.get("id", "")
                if evt_id and evt_id in seen_ids:
                    continue
                if item.get("status") == "cancelled":
                    continue

                title = item.get("summary", "").strip() or "(Sem título)"
                event_type = item.get("eventType", "")
                if is_ignored_event(title, event_type):
                    continue

                start = item.get("start", {})
                end = item.get("end", {})

                # All-day event check
                if "date" in start:
                    try:
                        start_d = datetime.strptime(start["date"], "%Y-%m-%d").date()
                        end_d = (
                            datetime.strptime(end["date"], "%Y-%m-%d").date()
                            if "date" in end
                            else start_d + timedelta(days=1)
                        )
                    except (ValueError, TypeError):
                        continue

                    if start_d <= target_date < end_d:
                        if evt_id:
                            seen_ids.add(evt_id)
                        matching_events.append({
                            "title": title,
                            "is_all_day": True,
                            "time_str": "",
                            "sort_dt": datetime(
                                target_date.year,
                                target_date.month,
                                target_date.day,
                                0,
                                0,
                                tzinfo=tz,
                            ),
                        })
                elif "dateTime" in start:
                    dt_str = start["dateTime"].replace("Z", "+00:00")
                    try:
                        dt = datetime.fromisoformat(dt_str)
                    except (ValueError, TypeError):
                        continue

                    dt_local = dt.astimezone(tz)
                    if dt_local.date() == target_date:
                        if evt_id:
                            seen_ids.add(evt_id)
                        matching_events.append({
                            "title": title,
                            "is_all_day": False,
                            "time_str": dt_local.strftime("%H:%M"),
                            "sort_dt": dt_local,
                        })

        # Sort chronologically by start time
        matching_events.sort(key=lambda x: x["sort_dt"])
        return matching_events[:5]

    def _execute_event_creation(self, pending: PendingEventState) -> str:
        """Create Google Calendar event or simulate payload if dry run."""
        from openjarvis.connectors.gcalendar import GCalendarConnector

        assert pending.title is not None
        assert pending.target_dt is not None

        weekday = WEEKDAYS_PT[pending.target_dt.weekday()]
        day_month = f"{pending.target_dt.day:02d}/{pending.target_dt.month:02d}"
        time_str = (
            f"{pending.target_dt.hour}h"
            if pending.target_dt.minute == 0
            else f"{pending.target_dt.hour}h{pending.target_dt.minute:02d}"
        )

        # Build payload with 10-minute popup reminder
        payload = build_event_payload(pending.title, pending.target_dt)

        # Rule 5: JARVIS_AGENDA_DRY_RUN=1 displays simulation without calling Google
        if os.environ.get("JARVIS_AGENDA_DRY_RUN") == "1":
            logger.info("[DRY RUN] Payload do evento: %s", json.dumps(payload, ensure_ascii=False))
            return f"Simulação: {pending.title}, {weekday} {day_month}, {time_str}. Nada foi criado."

        connector = GCalendarConnector()
        if not connector.is_connected():
            return (
                "O Google Calendar não está conectado. "
                "Conecte com o comando: jarvis connect gcalendar."
            )

        try:
            # Check for duplicates before creating
            duplicate = check_duplicate_event(
                pending.title,
                pending.target_dt,
                credentials_path=connector._credentials_path,
                calendar_id="primary",
            )
            if duplicate is not None:
                return (
                    f"Já existe um compromisso '{pending.title}' agendado para "
                    f"{weekday}, {day_month}, às {time_str}."
                )

            # Rule 6: Reuses call_with_refresh, does not leak token or secrets
            create_calendar_event(
                payload,
                credentials_path=connector._credentials_path,
                calendar_id="primary",
            )
            return f"Pronto: {pending.title}, {weekday} {day_month}, {time_str}."
        except httpx.HTTPStatusError as exc:
            # Rule 7: HTTP 403 handling without deleting credentials
            if exc.response is not None and exc.response.status_code == 403:
                return (
                    "Falta permissão de escrita no Google Calendar. "
                    "Para atualizar o acesso, refaça o login executando: jarvis connect gcalendar."
                )
            logger.error(
                "Erro na API do Google Calendar: %s",
                exc.response.status_code if exc.response is not None else "?",
            )
            return "Não foi possível criar o compromisso na agenda devido a um erro no Google Calendar."
        except Exception as exc:
            logger.error("Erro ao criar evento na agenda: %s", type(exc).__name__)
            return "Não foi possível criar o compromisso na agenda no momento."

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Handle request: summarize agenda, create events with confirmation, or fallback."""
        self._emit_turn_start(input)

        now = self._get_now()
        tz = self._get_tz()
        monotonic_now = self._get_monotonic()
        is_voice = self._is_voice_mode(context)

        # Check expiration of pending state (Rule 1: 2 minutes or 2 turns)
        if self._pending_event is not None:
            elapsed = monotonic_now - self._pending_event.created_at
            if elapsed > 120.0 or self._pending_event.turns_alive >= 4:
                self._pending_event = None

        # ------------------------------------------------------------------
        # If there is a pending event in memory
        # ------------------------------------------------------------------
        if self._pending_event is not None:
            self._pending_event.turns_alive += 1

            # Awaiting confirmation
            if self._pending_event.awaiting == "confirmation":
                # Rule 2: Confirms ONLY on short affirmative; anything else cancels
                if is_short_affirmative(input):
                    wake_mode = bool(context and context.metadata.get("wake_mode"))
                    wake_named = bool(context and context.metadata.get("wake_named"))
                    if wake_mode and not wake_named:
                        content = "Por segurança, confirme dizendo 'Junior, sim'."
                        self._pending_event.turns_alive = max(0, self._pending_event.turns_alive - 1)
                        self._emit_turn_end(content_length=len(content))
                        return AgentResult(content=content, turns=1)

                    pending = self._pending_event
                    self._pending_event = None
                    content = self._execute_event_creation(pending)
                else:
                    self._pending_event = None
                    content = "Marcação cancelada."

                self._emit_turn_end(content_length=len(content))
                return AgentResult(content=content, turns=1)

            # Awaiting title clarification
            elif self._pending_event.awaiting == "title":
                if is_explicit_cancel(input):
                    self._pending_event = None
                    content = "Marcação cancelada."
                else:
                    title = parse_title(input)
                    if not title:
                        if self._pending_event.retry_count == 0:
                            self._pending_event.retry_count += 1
                            content = "Não entendi. Pode repetir o título?"
                        else:
                            self._pending_event = None
                            content = "Marcação cancelada."
                    else:
                        self._pending_event.title = title
                        self._pending_event.retry_count = 0
                        self._pending_event.turns_alive = 0
                        if self._pending_event.target_time is not None and self._pending_event.target_date is not None:
                            target_dt = datetime.combine(
                                self._pending_event.target_date,
                                self._pending_event.target_time,
                                tzinfo=tz,
                            )
                            self._pending_event.target_dt = target_dt
                            self._pending_event.awaiting = "confirmation"
                            content = format_confirmation_prompt(title, target_dt)
                        else:
                            self._pending_event.awaiting = "time"
                            content = "Para qual horário você gostaria de marcar?"

                self._emit_turn_end(content_length=len(content))
                return AgentResult(content=content, turns=1)

            # Awaiting ambiguous time clarification
            elif self._pending_event.awaiting == "ambiguous_time":
                if is_explicit_cancel(input):
                    self._pending_event = None
                    content = "Marcação cancelada."
                else:
                    hour = self._pending_event.ambiguous_hour or 12
                    minute = self._pending_event.ambiguous_minute or 0
                    in_lower = input.lower().strip()
                    resolved_time: Optional[time] = None
                    is_12_night = False

                    if any(w in in_lower for w in ["tarde", "pm"]):
                        if hour < 12:
                            hour += 12
                        resolved_time = time(hour, minute)
                    elif any(w in in_lower for w in ["noite"]):
                        if hour == 12:
                            hour = 0
                            is_12_night = True
                        elif hour < 12:
                            hour += 12
                        resolved_time = time(hour, minute)
                    elif any(w in in_lower for w in ["manh", "madrugada", "am"]):
                        if hour == 12:
                            hour = 0
                        resolved_time = time(hour, minute)
                    elif any(w in in_lower for w in ["meio-dia", "meio dia"]):
                        resolved_time = time(12, 0)
                    elif any(w in in_lower for w in ["meia-noite", "meia noite"]):
                        resolved_time = time(0, 0)
                        is_12_night = True
                    else:
                        time_res = parse_time(
                            input,
                            now=now,
                            target_date=self._pending_event.target_date,
                            tz=tz,
                        )
                        if time_res.time is not None:
                            resolved_time = time_res.time
                            is_12_night = time_res.is_12_night
                        elif time_res.is_ambiguous:
                            self._pending_event.ambiguous_hour = time_res.ambiguous_hour
                            self._pending_event.ambiguous_minute = time_res.ambiguous_minute
                            self._pending_event.retry_count = 0
                            self._pending_event.turns_alive = 0
                            if time_res.ambiguous_hour == 12:
                                content = "Você prefere ao meio-dia ou à meia-noite?"
                            elif time_res.ambiguous_hour and time_res.ambiguous_hour < 6:
                                content = f"Você prefere às {time_res.ambiguous_hour} da manhã ou às {time_res.ambiguous_hour} da tarde?"
                            else:
                                content = f"Você prefere às {time_res.ambiguous_hour} da manhã ou às {time_res.ambiguous_hour} da noite?"
                            self._emit_turn_end(content_length=len(content))
                            return AgentResult(content=content, turns=1)

                    if resolved_time is None:
                        if self._pending_event.retry_count == 0:
                            self._pending_event.retry_count += 1
                            content = "Não entendi. Pode repetir o horário?"
                        else:
                            self._pending_event = None
                            content = "Marcação cancelada."
                        self._emit_turn_end(content_length=len(content))
                        return AgentResult(content=content, turns=1)

                    target_date = self._pending_event.target_date or now.date()
                    if is_12_night:
                        target_date = target_date + timedelta(days=1)

                    # Past check
                    if target_date == now.date():
                        target_dt_check = datetime(
                            now.year, now.month, now.day, resolved_time.hour, resolved_time.minute, tzinfo=tz
                        )
                        if target_dt_check <= now:
                            self._pending_event.awaiting = "time"
                            self._pending_event.retry_count = 0
                            self._pending_event.turns_alive = 0
                            content = "Esse horário já passou hoje. Para qual horário você gostaria de marcar?"
                            self._emit_turn_end(content_length=len(content))
                            return AgentResult(content=content, turns=1)

                    self._pending_event.target_time = resolved_time
                    self._pending_event.target_date = target_date
                    target_dt = datetime.combine(target_date, resolved_time, tzinfo=tz)
                    self._pending_event.target_dt = target_dt
                    self._pending_event.awaiting = "confirmation"
                    self._pending_event.retry_count = 0
                    self._pending_event.turns_alive = 0
                    content = format_confirmation_prompt(self._pending_event.title or "Compromisso", target_dt)

                self._emit_turn_end(content_length=len(content))
                return AgentResult(content=content, turns=1)

            # Awaiting time clarification
            elif self._pending_event.awaiting == "time":
                if is_explicit_cancel(input):
                    self._pending_event = None
                    content = "Marcação cancelada."
                else:
                    time_res = parse_time(
                        input,
                        now=now,
                        target_date=self._pending_event.target_date,
                        tz=tz,
                    )
                    if time_res.is_ambiguous:
                        self._pending_event.ambiguous_hour = time_res.ambiguous_hour
                        self._pending_event.ambiguous_minute = time_res.ambiguous_minute
                        self._pending_event.awaiting = "ambiguous_time"
                        self._pending_event.retry_count = 0
                        self._pending_event.turns_alive = 0
                        if time_res.ambiguous_hour == 12:
                            content = "Você prefere ao meio-dia ou à meia-noite?"
                        elif time_res.ambiguous_hour and time_res.ambiguous_hour < 6:
                            content = f"Você prefere às {time_res.ambiguous_hour} da manhã ou às {time_res.ambiguous_hour} da tarde?"
                        else:
                            content = f"Você prefere às {time_res.ambiguous_hour} da manhã ou às {time_res.ambiguous_hour} da noite?"
                    elif time_res.is_past:
                        self._pending_event.retry_count = 0
                        self._pending_event.turns_alive = 0
                        content = "Esse horário já passou hoje. Para qual horário você gostaria de marcar?"
                    elif time_res.time is None:
                        if self._pending_event.retry_count == 0:
                            self._pending_event.retry_count += 1
                            content = "Não entendi. Pode repetir o horário?"
                        else:
                            self._pending_event = None
                            content = "Marcação cancelada."
                    else:
                        target_time = time_res.time
                        target_date = self._pending_event.target_date or now.date()
                        if time_res.is_12_night:
                            target_date = target_date + timedelta(days=1)
                        target_dt = datetime.combine(target_date, target_time, tzinfo=tz)
                        self._pending_event.target_time = target_time
                        self._pending_event.target_date = target_date
                        self._pending_event.target_dt = target_dt
                        self._pending_event.awaiting = "confirmation"
                        self._pending_event.retry_count = 0
                        self._pending_event.turns_alive = 0
                        content = format_confirmation_prompt(self._pending_event.title or "Compromisso", target_dt)

                self._emit_turn_end(content_length=len(content))
                return AgentResult(content=content, turns=1)

        # ------------------------------------------------------------------
        # No pending event in memory
        # ------------------------------------------------------------------

        # Rule 1: A "sim" with no pending action does nothing (falls through to normal conversation)
        if is_short_affirmative(input):
            messages = self._build_messages(
                input, context, system_prompt=_DEFAULT_SYSTEM_PROMPT
            )
            result = self._generate(messages)
            content = result.get("content", "")
            self._emit_turn_end(content_length=len(content))
            return AgentResult(content=content, turns=1)

        # Agenda creation request
        if is_create_intent(input):
            # Rule 1: jarvis ask (single turn / not voice chat) does not confirm or create
            if not is_voice:
                content = "A criação de eventos só funciona no chat --voice."
                self._emit_turn_end(content_length=len(content))
                return AgentResult(content=content, turns=1)

            parsed = parse_create_event(input, now=now, tz=tz)

            if parsed.missing_title:
                self._pending_event = PendingEventState(
                    title=None,
                    target_date=parsed.target_date,
                    target_time=parsed.target_time,
                    awaiting="title",
                    created_at=monotonic_now,
                    turns_alive=0,
                )
                content = "Qual é o título do evento?"
                self._emit_turn_end(content_length=len(content))
                return AgentResult(content=content, turns=1)

            if parsed.is_ambiguous_time:
                self._pending_event = PendingEventState(
                    title=parsed.title,
                    target_date=parsed.target_date,
                    ambiguous_hour=parsed.ambiguous_hour,
                    ambiguous_minute=parsed.ambiguous_minute,
                    awaiting="ambiguous_time",
                    created_at=monotonic_now,
                    turns_alive=0,
                )
                if parsed.ambiguous_hour == 12:
                    content = "Você prefere ao meio-dia ou à meia-noite?"
                elif parsed.ambiguous_hour and parsed.ambiguous_hour < 6:
                    content = f"Você prefere às {parsed.ambiguous_hour} da manhã ou às {parsed.ambiguous_hour} da tarde?"
                else:
                    content = f"Você prefere às {parsed.ambiguous_hour} da manhã ou às {parsed.ambiguous_hour} da noite?"
                self._emit_turn_end(content_length=len(content))
                return AgentResult(content=content, turns=1)

            if parsed.is_past_time:
                self._pending_event = PendingEventState(
                    title=parsed.title,
                    target_date=parsed.target_date,
                    awaiting="time",
                    created_at=monotonic_now,
                    turns_alive=0,
                )
                content = "Esse horário já passou hoje. Para qual horário você gostaria de marcar?"
                self._emit_turn_end(content_length=len(content))
                return AgentResult(content=content, turns=1)

            if parsed.missing_time or parsed.target_time is None:
                self._pending_event = PendingEventState(
                    title=parsed.title,
                    target_date=parsed.target_date,
                    awaiting="time",
                    created_at=monotonic_now,
                    turns_alive=0,
                )
                content = "Para qual horário você gostaria de marcar?"
                self._emit_turn_end(content_length=len(content))
                return AgentResult(content=content, turns=1)

            assert parsed.target_dt is not None
            self._pending_event = PendingEventState(
                title=parsed.title,
                target_date=parsed.target_date,
                target_time=parsed.target_time,
                target_dt=parsed.target_dt,
                awaiting="confirmation",
                created_at=monotonic_now,
                turns_alive=0,
            )
            content = format_confirmation_prompt(parsed.title, parsed.target_dt)
            self._emit_turn_end(content_length=len(content))
            return AgentResult(content=content, turns=1)

        # Daily summary request
        if is_agenda_request(input):
            is_tomorrow = is_tomorrow_request(input)
            target_date = (now + timedelta(days=1)).date() if is_tomorrow else now.date()

            try:
                events = self._fetch_events_for_date(target_date, tz)
                content = format_agenda_phrase(events, is_tomorrow)
            except RuntimeError as exc:
                if str(exc) == "gcalendar_not_connected":
                    content = (
                        "O Google Calendar não está conectado. "
                        "Conecte com o comando: jarvis connect gcalendar."
                    )
                else:
                    content = "Não foi possível carregar seus compromissos no momento."
            except Exception as exc:
                logger.error("Error retrieving agenda: %s", type(exc).__name__)
                content = "Não foi possível carregar seus compromissos no momento."

            self._emit_turn_end(content_length=len(content))
            return AgentResult(content=content, turns=1)

        # Non-agenda request: respond as normal conversation
        messages = self._build_messages(
            input, context, system_prompt=_DEFAULT_SYSTEM_PROMPT
        )
        result = self._generate(messages)
        content = result.get("content", "")
        self._emit_turn_end(content_length=len(content))
        return AgentResult(content=content, turns=1)


__all__ = [
    "AgendaAgent",
    "PendingEventState",
    "format_agenda_phrase",
    "is_agenda_request",
    "is_ignored_calendar",
    "is_ignored_event",
    "is_tomorrow_request",
]
