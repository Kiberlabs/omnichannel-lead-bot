from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import BotTurnResult, HandoffPayload, IncomingMessage, Intent, LeadSession, LeadState, WarmNudge, ensure_utc
from .nlu import classify_message
from .storage import LeadStore

_NUDGE_WINDOW_HM = ("09:30", "21:00")


def _last_sunday_of_month(year: int, month: int) -> date:
    last_day = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return last_day - timedelta(days=(last_day.weekday() + 1) % 7)


class _EuropeMadridFallbackTZ(tzinfo):
    """EU CET/CEST fallback when tzdata is absent on host."""

    def _is_dst_utc(self, utc_dt: datetime) -> bool:
        naive_utc = utc_dt.replace(tzinfo=None)
        start = datetime.combine(_last_sunday_of_month(utc_dt.year, 3), time(1, 0))
        end = datetime.combine(_last_sunday_of_month(utc_dt.year, 10), time(1, 0))
        return start <= naive_utc < end

    def fromutc(self, dt: datetime) -> datetime:
        return dt + timedelta(hours=2 if self._is_dst_utc(dt) else 1)

    def utcoffset(self, dt: datetime | None) -> timedelta:
        if dt is None:
            return timedelta(hours=1)
        return timedelta(hours=2 if self._is_dst_utc(dt - timedelta(hours=1)) else 1)

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(hours=1) if self.utcoffset(dt) == timedelta(hours=2) else timedelta(0)

    def tzname(self, dt: datetime | None) -> str:
        return "CEST" if self.utcoffset(dt) == timedelta(hours=2) else "CET"


try:
    _MADRID_TZ: tzinfo = ZoneInfo("Europe/Madrid")
except ZoneInfoNotFoundError:
    _MADRID_TZ = _EuropeMadridFallbackTZ()

_COPY: dict[str, dict[str, str]] = {
    "es": {
        "ask_service": (
            "¡Hola! Soy el asistente técnico de KiberLabs. ¿Qué necesitas automatizar hoy? "
            "(Scraping de datos, Bot WhatsApp/Telegram, API/Panel B2B o E-commerce + IA)"
        ),
        "ask_budget": (
            "Perfecto, tomamos nota ({service}). Para asignarte al ingeniero adecuado, "
            "¿qué presupuesto aproximado manejáis para este proyecto (ej. 500€–1.500€)?"
        ),
        "handoff_qualified": (
            "¡Recibido! Proyecto de {service} (~{budget:.0f} EUR) registrado. "
            "Un ingeniero sénior de KiberLabs revisará tu caso y te responderá por aquí enseguida."
        ),
        "handoff_human": (
            "Entendido. Paso tu conversación directamente a un responsable de KiberLabs para atenderte en persona."
        ),
        "already_handed_off": (
            "Tu solicitud ya está en manos de nuestro equipo técnico; te responderemos en breve."
        ),
        "warm_nudge": (
            "Hola de nuevo — por si seguís valorando el proyecto{service_hint}, "
            "quedamos disponibles por aquí sin compromiso."
        ),
    },
    "en": {
        "ask_service": (
            "Hi! KiberLabs automated intake here. What would you like to build or automate? "
            "(Web Scraping, WhatsApp/Telegram Bot, B2B API/Dashboard, or E-commerce + AI)"
        ),
        "ask_budget": (
            "Got it ({service}). To route this to the right engineer, what approximate budget range "
            "do you have in mind (e.g. €500–€1,500)?"
        ),
        "handoff_qualified": (
            "All set! Your {service} scope (~€{budget:.0f}) is queued. "
            "A KiberLabs senior engineer will follow up directly in this chat shortly."
        ),
        "handoff_human": (
            "Understood — routing your conversation straight to a KiberLabs specialist now."
        ),
        "already_handed_off": (
            "Your request is already queued with our engineering team; someone will reply shortly."
        ),
        "warm_nudge": (
            "Quick check-in — if you are still scoping the project{service_hint}, "
            "happy to help whenever you are ready."
        ),
    },
}


class LeadQualificationEngine:
    """Deterministic 24/7 state machine for qualifying inbound B2B leads with zero LLM token burn."""

    def __init__(self, store: LeadStore) -> None:
        self.store = store

    def handle_message(self, msg: IncomingMessage) -> BotTurnResult:
        """Process one inbound webhook turn with idempotent deduplication and state transitions.

        Flow: check duplicate message_id -> run bilingual NLU -> merge captured service/budget
        into session -> compute next state/reply/handoff -> atomically commit to SQLite WAL.
        """
        with self.store.transaction():
            persisted = self.store.get_persisted_session(msg.channel, msg.sender_id)
            existing = (
                persisted
                if persisted is not None
                else LeadSession(channel=msg.channel, sender_id=msg.sender_id)
            )
            if self.store.is_duplicate(msg.channel, msg.message_id):
                return BotTurnResult(duplicate=True, intent=Intent.UNKNOWN, session=existing, reply_text=None)

            msg_ts = ensure_utc(msg.received_at)
            is_out_of_order = persisted is not None and msg_ts < existing.updated_at

            nlu = classify_message(msg.text, fallback_lang=existing.lang)
            session = existing.model_copy(deep=True)
            if not is_out_of_order:
                session.lang = nlu.lang
            session.urgent = session.urgent or nlu.urgent
            session.updated_at = max(existing.updated_at, msg_ts) if persisted is not None else msg_ts

            if nlu.service_need and (not is_out_of_order or session.service_need is None):
                session.service_need = nlu.service_need
            if nlu.budget_eur is not None and (not is_out_of_order or session.estimated_budget_eur is None):
                session.estimated_budget_eur = nlu.budget_eur

            copy = _COPY[session.lang]
            handoff: HandoffPayload | None = None
            reply: str | None = None

            if session.state == LeadState.SILENCED or nlu.intent == Intent.COMPLAINT_OPTOUT:
                was_silenced = session.state == LeadState.SILENCED
                session.state = LeadState.SILENCED
                session.urgent = True
                reply = None
                if not was_silenced:
                    handoff = HandoffPayload(
                        sender_id=session.sender_id,
                        channel=session.channel,
                        lang=session.lang,
                        reason="complaint_optout",
                        service_need=session.service_need,
                        estimated_budget_eur=session.estimated_budget_eur,
                        urgent=True,
                    )
            elif session.state == LeadState.SPAM_BLOCKED or nlu.intent == Intent.SPAM:
                session.state = LeadState.SPAM_BLOCKED
                reply = None
            elif session.state == LeadState.HANDOFF_READY:
                reply = copy["already_handed_off"]
            elif nlu.intent == Intent.HUMAN_ESCALATION:
                session.state = LeadState.HANDOFF_READY
                handoff = HandoffPayload(
                    sender_id=session.sender_id,
                    channel=session.channel,
                    lang=session.lang,
                    reason="human_requested",
                    service_need=session.service_need,
                    estimated_budget_eur=session.estimated_budget_eur,
                    urgent=session.urgent,
                )
                reply = copy["handoff_human"]
            elif session.service_need and session.estimated_budget_eur is not None:
                session.state = LeadState.HANDOFF_READY
                handoff = HandoffPayload(
                    sender_id=session.sender_id,
                    channel=session.channel,
                    lang=session.lang,
                    reason="qualified_lead",
                    service_need=session.service_need,
                    estimated_budget_eur=session.estimated_budget_eur,
                    urgent=session.urgent,
                )
                reply = copy["handoff_qualified"].format(
                    service=session.service_need,
                    budget=session.estimated_budget_eur,
                )
            elif session.service_need is None:
                session.state = LeadState.AWAITING_SERVICE
                reply = copy["ask_service"]
            else:
                session.state = LeadState.AWAITING_BUDGET
                reply = copy["ask_budget"].format(service=session.service_need)

            committed = self.store.commit_turn(msg, session, handoff)
            if not committed:
                return BotTurnResult(duplicate=True, intent=nlu.intent, session=existing, reply_text=None)

            return BotTurnResult(
                duplicate=False,
                intent=nlu.intent,
                session=session,
                reply_text=reply,
                handoff=handoff,
            )

    def dispatch_warm_nudges(
        self,
        send_fn: Callable[[WarmNudge], bool],
        *,
        now: datetime | None = None,
        max_nudges: int = 5,
    ) -> list[WarmNudge]:
        """Send at most one polite follow-up per idle lead during Europe/Madrid business hours (09:30-21:00)."""
        current = ensure_utc(now) if now is not None else datetime.now(timezone.utc)
        hm = current.astimezone(_MADRID_TZ).strftime("%H:%M")
        if max_nudges <= 0 or not (_NUDGE_WINDOW_HM[0] <= hm <= _NUDGE_WINDOW_HM[1]):
            return []

        sent: list[WarmNudge] = []
        for session in self.store.find_stale_leads_for_nudge(current):
            if len(sent) >= max_nudges:
                break
            nudge: WarmNudge | None = None
            with self.store.transaction():
                fresh = self.store.get_persisted_session(session.channel, session.sender_id)
                if fresh is None or fresh.state not in (LeadState.AWAITING_SERVICE, LeadState.AWAITING_BUDGET):
                    continue
                idle_sec = (current - fresh.updated_at).total_seconds()
                if not (12.0 * 3600.0 <= idle_sec <= 36.0 * 3600.0):
                    continue
                hint = f" ({fresh.service_need})" if fresh.service_need else ""
                text = _COPY[fresh.lang]["warm_nudge"].format(service_hint=hint)
                candidate = WarmNudge(channel=fresh.channel, sender_id=fresh.sender_id, lang=fresh.lang, text=text)
                if self.store.record_nudge_sent(
                    fresh.channel,
                    fresh.sender_id,
                    current,
                    text,
                    min_idle_hours=12.0,
                    max_idle_hours=36.0,
                ):
                    nudge = candidate
            if nudge is None:
                continue
            try:
                ok = send_fn(nudge)
            except BaseException:
                self.store.release_nudge(nudge.channel, nudge.sender_id)
                raise
            if not ok:
                self.store.release_nudge(nudge.channel, nudge.sender_id)
                continue
            sent.append(nudge)
        return sent
