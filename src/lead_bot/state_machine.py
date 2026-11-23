from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from .models import BotTurnResult, HandoffPayload, IncomingMessage, Intent, LeadSession, LeadState, WarmNudge
from .nlu import classify_message
from .storage import LeadStore

_NUDGE_WINDOW_HM = ("09:30", "21:00")

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
        existing = self.store.get_session(msg.channel, msg.sender_id)
        if self.store.is_duplicate(msg.channel, msg.message_id):
            return BotTurnResult(duplicate=True, intent=Intent.UNKNOWN, session=existing, reply_text=None)

        nlu = classify_message(msg.text, fallback_lang=existing.lang)
        session = existing.model_copy(deep=True)
        session.lang = nlu.lang
        session.urgent = session.urgent or nlu.urgent
        session.updated_at = msg.received_at

        if nlu.service_need:
            session.service_need = nlu.service_need
        if nlu.budget_eur is not None:
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
        """Send at most one polite follow-up per idle lead during business hours (09:30-21:00)."""
        current = now or datetime.now(timezone.utc)
        hm = current.strftime("%H:%M")
        if max_nudges <= 0 or not (_NUDGE_WINDOW_HM[0] <= hm <= _NUDGE_WINDOW_HM[1]):
            return []

        sent: list[WarmNudge] = []
        for session in self.store.find_stale_leads_for_nudge(current):
            if len(sent) >= max_nudges:
                break
            hint = f" ({session.service_need})" if session.service_need else ""
            text = _COPY[session.lang]["warm_nudge"].format(service_hint=hint)
            nudge = WarmNudge(channel=session.channel, sender_id=session.sender_id, lang=session.lang, text=text)
            if not send_fn(nudge):
                continue
            if self.store.record_nudge_sent(session.channel, session.sender_id, current, text):
                sent.append(nudge)
        return sent
