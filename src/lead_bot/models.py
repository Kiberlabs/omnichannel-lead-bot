from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def ensure_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class LeadState(str, Enum):
    """Finite state machine lifecycle for an inbound B2B prospect conversation."""

    NEW = "new"
    AWAITING_SERVICE = "awaiting_service"
    AWAITING_BUDGET = "awaiting_budget"
    HANDOFF_READY = "handoff_ready"
    SPAM_BLOCKED = "spam_blocked"
    SILENCED = "silenced"


class Intent(str, Enum):
    """Deterministic zero-token intent label extracted from an inbound message."""

    GREETING = "greeting"
    SERVICE_INQUIRY = "service_inquiry"
    BUDGET_PROVIDED = "budget_provided"
    HUMAN_ESCALATION = "human_escalation"
    COMPLAINT_OPTOUT = "complaint_optout"
    SPAM = "spam"
    UNKNOWN = "unknown"


class IncomingMessage(BaseModel):
    """Normalized inbound webhook event from WhatsApp, Telegram, or web chat."""

    channel: Literal["whatsapp", "telegram", "web"]
    sender_id: str = Field(min_length=2, max_length=80)
    message_id: str = Field(min_length=2, max_length=120)
    text: str = Field(min_length=1, max_length=2000)
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("sender_id", "message_id")
    @classmethod
    def _check_non_blank_id(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 2:
            raise ValueError("Identifier must contain at least 2 non-whitespace characters.")
        return stripped

    @field_validator("text")
    @classmethod
    def _check_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank.")
        return value

    @field_validator("received_at")
    @classmethod
    def _normalize_received_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class LeadSession(BaseModel):
    """Persisted per-(channel, sender_id) qualification state and captured slots."""

    sender_id: str
    channel: Literal["whatsapp", "telegram", "web"]
    lang: Literal["es", "en"] = "es"
    state: LeadState = LeadState.NEW
    service_need: str | None = None
    estimated_budget_eur: float | None = None
    urgent: bool = False
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("updated_at")
    @classmethod
    def _normalize_updated_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class HandoffPayload(BaseModel):
    """Structured ticket enqueued for a human engineer once a lead qualifies or escalates."""

    sender_id: str
    channel: str
    lang: str
    reason: Literal["qualified_lead", "human_requested", "complaint_optout"]
    service_need: str | None
    estimated_budget_eur: float | None
    urgent: bool


class WarmNudge(BaseModel):
    """Single-touch polite follow-up emitted for an idle unconverted lead."""

    channel: str
    sender_id: str
    lang: Literal["es", "en"]
    text: str


class BotTurnResult(BaseModel):
    """Outcome of handling one inbound webhook: updated session, outbound reply, and optional handoff."""

    duplicate: bool = False
    intent: Intent
    session: LeadSession
    reply_text: str | None
    handoff: HandoffPayload | None = None
