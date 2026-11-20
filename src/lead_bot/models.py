from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class LeadState(str, Enum):
    NEW = "new"
    AWAITING_SERVICE = "awaiting_service"
    AWAITING_BUDGET = "awaiting_budget"
    HANDOFF_READY = "handoff_ready"
    SPAM_BLOCKED = "spam_blocked"


class Intent(str, Enum):
    GREETING = "greeting"
    SERVICE_INQUIRY = "service_inquiry"
    BUDGET_PROVIDED = "budget_provided"
    HUMAN_ESCALATION = "human_escalation"
    SPAM = "spam"
    UNKNOWN = "unknown"


class IncomingMessage(BaseModel):
    channel: Literal["whatsapp", "telegram", "web"]
    sender_id: str = Field(min_length=2, max_length=80)
    message_id: str = Field(min_length=2, max_length=120)
    text: str = Field(min_length=1, max_length=2000)
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LeadSession(BaseModel):
    sender_id: str
    channel: Literal["whatsapp", "telegram", "web"]
    lang: Literal["es", "en"] = "es"
    state: LeadState = LeadState.NEW
    service_need: str | None = None
    estimated_budget_eur: float | None = None
    urgent: bool = False
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HandoffPayload(BaseModel):
    sender_id: str
    channel: str
    lang: str
    reason: Literal["qualified_lead", "human_requested"]
    service_need: str | None
    estimated_budget_eur: float | None
    urgent: bool


class BotTurnResult(BaseModel):
    duplicate: bool = False
    intent: Intent
    session: LeadSession
    reply_text: str | None
    handoff: HandoffPayload | None = None
