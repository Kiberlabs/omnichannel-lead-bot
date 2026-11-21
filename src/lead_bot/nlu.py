from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .models import Intent

_SPAM_PATTERNS = (
    "crypto airdrop",
    "casino bonus",
    "forex signal",
    "buy followers",
    "click here to claim",
    "gana dinero rapido",
)

_HUMAN_PATTERNS = (
    "hablar con una persona",
    "agente humano",
    "atencion humana",
    "pasar con alguien",
    "human agent",
    "speak to a person",
    "real person",
    "talk to a human",
)

_SERVICE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "web_scraping": ("scraping", "scraper", "extraer datos", "precios", "catalogo", "price monitor"),
    "messaging_bot": ("bot", "whatsapp", "telegram", "chatbot", "atencion automatica"),
    "b2b_automation": ("automatizacion", "automatizar", "automation", "workflow", "api", "dashboard", "panel"),
    "ecommerce_ai": ("tienda", "ecommerce", "shopify", "woocommerce", "video", "ffmpeg"),
}

_URGENT_WORDS = ("urgente", "hoy mismo", "cuanto antes", "urgent", "asap", "immediately")
_ES_HINTS = ("hola", "buenas", "necesito", "quiero", "presupuesto", "precio", "empresa", "para", "con")
_EN_HINTS = ("hello", "hi", "need", "looking", "budget", "price", "company", "build", "quote")

# Matches amounts like "1.200€", "1.200,50€", "€850", "$1,500", "600 euros", "850.00 usd"
_AMT = r"(\d{1,3}(?:[.,]\d{3})+|\d{2,6})(?:[.,]\d{1,2})?"
_BUDGET_RE = re.compile(
    rf"(?:[€$]\s*{_AMT}|{_AMT}\s*(?:€|eur|euros?|usd|\$))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NLUResult:
    intent: Intent
    lang: Literal["es", "en"]
    service_need: str | None
    budget_eur: float | None
    urgent: bool


def _normalize(text: str) -> str:
    table = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")
    return text.translate(table).lower().strip()


def detect_language(text: str, fallback: Literal["es", "en"] = "es") -> Literal["es", "en"]:
    """Score Spanish vs. English lexical hints (plus inverted '?'/'ñ') and fall back on ties."""
    tokens = set(re.findall(r"[a-zñ]+", text.lower()))
    es_score = len(tokens.intersection(_ES_HINTS)) + (2 if "¿" in text or "ñ" in text.lower() else 0)
    en_score = len(tokens.intersection(_EN_HINTS))
    if en_score > es_score:
        return "en"
    if es_score > en_score:
        return "es"
    return fallback


def extract_budget_eur(text: str) -> float | None:
    """Parse European/US currency expressions (e.g. '1.200€', '$1,500') within 50–250,000 EUR."""
    match = _BUDGET_RE.search(text)
    if not match:
        return None
    raw = (match.group(1) or match.group(2) or "").replace(".", "").replace(",", "")
    value = float(raw)
    return value if 50.0 <= value <= 250_000.0 else None


def classify_message(text: str, fallback_lang: Literal["es", "en"] = "es") -> NLUResult:
    """Zero-token bilingual NLU pipeline for intent, service, budget, urgency, and spam detection.

    Priority order: spam (>=3 links or scam phrases) -> human escalation ->
    service inquiry -> budget provided -> greeting -> unknown.
    """
    norm = _normalize(text)
    lang = detect_language(text, fallback=fallback_lang)

    if norm.count("http://") + norm.count("https://") >= 3 or any(p in norm for p in _SPAM_PATTERNS):
        return NLUResult(Intent.SPAM, lang, None, None, False)

    urgent = any(w in norm for w in _URGENT_WORDS)
    budget = extract_budget_eur(text)

    detected_service: str | None = None
    for category, keywords in _SERVICE_KEYWORDS.items():
        if any(kw in norm for kw in keywords):
            detected_service = category
            break

    if any(p in norm for p in _HUMAN_PATTERNS):
        return NLUResult(Intent.HUMAN_ESCALATION, lang, detected_service, budget, urgent)

    if detected_service is not None:
        return NLUResult(Intent.SERVICE_INQUIRY, lang, detected_service, budget, urgent)

    if budget is not None:
        return NLUResult(Intent.BUDGET_PROVIDED, lang, None, budget, urgent)

    if any(g in norm.split() for g in ("hola", "buenas", "hello", "hi", "hey")):
        return NLUResult(Intent.GREETING, lang, None, None, urgent)

    return NLUResult(Intent.UNKNOWN, lang, None, None, urgent)
