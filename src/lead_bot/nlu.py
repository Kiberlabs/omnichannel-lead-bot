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

_COMPLAINT_OPTOUT_PATTERNS = (
    "stop messaging",
    "do not contact",
    "dont contact",
    "don't contact",
    "unsubscribe",
    "opt out",
    "opt-out",
    "remove my",
    "delete my data",
    "erase my data",
    "gdpr removal",
    "gdpr deletion",
    "no me escribas",
    "deja de escribir",
    "darme de baja",
    "darse de baja",
    "baja rgpd",
    "baja gdpr",
    "quita mi anuncio",
    "borra mis datos",
    "borrar mis datos",
    "eliminar mis datos",
    "suprimir mis datos",
    "derecho al olvido",
    "denunciar",
)

_SERVICE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "web_scraping": (
        "scraping",
        "scraper",
        "extraer datos",
        "monitorizar precios",
        "monitoreo de precios",
        "seguimiento de precios",
        "comparar precios",
        "comparador de precios",
        "rastrear precios",
        "extraer precios",
        "precios de competencia",
        "precios de la competencia",
        "precios de proveedores",
        "catalogo",
        "price monitor",
    ),
    "messaging_bot": ("bot", "whatsapp", "telegram", "chatbot", "atencion automatica"),
    "b2b_automation": ("automatizacion", "automatizar", "automation", "workflow", "api", "dashboard"),
    "ecommerce_ai": ("tienda", "ecommerce", "shopify", "woocommerce", "video", "ffmpeg"),
}

_PANEL_GUARDED = (
    r"(?<!\bsolar )\bpanel"
    r"(?!\s+(?:solar|fotovoltaico|termico|sandwich|acustico|electrico|de\s+(?:madera|yeso))\b)"
)

_URGENT_WORDS = ("urgente", "hoy mismo", "cuanto antes", "urgent", "asap", "immediately")
_ES_HINTS = ("hola", "buenas", "necesito", "quiero", "presupuesto", "precio", "empresa", "para", "con")
_EN_HINTS = ("hello", "hi", "need", "looking", "budget", "price", "company", "build", "quote")


def _compile_keywords_re(words: tuple[str, ...], extra_patterns: tuple[str, ...] = ()) -> re.Pattern[str]:
    alts = [rf"\b{r'\s+'.join(map(re.escape, w.split()))}\b" for w in words]
    alts.extend(extra_patterns)
    return re.compile(rf"(?:{'|'.join(alts)})")


_URGENT_RE = _compile_keywords_re(_URGENT_WORDS)
_HUMAN_RE = _compile_keywords_re(_HUMAN_PATTERNS)
_OPTOUT_RE = _compile_keywords_re(_COMPLAINT_OPTOUT_PATTERNS)
_SERVICE_PATTERNS = {
    cat: _compile_keywords_re(kws, (_PANEL_GUARDED,) if cat == "b2b_automation" else ())
    for cat, kws in _SERVICE_KEYWORDS.items()
}
_GREETING_RE = _compile_keywords_re(("hola", "buenas", "hello", "hi", "hey"))

# Matches amounts like "1.200€", "1.200,50€", "€850", "$1,500", "$3k", "10k EUR", "$0.2M"
_AMT = r"(\d{1,3}(?:[.,]\d{3})+|\d{1,6})([.,]\d{1,2})?([kKmM])?(?!\d|[.,]\d)"
_BUDGET_RE = re.compile(
    rf"(?:[€$]\s*{_AMT}(?![A-Za-zÀ-ÿ])|(?<![\d.,]){_AMT}\s*(?:€|\$|(?:eur|euros?|usd)\b))",
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
    table = str.maketrans("áéíóúüÁÉÍÓÚÜ\u2019\u2018", "aeiouuAEIOUU''")
    return re.sub(r"\s+", " ", text.translate(table).lower()).strip()


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
    """Parse European/US currency expressions (e.g. '1.200€', '$1,500', '$3k') within 50–250,000 EUR."""
    for match in _BUDGET_RE.finditer(text):
        int_part = match.group(1) or match.group(4) or ""
        dec_part = match.group(2) or match.group(5) or ""
        suffix = (match.group(3) or match.group(6) or "").lower()
        raw = int_part.replace(".", "").replace(",", "")
        if dec_part:
            raw = f"{raw}.{dec_part[1:]}"
        value = float(raw)
        if suffix == "k":
            value *= 1_000.0
        elif suffix == "m":
            value *= 1_000_000.0
        if 50.0 <= value <= 250_000.0:
            return value
    return None


def classify_message(text: str, fallback_lang: Literal["es", "en"] = "es") -> NLUResult:
    """Zero-token bilingual NLU pipeline for intent, service, budget, urgency, and spam detection.

    Priority order: spam (>=3 links or scam phrases) -> human escalation ->
    service inquiry -> budget provided -> greeting -> unknown.
    """
    norm = _normalize(text)
    lang = detect_language(text, fallback=fallback_lang)

    if norm.count("http://") + norm.count("https://") >= 3 or any(p in norm for p in _SPAM_PATTERNS):
        return NLUResult(Intent.SPAM, lang, None, None, False)

    if _OPTOUT_RE.search(norm) is not None:
        return NLUResult(Intent.COMPLAINT_OPTOUT, lang, None, None, True)

    urgent = _URGENT_RE.search(norm) is not None
    budget = extract_budget_eur(text)

    detected_service: str | None = None
    for category, pattern in _SERVICE_PATTERNS.items():
        if pattern.search(norm) is not None:
            detected_service = category
            break

    if _HUMAN_RE.search(norm) is not None:
        return NLUResult(Intent.HUMAN_ESCALATION, lang, detected_service, budget, urgent)

    if detected_service is not None:
        return NLUResult(Intent.SERVICE_INQUIRY, lang, detected_service, budget, urgent)

    if budget is not None:
        return NLUResult(Intent.BUDGET_PROVIDED, lang, None, budget, urgent)

    if _GREETING_RE.search(norm) is not None:
        return NLUResult(Intent.GREETING, lang, None, None, urgent)

    return NLUResult(Intent.UNKNOWN, lang, None, None, urgent)
