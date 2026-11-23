from __future__ import annotations

import json
from pathlib import Path

import pytest

from lead_bot.cli import main as cli_main
from lead_bot.models import IncomingMessage, Intent, LeadState
from lead_bot.nlu import extract_budget_eur
from lead_bot.state_machine import LeadQualificationEngine
from lead_bot.storage import LeadStore


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("Presupuesto de 1.200€ cerrado", 1200.0),
        ("Around $1,500 for MVP", 1500.0),
        ("Podemos pagar 650 euros", 650.0),
        ("850.00 usd fixed price", 850.0),
        ("Sin cifra todavia", None),
    ],
)
def test_budget_extraction_formats(raw_text: str, expected: float | None) -> None:
    assert extract_budget_eur(raw_text) == expected


def test_spanish_multi_turn_qualification_and_webhook_dedup(tmp_path: Path) -> None:
    with LeadStore(tmp_path / "leads.db") as store:
        assert store.journal_mode() == "wal"
        engine = LeadQualificationEngine(store)

        t1 = engine.handle_message(
            IncomingMessage(channel="whatsapp", sender_id="u1", message_id="m1", text="Hola buenas")
        )
        assert t1.session.state == LeadState.AWAITING_SERVICE
        assert t1.session.lang == "es"

        t2 = engine.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="u1",
                message_id="m2",
                text="Quiero un bot de WhatsApp urgente para mi clínica",
            )
        )
        assert t2.session.state == LeadState.AWAITING_BUDGET
        assert t2.session.service_need == "messaging_bot"
        assert t2.session.urgent is True

        # Webhook retry with identical message_id must be ignored idempotently
        dup = engine.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="u1",
                message_id="m2",
                text="Quiero un bot de WhatsApp urgente para mi clínica",
            )
        )
        assert dup.duplicate is True
        assert dup.reply_text is None

        t3 = engine.handle_message(
            IncomingMessage(channel="whatsapp", sender_id="u1", message_id="m3", text="Sobre 1.100€")
        )
        assert t3.session.state == LeadState.HANDOFF_READY
        assert t3.handoff is not None
        assert t3.handoff.reason == "qualified_lead"
        assert t3.handoff.estimated_budget_eur == 1100.0
        assert len(store.pending_handoffs()) == 1


def test_english_single_shot_qualification_and_human_escape(tmp_path: Path) -> None:
    with LeadStore(tmp_path / "en.db") as store:
        engine = LeadQualificationEngine(store)

        one_shot = engine.handle_message(
            IncomingMessage(
                channel="telegram",
                sender_id="buyer_uk",
                message_id="tg1",
                text="Hi, we need a Python price monitor scraper, budget is $900",
            )
        )
        assert one_shot.session.lang == "en"
        assert one_shot.session.state == LeadState.HANDOFF_READY
        assert one_shot.handoff is not None
        assert one_shot.handoff.service_need == "web_scraping"

        esc = engine.handle_message(
            IncomingMessage(
                channel="web",
                sender_id="buyer_vip",
                message_id="w1",
                text="Hello, I would like to speak to a person please",
            )
        )
        assert esc.intent == Intent.HUMAN_ESCALATION
        assert esc.session.state == LeadState.HANDOFF_READY
        assert esc.handoff is not None
        assert esc.handoff.reason == "human_requested"


def test_spam_silently_blocked(tmp_path: Path) -> None:
    with LeadStore(tmp_path / "spam.db") as store:
        engine = LeadQualificationEngine(store)
        res = engine.handle_message(
            IncomingMessage(
                channel="web",
                sender_id="sp1",
                message_id="s1",
                text="Claim free crypto airdrop now!",
            )
        )
        assert res.session.state == LeadState.SPAM_BLOCKED
        assert res.reply_text is None
        assert store.pending_handoffs() == []


def test_cli_simulation_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "data" / "demo_conversation.json"
    db_file = tmp_path / "sim.db"

    rc = cli_main(["simulate", "--script", str(script), "--db", str(db_file), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["turns"]) == 6
    assert sum(1 for t in out["turns"] if t["duplicate"]) == 1
    assert len(out["handoff_queue"]) == 2


def test_complaint_optout_silences_thread_permanently_and_alerts_operator(tmp_path: Path) -> None:
    from lead_bot.models import IncomingMessage, Intent, LeadState
    from lead_bot.state_machine import LeadQualificationEngine
    from lead_bot.storage import LeadStore

    store = LeadStore(tmp_path / "optout.db")
    try:
        engine = LeadQualificationEngine(store)
        res1 = engine.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="user_optout_1",
                message_id="m_opt_1",
                text="No me escribas mas, darme de baja RGPD",
            )
        )
        assert res1.intent == Intent.COMPLAINT_OPTOUT
        assert res1.session.state == LeadState.SILENCED
        assert res1.reply_text is None
        assert res1.handoff is not None and res1.handoff.reason == "complaint_optout" and res1.handoff.urgent is True

        # Follow-up message remains silent and does not duplicate operator ticket
        res2 = engine.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="user_optout_1",
                message_id="m_opt_2",
                text="Hola, sigo aqui",
            )
        )
        assert res2.session.state == LeadState.SILENCED
        assert res2.reply_text is None
        assert res2.handoff is None
        assert len(store.pending_handoffs()) == 1
    finally:
        store.close()


def test_single_touch_warm_nudge_bounds_hours_and_failed_dispatch_skip(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    from lead_bot.models import IncomingMessage
    from lead_bot.state_machine import LeadQualificationEngine
    from lead_bot.storage import LeadStore

    store = LeadStore(tmp_path / "nudge.db")
    try:
        engine = LeadQualificationEngine(store)
        t0 = datetime(2026, 11, 22, 16, 0, tzinfo=timezone.utc)
        # 2 idle leads (20h old at t_run) + 1 silenced lead (must never be nudged)
        engine.handle_message(
            IncomingMessage(channel="whatsapp", sender_id="lead_fail_1", message_id="n1", text="Need web scraping", received_at=t0)
        )
        engine.handle_message(
            IncomingMessage(channel="telegram", sender_id="lead_ok_2", message_id="n2", text="Necesito un bot de WhatsApp", received_at=t0 + timedelta(minutes=5))
        )
        engine.handle_message(
            IncomingMessage(channel="whatsapp", sender_id="lead_silenced", message_id="n3", text="Please unsubscribe me", received_at=t0)
        )

        # Outside business hours (22:30) -> zero nudges sent
        night = datetime(2026, 11, 23, 22, 30, tzinfo=timezone.utc)
        assert engine.dispatch_warm_nudges(lambda _: True, now=night, max_nudges=2) == []

        # Inside business hours (12:00, 20h idle): first fails dispatch, second succeeds without wasting max_nudges=1
        midday = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)
        attempted: list[str] = []

        def _send(nudge) -> bool:
            attempted.append(nudge.sender_id)
            return nudge.sender_id != "lead_fail_1"

        sent = engine.dispatch_warm_nudges(_send, now=midday, max_nudges=1)
        assert attempted == ["lead_fail_1", "lead_ok_2"]
        assert len(sent) == 1 and sent[0].sender_id == "lead_ok_2" and sent[0].lang == "es"

        # Re-running never nudges lead_ok_2 a second time (single-touch idempotency)
        sent_again = engine.dispatch_warm_nudges(lambda n: n.sender_id == "lead_ok_2", now=midday, max_nudges=5)
        assert sent_again == []
    finally:
        store.close()
