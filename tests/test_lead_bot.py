from __future__ import annotations

import json
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from lead_bot.cli import main as cli_main
from lead_bot.models import HandoffPayload, IncomingMessage, Intent, LeadSession, LeadState
from lead_bot.nlu import classify_message, extract_budget_eur
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
        ("budget is $1000000", None),
        ("1200000 EUR", None),
        ("1000000€", None),
        ("1000 eurovision", None),
        ("€120.3456", None),
        ("850.50 EUR", 850.5),
        ("1.200,50€", 1200.5),
        ("$1,500.75", 1500.75),
        ("$100000", 100000.0),
        ("$500k", None),
        ("€500k", None),
        ("$500M", None),
        ("500k EUR", None),
        ("500EUR", 500.0),
        ("500€", 500.0),
        ("$1,500.", 1500.0),
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


def test_nlu_word_boundary_no_false_positives() -> None:
    r1 = classify_message("Necesito informacion rapido por favor")
    assert r1.intent == Intent.UNKNOWN
    assert r1.service_need is None

    r2 = classify_message("Do you sell robot vacuum cleaners?")
    assert r2.intent == Intent.UNKNOWN
    assert r2.service_need is None

    r3 = classify_message("Quiero unas botas")
    assert r3.intent == Intent.UNKNOWN
    assert r3.service_need is None

    r4 = classify_message("Hola! Necesito extraer  datos de precios, es urgente!")
    assert r4.intent == Intent.SERVICE_INQUIRY
    assert r4.service_need == "web_scraping"
    assert r4.urgent is True

    r5 = classify_message("Hola!")
    assert r5.intent == Intent.GREETING

    r6 = classify_message("Please don't contact me again")
    assert r6.intent == Intent.COMPLAINT_OPTOUT

    for curly in ("Please don\u2019t contact me again", "Please don\u2018t contact me again"):
        res_curly = classify_message(curly)
        assert res_curly.intent == Intent.COMPLAINT_OPTOUT
        assert res_curly.urgent is True


def test_concurrent_same_sender_turns_atomic_and_dedup(tmp_path: Path) -> None:
    db_file = tmp_path / "race.db"
    with LeadStore(db_file) as store:
        engine = LeadQualificationEngine(store)
        barrier = threading.Barrier(2)

        def _send(msg_id: str, text: str):
            barrier.wait()
            return engine.handle_message(
                IncomingMessage(channel="whatsapp", sender_id="u_race", message_id=msg_id, text=text)
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_send, "m_service", "Quiero un bot de WhatsApp")
            f2 = pool.submit(_send, "m_budget", "Sobre 1.100€")
            r1, r2 = f1.result(), f2.result()

        assert not r1.duplicate and not r2.duplicate
        final = store.get_session("whatsapp", "u_race")
        assert final.state == LeadState.HANDOFF_READY
        assert final.service_need == "messaging_bot"
        assert final.estimated_budget_eur == 1100.0
        assert len(store.pending_handoffs()) == 1

    # 6-thread duplicate message_id storm across shared and separate LeadStore instances
    dedup_db = tmp_path / "dedup_storm.db"
    storm_barrier = threading.Barrier(6)

    def _dup_worker(idx: int):
        storm_barrier.wait()
        with LeadStore(dedup_db) as s:
            eng = LeadQualificationEngine(s)
            return eng.handle_message(
                IncomingMessage(
                    channel="telegram",
                    sender_id="storm_user",
                    message_id="same_msg",
                    text="Need a WhatsApp bot, budget $900",
                )
            )

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_dup_worker, range(6)))

    assert sum(1 for r in results if not r.duplicate) == 1
    assert sum(1 for r in results if r.duplicate) == 5
    with LeadStore(dedup_db) as s:
        assert len(s.pending_handoffs()) == 1


def test_commit_turn_propagates_non_duplicate_integrity_error(tmp_path: Path) -> None:
    with LeadStore(tmp_path / "integrity.db") as store:
        msg = IncomingMessage(channel="web", sender_id="u1", message_id="m_int", text="hola")
        session = LeadSession(channel="web", sender_id="u1", state=LeadState.HANDOFF_READY)
        broken_handoff = HandoffPayload.model_construct(
            sender_id="u1",
            channel=None,
            lang="es",
            reason="qualified_lead",
            service_need="messaging_bot",
            estimated_budget_eur=800.0,
            urgent=False,
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.commit_turn(msg, session, handoff=broken_handoff)


def test_concurrent_dispatch_warm_nudges_single_send_and_exception_release(tmp_path: Path) -> None:
    db_file = tmp_path / "nudge_race.db"
    t0 = datetime(2026, 11, 22, 16, 0, tzinfo=timezone.utc)
    midday = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)

    with LeadStore(db_file) as store:
        engine = LeadQualificationEngine(store)
        engine.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="lead_race",
                message_id="nr1",
                text="Need web scraping",
                received_at=t0,
            )
        )

        barrier = threading.Barrier(4)
        send_calls: list[str] = []
        send_lock = threading.Lock()

        def _slow_send(nudge) -> bool:
            time.sleep(0.02)
            with send_lock:
                send_calls.append(nudge.sender_id)
            return True

        def _worker(_: int):
            barrier.wait()
            return engine.dispatch_warm_nudges(_slow_send, now=midday, max_nudges=5)

        with ThreadPoolExecutor(max_workers=4) as pool:
            batches = list(pool.map(_worker, range(4)))

        assert len(send_calls) == 1
        assert sum(len(b) for b in batches) == 1

    # Exception in send_fn releases claim so a later retry succeeds
    with LeadStore(tmp_path / "nudge_exc.db") as store:
        engine = LeadQualificationEngine(store)
        engine.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="lead_retry",
                message_id="nr2",
                text="Need web scraping",
                received_at=t0,
            )
        )

        def _boom(_nudge) -> bool:
            raise RuntimeError("Gateway timeout")

        with pytest.raises(RuntimeError, match="Gateway timeout"):
            engine.dispatch_warm_nudges(_boom, now=midday, max_nudges=1)

        retried = engine.dispatch_warm_nudges(lambda _: True, now=midday, max_nudges=1)
        assert len(retried) == 1 and retried[0].sender_id == "lead_retry"


def test_mixed_naive_and_offset_datetimes_nudge_and_sort(tmp_path: Path) -> None:
    tz_plus2 = timezone(timedelta(hours=2))
    older_offset = datetime(2026, 11, 22, 17, 30, tzinfo=tz_plus2)  # 15:30 UTC
    newer_naive = datetime(2026, 11, 22, 16, 0)  # naive -> 16:00 UTC

    with LeadStore(tmp_path / "tz_nudge.db") as store:
        engine = LeadQualificationEngine(store)
        engine.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="lead_newer",
                message_id="tz1",
                text="Need web scraping",
                received_at=newer_naive,
            )
        )
        engine.handle_message(
            IncomingMessage(
                channel="telegram",
                sender_id="lead_older",
                message_id="tz2",
                text="Necesito un bot de WhatsApp",
                received_at=older_offset,
            )
        )

        naive_now = datetime(2026, 11, 23, 12, 0)
        stale = store.find_stale_leads_for_nudge(naive_now)
        assert [s.sender_id for s in stale] == ["lead_older", "lead_newer"]

        sent = engine.dispatch_warm_nudges(lambda _: True, now=naive_now, max_nudges=2)
        assert [n.sender_id for n in sent] == ["lead_older", "lead_newer"]


def test_dispatch_warm_nudges_skips_lead_qualified_during_toctou_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_file = tmp_path / "toctou_nudge.db"
    t0 = datetime(2026, 11, 22, 16, 0, tzinfo=timezone.utc)
    midday = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)

    with LeadStore(db_file) as store_a, LeadStore(db_file) as store_b:
        engine_a = LeadQualificationEngine(store_a)
        engine_b = LeadQualificationEngine(store_b)

        engine_a.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="toctou-lead",
                message_id="m1",
                text="Need web scraping",
                received_at=t0,
            )
        )
        assert store_a.record_nudge_sent("whatsapp", "missing-lead", midday, "hi") is False

        orig_find = store_a.find_stale_leads_for_nudge

        def _interleaved_find(now: datetime, **kwargs):
            stale = orig_find(now, **kwargs)
            engine_b.handle_message(
                IncomingMessage(
                    channel="whatsapp",
                    sender_id="toctou-lead",
                    message_id="m2",
                    text="Budget is 1.200 EUR",
                    received_at=midday,
                )
            )
            return stale

        monkeypatch.setattr(store_a, "find_stale_leads_for_nudge", _interleaved_find)
        send_calls: list[str] = []
        sent = engine_a.dispatch_warm_nudges(
            lambda n: send_calls.append(n.sender_id) or True,
            now=midday,
            max_nudges=5,
        )
        assert sent == []
        assert send_calls == []
        assert store_b.get_session("whatsapp", "toctou-lead").state == LeadState.HANDOFF_READY


def test_gdpr_rgpd_inquiry_vs_optout_discrimination(tmp_path: Path) -> None:
    for benign in (
        "¿Vuestro bot cumple con RGPD?",
        "Is this GDPR compliant?",
        "Necesito API workflow con DPA GDPR",
    ):
        res = classify_message(benign)
        assert res.intent != Intent.COMPLAINT_OPTOUT

    with LeadStore(tmp_path / "gdpr.db") as store:
        engine = LeadQualificationEngine(store)
        t = engine.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="gdpr_q",
                message_id="gq1",
                text="¿Vuestro bot cumple con RGPD?",
            )
        )
        assert t.session.state != LeadState.SILENCED

    for idx, optout in enumerate((
        "Quiero darme de baja",
        "Solicito darse de baja",
        "GDPR delete my data",
        "Quiero ejercer derecho al olvido RGPD y borrar mis datos",
        "opt-out please",
    )):
        res = classify_message(optout)
        assert res.intent == Intent.COMPLAINT_OPTOUT, f"Failed opt-out #{idx}: {optout}"


def test_nlu_panel_solar_precios_and_human_multispace_guards() -> None:
    for solar in (
        "Necesito un panel solar",
        "Presupuesto solar panel 5000€",
        "Quiero instalar un panel fotovoltaico",
    ):
        res = classify_message(solar)
        assert res.service_need is None

    assert classify_message("Quiero un panel de control B2B").service_need == "b2b_automation"
    assert classify_message("Need an admin panel").service_need == "b2b_automation"

    g = classify_message("Hola, ¿qué precios tenéis?")
    assert g.intent == Intent.GREETING
    assert g.service_need is None

    bot_q = classify_message("¿Qué precios tenéis para un bot de WhatsApp?")
    assert bot_q.service_need == "messaging_bot"

    scrape_q = classify_message("Necesito monitorizar precios de competencia")
    assert scrape_q.service_need == "web_scraping"

    h1 = classify_message("Quiero hablar   con\tuna  persona")
    assert h1.intent == Intent.HUMAN_ESCALATION
    h2 = classify_message("Can I talk to   a   human?")
    assert h2.intent == Intent.HUMAN_ESCALATION

    not_human = classify_message("Is this real personal data?")
    assert not_human.intent != Intent.HUMAN_ESCALATION


def test_budget_k_m_suffixes_and_multi_amount_skip() -> None:
    cases = [
        ("$3k", 3000.0),
        ("€1.5k", 1500.0),
        ("2,5k€", 2500.0),
        ("10k EUR", 10000.0),
        ("$0.2M", 200000.0),
        ("0.1M€", 100000.0),
        ("$10 setup, total $1,500", 1500.0),
        ("€5/mes o $3k anual", 3000.0),
        ("$500k o $2k", 2000.0),
        ("$49.99 y luego $50", 50.0),
        ("$100 más IVA", 100.0),
    ]
    for raw, expected in cases:
        assert extract_budget_eur(raw) == expected, f"Mismatch for {raw!r}"

    for rejected in ("4k video", "100k followers", "$100M", "$1M", "3kg EUR"):
        assert extract_budget_eur(rejected) is None, f"Should reject {rejected!r}"


def test_seeded_nlu_adversarial_fuzzing() -> None:
    rng = random.Random(20261123)
    spaces = [" ", "\t", "\n", "  "]
    noise_low = ["$5", "€10", "15 EUR", "$49"]
    valid_budgets = [
        ("$3k", 3000.0),
        ("€1.5k", 1500.0),
        ("2,5k€", 2500.0),
        ("0.15M€", 150000.0),
        ("$1,250", 1250.0),
        ("800 euros", 800.0),
    ]
    for _ in range(250):
        sep1 = rng.choice(spaces)
        sep2 = rng.choice(spaces)
        b_str, b_val = rng.choice(valid_budgets)
        prefix = f"Setup {rng.choice(noise_low)}{sep1}y{sep2}" if rng.random() < 0.7 else ""
        msg = f"Hola{sep1}¿cumple con RGPD?{sep2}Presupuesto{sep1}{prefix}{b_str}"
        if rng.random() < 0.3:
            msg = msg.replace("'", "\u2019")
        res = classify_message(msg)
        assert res.intent != Intent.COMPLAINT_OPTOUT
        assert res.service_need is None
        assert res.budget_eur == b_val


@pytest.mark.parametrize("bad_text", [" ", "\t\n", "\u00a0  "])
def test_incoming_message_rejects_whitespace_only(bad_text: str) -> None:
    with pytest.raises(ValidationError):
        IncomingMessage(channel="whatsapp", sender_id="u1", message_id="m1", text=bad_text)


@pytest.mark.parametrize("bad_id", ["  ", " a ", "\t\n"])
def test_incoming_message_rejects_blank_or_short_ids(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        IncomingMessage(channel="whatsapp", sender_id=bad_id, message_id="m1", text="hola")
    with pytest.raises(ValidationError):
        IncomingMessage(channel="whatsapp", sender_id="u1", message_id=bad_id, text="hola")


def test_out_of_order_messages_preserve_monotonic_updated_at_and_slots(tmp_path: Path) -> None:
    t1 = datetime(2026, 11, 22, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
    with LeadStore(tmp_path / "ooo.db") as store:
        engine = LeadQualificationEngine(store)
        r_new = engine.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="u_ooo",
                message_id="m_new",
                text="Necesito un bot de WhatsApp, presupuesto 1.500€",
                received_at=t2,
            )
        )
        assert r_new.session.state == LeadState.HANDOFF_READY
        assert r_new.session.lang == "es"
        assert r_new.session.updated_at == t2

        # Delayed older message in English for another service must not clobber updated_at/lang/slots
        r_old = engine.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="u_ooo",
                message_id="m_old",
                text="Hi, looking for a web scraper around $800",
                received_at=t1,
            )
        )
        assert r_old.session.updated_at == t2
        assert r_old.session.lang == "es"
        assert r_old.session.service_need == "messaging_bot"
        assert r_old.session.estimated_budget_eur == 1500.0
        assert len(store.pending_handoffs()) == 1

        # Direct store.commit_turn regression check on monotonic updated_at
        stale_session = r_new.session.model_copy(update={"updated_at": t1})
        stale_msg = IncomingMessage(channel="whatsapp", sender_id="u_ooo", message_id="m_direct", text="hi", received_at=t1)
        assert store.commit_turn(stale_msg, stale_session) is True
        assert store.get_session("whatsapp", "u_ooo").updated_at == t2


def test_warm_nudge_europe_madrid_cet_cest_and_dst_boundaries(tmp_path: Path) -> None:
    # Winter CET (UTC+1): 08:30 UTC == 09:30 CET, 20:00 UTC == 21:00 CET
    # Summer CEST (UTC+2): 07:30 UTC == 09:30 CEST, 19:00 UTC == 21:00 CEST
    # DST transitions in 2026: March 29 (spring forward), October 25 (fall back)
    cases = [
        (datetime(2026, 1, 15, 8, 29, tzinfo=timezone.utc), False),
        (datetime(2026, 1, 15, 8, 30, tzinfo=timezone.utc), True),
        (datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc), True),
        (datetime(2026, 1, 15, 20, 1, tzinfo=timezone.utc), False),
        (datetime(2026, 7, 15, 7, 29, tzinfo=timezone.utc), False),
        (datetime(2026, 7, 15, 7, 30, tzinfo=timezone.utc), True),
        (datetime(2026, 7, 15, 19, 0, tzinfo=timezone.utc), True),
        (datetime(2026, 7, 15, 19, 1, tzinfo=timezone.utc), False),
        (datetime(2026, 3, 29, 7, 30, tzinfo=timezone.utc), True),
        (datetime(2026, 10, 25, 8, 30, tzinfo=timezone.utc), True),
        (datetime(2026, 10, 25, 8, 29, tzinfo=timezone.utc), False),
    ]
    with LeadStore(tmp_path / "dst.db") as store:
        engine = LeadQualificationEngine(store)
        for idx, (now_dt, should_send) in enumerate(cases):
            sender = f"u_dst_{idx}"
            engine.handle_message(
                IncomingMessage(
                    channel="whatsapp",
                    sender_id=sender,
                    message_id=f"m_dst_{idx}",
                    text="Need web scraping",
                    received_at=now_dt - timedelta(hours=18),
                )
            )
            sent = engine.dispatch_warm_nudges(lambda n, s=sender: n.sender_id == s, now=now_dt, max_nudges=5)
            assert any(n.sender_id == sender for n in sent) is should_send, (
                f"Failed DST boundary at {now_dt.isoformat()}"
            )


def test_warm_nudge_toctou_fresh_reply_skips_nudge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_file = tmp_path / "toctou_fresh.db"
    t0 = datetime(2026, 11, 22, 16, 0, tzinfo=timezone.utc)
    midday = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)

    with LeadStore(db_file) as store_a, LeadStore(db_file) as store_b:
        engine_a = LeadQualificationEngine(store_a)
        engine_b = LeadQualificationEngine(store_b)
        engine_a.handle_message(
            IncomingMessage(
                channel="whatsapp",
                sender_id="fresh-lead",
                message_id="m1",
                text="Need web scraping",
                received_at=t0,
            )
        )
        orig_find = store_a.find_stale_leads_for_nudge

        def _interleaved_reply(now: datetime, **kwargs):
            stale = orig_find(now, **kwargs)
            engine_b.handle_message(
                IncomingMessage(
                    channel="whatsapp",
                    sender_id="fresh-lead",
                    message_id="m2",
                    text="Hola",
                    received_at=midday,
                )
            )
            return stale

        monkeypatch.setattr(store_a, "find_stale_leads_for_nudge", _interleaved_reply)
        sent = engine_a.dispatch_warm_nudges(lambda _: True, now=midday, max_nudges=5)
        assert sent == []


def test_high_concurrency_16_thread_barrier_leadstore_and_nudge_storm(tmp_path: Path) -> None:
    db_file = tmp_path / "barrier16.db"
    barrier = threading.Barrier(16)

    def _worker(idx: int) -> list[bool]:
        barrier.wait()
        with LeadStore(db_file) as store:
            eng = LeadQualificationEngine(store)
            r1 = eng.handle_message(
                IncomingMessage(
                    channel="whatsapp",
                    sender_id=f"u_{idx % 4}",
                    message_id=f"uniq_{idx}",
                    text="Necesito un bot de WhatsApp, presupuesto 1.000€",
                )
            )
            r_dup = eng.handle_message(
                IncomingMessage(
                    channel="telegram",
                    sender_id="shared_dup",
                    message_id="shared_msg_id",
                    text="Need a scraper, budget $900",
                )
            )
            return [r1.duplicate, r_dup.duplicate]

    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(_worker, range(16)))

    assert all(o[0] is False for o in outcomes)
    assert sum(1 for o in outcomes if o[1] is False) == 1
    assert sum(1 for o in outcomes if o[1] is True) == 15

    # 8-thread warm nudge storm over 3 stale leads
    storm_db = tmp_path / "nudge_storm8.db"
    t0 = datetime(2026, 11, 22, 16, 0, tzinfo=timezone.utc)
    midday = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)
    with LeadStore(storm_db) as store:
        eng = LeadQualificationEngine(store)
        for i in range(3):
            eng.handle_message(
                IncomingMessage(
                    channel="whatsapp",
                    sender_id=f"stale_{i}",
                    message_id=f"sm_{i}",
                    text="Need web scraping",
                    received_at=t0,
                )
            )

    b8 = threading.Barrier(8)

    def _nudge_worker(_: int) -> int:
        b8.wait()
        with LeadStore(storm_db) as s:
            return len(LeadQualificationEngine(s).dispatch_warm_nudges(lambda _: True, now=midday, max_nudges=10))

    with ThreadPoolExecutor(max_workers=8) as pool:
        sent_counts = list(pool.map(_nudge_worker, range(8)))

    assert sum(sent_counts) == 3
