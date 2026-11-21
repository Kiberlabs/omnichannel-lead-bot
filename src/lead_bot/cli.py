from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import IncomingMessage
from .state_machine import LeadQualificationEngine
from .storage import LeadStore


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: replays a JSON script of webhooks through LeadQualificationEngine and prints summary/JSON."""
    parser = argparse.ArgumentParser(prog="lead-bot", description="KiberLabs 24/7 Omnichannel Lead Qualifier")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("simulate", help="Replay a JSON script of inbound WhatsApp/Telegram/Web messages")
    sim.add_argument("--script", type=Path, required=True, help="Path to JSON array of inbound messages")
    sim.add_argument("--db", type=Path, default=Path("data/leads.db"), help="SQLite WAL database path")
    sim.add_argument("--json", action="store_true", help="Output structured JSON transcript and handoffs")

    args = parser.parse_args(argv)
    raw_messages = json.loads(args.script.read_text(encoding="utf-8"))
    messages = [IncomingMessage.model_validate(item) for item in raw_messages]

    turns_out: list[dict[str, object]] = []
    with LeadStore(args.db) as store:
        engine = LeadQualificationEngine(store)
        for msg in messages:
            res = engine.handle_message(msg)
            turns_out.append({
                "message_id": msg.message_id,
                "sender_id": msg.sender_id,
                "duplicate": res.duplicate,
                "intent": res.intent.value,
                "state": res.session.state.value,
                "reply_text": res.reply_text,
                "handoff": res.handoff.model_dump() if res.handoff else None,
            })
        handoffs = store.pending_handoffs()

    if args.json:
        print(json.dumps({"turns": turns_out, "handoff_queue": handoffs}, indent=2, ensure_ascii=False))
    else:
        print(f"[KiberLabs LeadBot] Processed={len(turns_out)} turns | Queued Handoffs={len(handoffs)}")
        for t in turns_out:
            status = "DUP-SKIPPED" if t["duplicate"] else str(t["state"]).upper()
            print(f"  [{status}] ({t['sender_id']}) -> {t['reply_text'] or '<silent>'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
