from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .models import HandoffPayload, IncomingMessage, LeadSession, LeadState

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS lead_sessions (
    channel TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    lang TEXT NOT NULL,
    state TEXT NOT NULL,
    service_need TEXT,
    estimated_budget_eur REAL,
    urgent INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (channel, sender_id)
);

CREATE TABLE IF NOT EXISTS processed_messages (
    channel TEXT NOT NULL,
    message_id TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    received_at TEXT NOT NULL,
    PRIMARY KEY (channel, message_id)
);

CREATE TABLE IF NOT EXISTS handoff_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    lang TEXT NOT NULL,
    reason TEXT NOT NULL,
    service_need TEXT,
    estimated_budget_eur REAL,
    urgent INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lead_nudges (
    channel TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    nudged_at TEXT NOT NULL,
    nudge_text TEXT NOT NULL,
    PRIMARY KEY (channel, sender_id)
);
"""


class LeadStore:
    """Concurrent SQLite WAL store with webhook message-id deduplication and atomic handoff queue."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize SQLite WAL database (busy_timeout=5000ms) and ensure session/queue tables exist."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def journal_mode(self) -> str:
        row = self._conn.execute("PRAGMA journal_mode;").fetchone()
        return str(row[0]).lower() if row else ""

    def is_duplicate(self, channel: str, message_id: str) -> bool:
        """Return True if (channel, message_id) has already been processed by a prior webhook delivery."""
        row = self._conn.execute(
            "SELECT 1 FROM processed_messages WHERE channel = ? AND message_id = ?",
            (channel, message_id),
        ).fetchone()
        return row is not None

    def get_session(self, channel: str, sender_id: str) -> LeadSession:
        """Load persisted LeadSession for (channel, sender_id), or return a fresh NEW session."""
        row = self._conn.execute(
            "SELECT channel, sender_id, lang, state, service_need, estimated_budget_eur, urgent, updated_at "
            "FROM lead_sessions WHERE channel = ? AND sender_id = ?",
            (channel, sender_id),
        ).fetchone()
        if row is None:
            return LeadSession(channel=channel, sender_id=sender_id)  # type: ignore[arg-type]
        return LeadSession(
            channel=row["channel"],
            sender_id=row["sender_id"],
            lang=row["lang"],
            state=LeadState(row["state"]),
            service_need=row["service_need"],
            estimated_budget_eur=row["estimated_budget_eur"],
            urgent=bool(row["urgent"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def commit_turn(
        self,
        msg: IncomingMessage,
        session: LeadSession,
        handoff: HandoffPayload | None = None,
    ) -> bool:
        """Atomically record message_id, upsert session state, and enqueue handoff. Returns False on duplicate race."""
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO processed_messages (channel, message_id, sender_id, received_at) VALUES (?, ?, ?, ?)",
                    (msg.channel, msg.message_id, msg.sender_id, msg.received_at.isoformat()),
                )
                self._conn.execute(
                    """
                    INSERT INTO lead_sessions (
                        channel, sender_id, lang, state, service_need, estimated_budget_eur, urgent, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(channel, sender_id) DO UPDATE SET
                        lang = excluded.lang,
                        state = excluded.state,
                        service_need = excluded.service_need,
                        estimated_budget_eur = excluded.estimated_budget_eur,
                        urgent = excluded.urgent,
                        updated_at = excluded.updated_at
                    """,
                    (
                        session.channel,
                        session.sender_id,
                        session.lang,
                        session.state.value,
                        session.service_need,
                        session.estimated_budget_eur,
                        int(session.urgent),
                        session.updated_at.isoformat(),
                    ),
                )
                if handoff is not None:
                    self._conn.execute(
                        """
                        INSERT INTO handoff_queue (
                            channel, sender_id, lang, reason, service_need, estimated_budget_eur, urgent, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            handoff.channel,
                            handoff.sender_id,
                            handoff.lang,
                            handoff.reason,
                            handoff.service_need,
                            handoff.estimated_budget_eur,
                            int(handoff.urgent),
                            msg.received_at.isoformat(),
                        ),
                    )
            return True
        except sqlite3.IntegrityError:
            return False

    def pending_handoffs(self) -> list[dict[str, object]]:
        """Return all queued human-handoff tickets in FIFO order (oldest first)."""
        rows = self._conn.execute(
            "SELECT channel, sender_id, lang, reason, service_need, estimated_budget_eur, urgent, created_at "
            "FROM handoff_queue ORDER BY id ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def find_stale_leads_for_nudge(
        self,
        now: datetime,
        *,
        min_idle_hours: float = 12.0,
        max_idle_hours: float = 36.0,
    ) -> list[LeadSession]:
        """Return unconverted AWAITING_SERVICE/AWAITING_BUDGET leads idle within [min, max] hours and never nudged."""
        rows = self._conn.execute(
            """
            SELECT s.channel, s.sender_id, s.lang, s.state, s.service_need,
                   s.estimated_budget_eur, s.urgent, s.updated_at
            FROM lead_sessions s
            LEFT JOIN lead_nudges n ON n.channel = s.channel AND n.sender_id = s.sender_id
            WHERE n.sender_id IS NULL
              AND s.state IN ('awaiting_service', 'awaiting_budget')
            ORDER BY s.updated_at ASC
            """
        ).fetchall()
        min_sec = min_idle_hours * 3600.0
        max_sec = max_idle_hours * 3600.0
        out: list[LeadSession] = []
        for row in rows:
            updated_at = datetime.fromisoformat(row["updated_at"])
            idle_sec = (now - updated_at).total_seconds()
            if min_sec <= idle_sec <= max_sec:
                out.append(
                    LeadSession(
                        channel=row["channel"],
                        sender_id=row["sender_id"],
                        lang=row["lang"],
                        state=LeadState(row["state"]),
                        service_need=row["service_need"],
                        estimated_budget_eur=row["estimated_budget_eur"],
                        urgent=bool(row["urgent"]),
                        updated_at=updated_at,
                    )
                )
        return out

    def record_nudge_sent(self, channel: str, sender_id: str, nudged_at: datetime, text: str) -> bool:
        """Persist single-touch nudge atomically; returns False if lead was already nudged."""
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO lead_nudges (channel, sender_id, nudged_at, nudge_text) VALUES (?, ?, ?, ?)",
                    (channel, sender_id, nudged_at.isoformat(), text),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> LeadStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
