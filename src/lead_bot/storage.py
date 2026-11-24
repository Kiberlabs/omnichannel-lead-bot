from __future__ import annotations

import random
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .models import HandoffPayload, IncomingMessage, LeadSession, LeadState, ensure_utc

_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


def _lock_for_path(db_path: Path) -> threading.RLock:
    if str(db_path) == ":memory:":
        return threading.RLock()
    key = db_path.resolve()
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock

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
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _lock_for_path(self.db_path)
        self._tx_depth = 0
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        with self._lock:
            for attempt in range(20):
                try:
                    if self.journal_mode() != "wal":
                        self._conn.execute("PRAGMA journal_mode=WAL;")
                    self._conn.execute("PRAGMA synchronous=NORMAL;")
                    self._conn.executescript(SCHEMA_SQL)
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == 19:
                        raise
                    time.sleep(0.02 + random.uniform(0.0, 0.03))

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Serialize read-modify-write turns across threads and connections via BEGIN IMMEDIATE."""
        with self._lock:
            is_outer = self._tx_depth == 0
            if is_outer:
                for attempt in range(20):
                    try:
                        self._conn.execute("BEGIN IMMEDIATE;")
                        break
                    except sqlite3.OperationalError as exc:
                        if "locked" not in str(exc).lower() or attempt == 19:
                            raise
                        time.sleep(0.02 + random.uniform(0.0, 0.03))
            self._tx_depth += 1
            try:
                yield
            except BaseException:
                if is_outer and self._conn.in_transaction:
                    self._conn.execute("ROLLBACK;")
                raise
            else:
                if is_outer and self._conn.in_transaction:
                    self._conn.execute("COMMIT;")
            finally:
                self._tx_depth -= 1

    def journal_mode(self) -> str:
        with self._lock:
            row = self._conn.execute("PRAGMA journal_mode;").fetchone()
            return str(row[0]).lower() if row else ""

    def is_duplicate(self, channel: str, message_id: str) -> bool:
        """Return True if (channel, message_id) has already been processed by a prior webhook delivery."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM processed_messages WHERE channel = ? AND message_id = ?",
                (channel, message_id),
            ).fetchone()
            return row is not None

    def get_persisted_session(self, channel: str, sender_id: str) -> LeadSession | None:
        """Load persisted LeadSession for (channel, sender_id), or None if no row exists yet."""
        with self._lock:
            row = self._conn.execute(
                "SELECT channel, sender_id, lang, state, service_need, estimated_budget_eur, urgent, updated_at "
                "FROM lead_sessions WHERE channel = ? AND sender_id = ?",
                (channel, sender_id),
            ).fetchone()
        if row is None:
            return None
        return LeadSession(
            channel=row["channel"],
            sender_id=row["sender_id"],
            lang=row["lang"],
            state=LeadState(row["state"]),
            service_need=row["service_need"],
            estimated_budget_eur=row["estimated_budget_eur"],
            urgent=bool(row["urgent"]),
            updated_at=ensure_utc(datetime.fromisoformat(row["updated_at"])),
        )

    def get_session(self, channel: str, sender_id: str) -> LeadSession:
        """Load persisted LeadSession for (channel, sender_id), or return a fresh NEW session."""
        persisted = self.get_persisted_session(channel, sender_id)
        if persisted is not None:
            return persisted
        return LeadSession(channel=channel, sender_id=sender_id)  # type: ignore[arg-type]

    def commit_turn(
        self,
        msg: IncomingMessage,
        session: LeadSession,
        handoff: HandoffPayload | None = None,
    ) -> bool:
        """Atomically record message_id, upsert session state, and enqueue handoff. Returns False on duplicate race."""
        received_iso = ensure_utc(msg.received_at).isoformat(timespec="microseconds")
        updated_iso = ensure_utc(session.updated_at).isoformat(timespec="microseconds")
        with self.transaction():
            cur = self._conn.execute(
                "INSERT INTO processed_messages (channel, message_id, sender_id, received_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(channel, message_id) DO NOTHING",
                (msg.channel, msg.message_id, msg.sender_id, received_iso),
            )
            if cur.rowcount == 0:
                return False
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
                    updated_at = MAX(lead_sessions.updated_at, excluded.updated_at)
                """,
                (
                    session.channel,
                    session.sender_id,
                    session.lang,
                    session.state.value,
                    session.service_need,
                    session.estimated_budget_eur,
                    int(session.urgent),
                    updated_iso,
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
                        received_iso,
                    ),
                )
        return True

    def pending_handoffs(self) -> list[dict[str, object]]:
        """Return all queued human-handoff tickets in FIFO order (oldest first)."""
        with self._lock:
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
        now_utc = ensure_utc(now)
        with self._lock:
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
            updated_at = ensure_utc(datetime.fromisoformat(row["updated_at"]))
            idle_sec = (now_utc - updated_at).total_seconds()
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

    def record_nudge_sent(
        self,
        channel: str,
        sender_id: str,
        nudged_at: datetime,
        text: str,
        *,
        min_idle_hours: float | None = None,
        max_idle_hours: float | None = None,
    ) -> bool:
        """Persist single-touch nudge atomically; returns False if already nudged, not awaiting, or outside idle bounds."""
        now_utc = ensure_utc(nudged_at)
        nudged_iso = now_utc.isoformat(timespec="microseconds")
        with self.transaction():
            if min_idle_hours is not None and max_idle_hours is not None:
                row = self._conn.execute(
                    "SELECT state, updated_at FROM lead_sessions WHERE channel = ? AND sender_id = ?",
                    (channel, sender_id),
                ).fetchone()
                if row is None or row["state"] not in ("awaiting_service", "awaiting_budget"):
                    return False
                idle_sec = (now_utc - ensure_utc(datetime.fromisoformat(row["updated_at"]))).total_seconds()
                if not (min_idle_hours * 3600.0 <= idle_sec <= max_idle_hours * 3600.0):
                    return False
            cur = self._conn.execute(
                """
                INSERT INTO lead_nudges (channel, sender_id, nudged_at, nudge_text)
                SELECT ?, ?, ?, ?
                WHERE EXISTS (
                    SELECT 1 FROM lead_sessions
                    WHERE channel = ? AND sender_id = ?
                      AND state IN ('awaiting_service', 'awaiting_budget')
                )
                ON CONFLICT(channel, sender_id) DO NOTHING
                """,
                (channel, sender_id, nudged_iso, text, channel, sender_id),
            )
            return cur.rowcount > 0

    def release_nudge(self, channel: str, sender_id: str) -> None:
        """Release a pre-claimed nudge slot if outbound delivery failed."""
        with self.transaction():
            self._conn.execute(
                "DELETE FROM lead_nudges WHERE channel = ? AND sender_id = ?",
                (channel, sender_id),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> LeadStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
