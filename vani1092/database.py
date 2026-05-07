"""SQLite database — one file, zero setup."""

import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from .config import DATABASE_PATH


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY,
                caller_number TEXT,
                language TEXT,
                status TEXT DEFAULT 'active',
                transcript TEXT DEFAULT '[]',
                summary TEXT,
                intent TEXT,
                urgency TEXT,
                location TEXT,
                confidence REAL,
                confirmation_given INTEGER DEFAULT 0,
                operator_id TEXT,
                handoff_payload TEXT,
                created_at TEXT,
                ended_at TEXT
            );

            CREATE TABLE IF NOT EXISTS operators (
                id TEXT PRIMARY KEY,
                name TEXT,
                languages TEXT DEFAULT '[]',
                status TEXT DEFAULT 'online',
                current_call_id TEXT
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT,
                operator_id TEXT,
                ai_was_correct INTEGER,
                corrected_intent TEXT,
                notes TEXT,
                created_at TEXT
            );
        """)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_call(call_id: str, caller_number: str = "", language: str = ""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO calls (id, caller_number, language, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (call_id, caller_number, language, "active", datetime.now(timezone.utc).isoformat()),
        )


def update_call(call_id: str, **kwargs):
    with get_conn() as conn:
        fields = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [call_id]
        conn.execute(f"UPDATE calls SET {fields} WHERE id = ?", values)


def get_call(call_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
        return dict(row) if row else None


def list_active_calls():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM calls WHERE status IN ('active', 'waiting_operator') ORDER BY CASE urgency WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END, created_at"
        ).fetchall()
        return [dict(r) for r in rows]


def add_transcript_turn(call_id: str, speaker: str, text: str):
    call = get_call(call_id)
    if not call:
        return
    transcript = json.loads(call.get("transcript", "[]"))
    transcript.append({"speaker": speaker, "text": text, "time": datetime.now(timezone.utc).isoformat()})
    update_call(call_id, transcript=json.dumps(transcript))


def create_operator(op_id: str, name: str, languages: list):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO operators (id, name, languages, status) VALUES (?, ?, ?, ?)",
            (op_id, name, json.dumps(languages), "online"),
        )


def list_operators():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM operators").fetchall()
        return [dict(r) for r in rows]


def assign_call_to_operator(call_id: str, operator_id: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE calls SET operator_id = ?, status = 'with_operator' WHERE id = ?",
            (operator_id, call_id),
        )
        conn.execute(
            "UPDATE operators SET current_call_id = ?, status = 'busy' WHERE id = ?",
            (call_id, operator_id),
        )


def add_feedback(call_id: str, operator_id: str, ai_was_correct: bool, corrected_intent: str = "", notes: str = ""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (call_id, operator_id, ai_was_correct, corrected_intent, notes, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (call_id, operator_id, int(ai_was_correct), corrected_intent, notes, datetime.now(timezone.utc).isoformat()),
        )
