"""
NOVA — Task backlog + append-only episodic log (SQLite, project guide §9).

Two tables in one local DB:
  tasks         — the work queue; rows update as status changes.
  episodic_log  — append-only forever, enforced in two independent layers
                  (SQLite has no GRANT system, so these stand in for a
                  missing-DELETE-grant):
                  schema level — triggers reject UPDATE/DELETE for ANY
                  connection to the file, including ones this code never sees;
                  connection level — an authorizer denies UPDATE/DELETE at
                  statement-prepare time on the application's own connection.
                  No update/delete API exists on top of either.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

DEFAULT_DB = Path(os.environ.get("NOVA_DB", Path(__file__).parent / "nova.db"))

# The one sanctioned wording for an owner abort (§14). Defined HERE — not in
# nodes/common.py — because both writers already import backlog: the dashboard
# endpoint (tasks.final_output + the task_aborted audit entry) and the graph's
# aborted_update (error_log, which drive_task then re-writes to final_output).
# Importing nodes.common instead would drag litellm into the dashboard
# process (measured: +161 MB RSS, +2 s startup) for one string.
ABORT_REASON = "Aborted by user via dashboard"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    privacy_sensitive INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    final_output TEXT,
    abort_requested INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runtime_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    updated_at TEXT NOT NULL,
    tick INTEGER NOT NULL,
    interval_seconds REAL NOT NULL,
    gemini_used INTEGER NOT NULL,
    gemini_limit INTEGER NOT NULL,
    groq_used INTEGER NOT NULL,
    groq_limit INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS approval_decisions (
    task_id TEXT PRIMARY KEY,
    approved INTEGER NOT NULL,
    decided_at TEXT NOT NULL,
    applied_at TEXT
);
CREATE TABLE IF NOT EXISTS ingested_emails (
    message_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodic_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    task_id TEXT,
    actor TEXT NOT NULL,
    action_type TEXT NOT NULL,
    detail TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS episodic_log_no_update
    BEFORE UPDATE ON episodic_log
    BEGIN SELECT RAISE(ABORT, 'episodic_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS episodic_log_no_delete
    BEFORE DELETE ON episodic_log
    BEGIN SELECT RAISE(ABORT, 'episodic_log is append-only'); END;
"""


class TaskRecord(BaseModel):
    task_id: str
    goal: str
    source: str = "manual"
    privacy_sensitive: bool = False
    status: str = "pending"


class EpisodicEntry(BaseModel):
    """One external action or lifecycle event, with enough detail to
    reconstruct why it happened (project guide §4)."""
    task_id: Optional[str]
    actor: str          # agent role or "system" / "human"
    action_type: str    # e.g. "llm_call", "file_write", "app_launch", "email_fetch"
    detail: dict = Field(default_factory=dict)
    reason: str


def _deny_episodic_mutation(action: int, arg1, arg2, db_name, trigger_name) -> int:
    if (action in (sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE)
            and arg1 == "episodic_log"):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


class Backlog:
    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # Migrations for DBs created before these columns existed — CREATE
        # TABLE IF NOT EXISTS never adds columns to an existing table.
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(tasks)")}
        if "final_output" not in cols:
            self._conn.execute("ALTER TABLE tasks ADD COLUMN final_output TEXT")
        if "abort_requested" not in cols:
            self._conn.execute(
                "ALTER TABLE tasks ADD COLUMN abort_requested INTEGER NOT NULL DEFAULT 0")
        self._conn.commit()
        # After the schema exists: from here on this connection cannot even
        # prepare an UPDATE/DELETE against the episodic log.
        self._conn.set_authorizer(_deny_episodic_mutation)

    # --- task queue ---

    def add_task(self, task: TaskRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO tasks (task_id, goal, source, privacy_sensitive, "
            "status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (task.task_id, task.goal, task.source, int(task.privacy_sensitive),
             task.status, now, now),
        )
        self._conn.commit()
        self.log(EpisodicEntry(task_id=task.task_id, actor="system",
                               action_type="task_created",
                               detail=task.model_dump(), reason=f"new task from {task.source}"))

    def update_status(self, task_id: str, status: str,
                      final_output: Optional[str] = None) -> None:
        """final_output, when given (terminal states), is persisted with the
        status so read-only consumers (the dashboard chat history) can show a
        task's result without opening the graph checkpoint. For failed tasks
        the callers store the last error here — it is 'NOVA's reply' either way."""
        now = datetime.now(timezone.utc).isoformat()
        if final_output is None:
            self._conn.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
                (status, now, task_id))
        else:
            self._conn.execute(
                "UPDATE tasks SET status=?, updated_at=?, final_output=? WHERE task_id=?",
                (status, now, final_output, task_id))
        self._conn.commit()
        self.log(EpisodicEntry(task_id=task_id, actor="system",
                               action_type="status_change", detail={"status": status},
                               reason="task lifecycle update"))

    @staticmethod
    def _to_record(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(task_id=row["task_id"], goal=row["goal"], source=row["source"],
                          privacy_sensitive=bool(row["privacy_sensitive"]), status=row["status"])

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return self._to_record(row) if row is not None else None

    def pending_tasks(self) -> list[TaskRecord]:
        return self.tasks_with_status("pending")

    def tasks_with_status(self, *statuses: str) -> list[TaskRecord]:
        """Tasks in any of the given statuses, oldest first — the loop uses
        this at startup to pick up work a previous process left unfinished."""
        marks = ",".join("?" * len(statuses))
        rows = self._conn.execute(
            f"SELECT * FROM tasks WHERE status IN ({marks}) ORDER BY created_at",
            statuses).fetchall()
        return [self._to_record(r) for r in rows]

    # --- email ingestion dedup ---
    # Mail.Read is read-only by construction, so messages stay "unread" in the
    # mailbox and every poll re-fetches them; this table is what stops one
    # email from becoming a new task per poll.

    def recent_chat_turns(self, exclude_task_id: Optional[str] = None,
                          limit: int = 8) -> list[tuple[str, str]]:
        """The last `limit` COMPLETED conversation turns (manual + tutor
        tasks, status done, reply present), oldest-first, as (goal,
        final_output) pairs. This is the §15 chat-context window: every
        task starts stateless, so the tutor rebuilds continuity from here.
        Failed turns are excluded — their final_output is an error string,
        noise as conversation."""
        rows = self._conn.execute(
            "SELECT goal, final_output FROM ("
            "  SELECT goal, final_output, created_at, task_id FROM tasks "
            "  WHERE source IN ('manual', 'tutor') AND status = 'done' "
            "    AND final_output IS NOT NULL AND task_id != ? "
            "  ORDER BY created_at DESC, task_id DESC LIMIT ?"
            ") ORDER BY created_at ASC, task_id ASC",
            (exclude_task_id or "", limit)).fetchall()
        return [(r["goal"], r["final_output"]) for r in rows]

    # --- abort signalling (§13/§14) ---
    # The dashboard's POST /api/task/abort runs in a different PROCESS from
    # the loop holding the in-flight model call, so the signal must be
    # durable: a flag on the task row, polled by llm.complete() between
    # stream chunks. Never cleared — task_ids are never reused, and a stale
    # flag on a terminal task can never fire again.

    def request_abort(self, task_id: str) -> None:
        self._conn.execute(
            "UPDATE tasks SET abort_requested=1, updated_at=? WHERE task_id=?",
            (datetime.now(timezone.utc).isoformat(), task_id))
        self._conn.commit()

    def abort_requested(self, task_id: str) -> bool:
        row = self._conn.execute(
            "SELECT abort_requested FROM tasks WHERE task_id=?",
            (task_id,)).fetchone()
        return bool(row and row["abort_requested"])

    def email_already_ingested(self, message_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM ingested_emails WHERE message_id=?", (message_id,)).fetchone()
        return row is not None

    def record_email_ingested(self, message_id: str, task_id: str) -> None:
        self._conn.execute(
            "INSERT INTO ingested_emails VALUES (?,?,?)",
            (message_id, task_id, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    # --- loop heartbeat (single row; the dashboard reads liveness/quota) ---

    def write_heartbeat(self, tick: int, interval_seconds: float,
                        gemini_used: int, gemini_limit: int,
                        groq_used: int, groq_limit: int) -> None:
        self._conn.execute(
            "INSERT INTO runtime_status (id, updated_at, tick, interval_seconds, "
            "gemini_used, gemini_limit, groq_used, groq_limit) "
            "VALUES (1,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at, "
            "tick=excluded.tick, interval_seconds=excluded.interval_seconds, "
            "gemini_used=excluded.gemini_used, gemini_limit=excluded.gemini_limit, "
            "groq_used=excluded.groq_used, groq_limit=excluded.groq_limit",
            (datetime.now(timezone.utc).isoformat(), tick, interval_seconds,
             gemini_used, gemini_limit, groq_used, groq_limit),
        )
        self._conn.commit()

    def read_heartbeat(self) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM runtime_status WHERE id=1").fetchone()
        return dict(row) if row is not None else None

    # --- headless approval decisions (project guide §13) ---
    # One-shot `nova approve/reject` processes write here; the running loop
    # polls and applies on its tick cadence. No stdin involved anywhere.

    def record_decision(self, task_id: str, approved: bool) -> None:
        """Last decision wins, and re-deciding re-opens an applied row —
        a task can legitimately reach the gate more than once."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO approval_decisions (task_id, approved, decided_at, applied_at) "
            "VALUES (?,?,?,NULL) "
            "ON CONFLICT(task_id) DO UPDATE SET approved=excluded.approved, "
            "decided_at=excluded.decided_at, applied_at=NULL",
            (task_id, int(approved), now),
        )
        self._conn.commit()
        self.log(EpisodicEntry(task_id=task_id, actor="human",
                               action_type="approval_decision",
                               detail={"approved": approved},
                               reason="owner decision recorded by nova approve/reject"))

    def unapplied_decision(self, task_id: str) -> Optional[bool]:
        row = self._conn.execute(
            "SELECT approved FROM approval_decisions "
            "WHERE task_id=? AND applied_at IS NULL", (task_id,)).fetchone()
        return bool(row["approved"]) if row is not None else None

    def unapplied_decisions(self) -> list[tuple[str, bool]]:
        rows = self._conn.execute(
            "SELECT task_id, approved FROM approval_decisions "
            "WHERE applied_at IS NULL ORDER BY decided_at").fetchall()
        return [(r["task_id"], bool(r["approved"])) for r in rows]

    def mark_decision_applied(self, task_id: str) -> None:
        self._conn.execute(
            "UPDATE approval_decisions SET applied_at=? WHERE task_id=?",
            (datetime.now(timezone.utc).isoformat(), task_id),
        )
        self._conn.commit()

    # --- episodic log (append-only) ---

    def log(self, entry: EpisodicEntry) -> None:
        self._conn.execute(
            "INSERT INTO episodic_log (timestamp, task_id, actor, action_type, detail, reason) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), entry.task_id, entry.actor,
             entry.action_type, json.dumps(entry.detail, default=str), entry.reason),
        )
        self._conn.commit()

    def log_entries(self, task_id: Optional[str] = None) -> list[dict]:
        q = "SELECT * FROM episodic_log"
        params: tuple = ()
        if task_id:
            q += " WHERE task_id=?"
            params = (task_id,)
        return [dict(r) for r in self._conn.execute(q + " ORDER BY id", params).fetchall()]

    def close(self) -> None:
        self._conn.close()
