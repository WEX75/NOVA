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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    privacy_sensitive INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
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
        self._conn.commit()
        # After the schema exists: from here on this connection cannot even
        # prepare an UPDATE/DELETE against the episodic log.
        self._conn.set_authorizer(_deny_episodic_mutation)

    # --- task queue ---

    def add_task(self, task: TaskRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO tasks VALUES (?,?,?,?,?,?,?)",
            (task.task_id, task.goal, task.source, int(task.privacy_sensitive),
             task.status, now, now),
        )
        self._conn.commit()
        self.log(EpisodicEntry(task_id=task.task_id, actor="system",
                               action_type="task_created",
                               detail=task.model_dump(), reason=f"new task from {task.source}"))

    def update_status(self, task_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
            (status, datetime.now(timezone.utc).isoformat(), task_id),
        )
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
