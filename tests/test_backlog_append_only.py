"""Episodic log is append-only in two independent layers (project guide §4/§9),
neither of which is convention in app code:
  connection level — the authorizer on the application's own connection
  refuses to even PREPARE an UPDATE/DELETE ("not authorized");
  schema level — the triggers travel with the DB file and reject tampering
  from ANY connection, including ones the application never created."""

import sqlite3

import pytest

from backlog import EpisodicEntry


def _seed(backlog) -> None:
    backlog.log(EpisodicEntry(task_id="t1", actor="executor",
                              action_type="file_write", detail={"path": "a.txt"},
                              reason="test entry"))


def test_update_denied_at_connection_level(temp_backlog):
    _seed(temp_backlog)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        temp_backlog._conn.execute("UPDATE episodic_log SET reason='rewritten history'")


def test_delete_denied_at_connection_level(temp_backlog):
    _seed(temp_backlog)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        temp_backlog._conn.execute("DELETE FROM episodic_log")


def test_schema_triggers_reject_tampering_from_a_foreign_connection(temp_backlog):
    """A raw second connection to the same file — no authorizer, no app code —
    still cannot rewrite history: the enforcement lives in the DB itself."""
    _seed(temp_backlog)
    raw = sqlite3.connect(str(temp_backlog.db_path))
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            raw.execute("DELETE FROM episodic_log")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            raw.execute("UPDATE episodic_log SET reason='rewritten history'")
    finally:
        raw.close()


def test_log_still_appendable_after_rejected_tampering(temp_backlog):
    _seed(temp_backlog)
    try:
        temp_backlog._conn.execute("DELETE FROM episodic_log WHERE task_id='t1'")
    except sqlite3.DatabaseError:
        pass
    _seed(temp_backlog)
    assert len(temp_backlog.log_entries(task_id="t1")) == 2  # nothing lost, append works


def test_task_lifecycle_is_logged(temp_backlog):
    from backlog import TaskRecord
    temp_backlog.add_task(TaskRecord(task_id="t-life", goal="do a thing"))
    temp_backlog.update_status("t-life", "done")
    kinds = [e["action_type"] for e in temp_backlog.log_entries(task_id="t-life")]
    assert kinds == ["task_created", "status_change"]


def test_tasks_table_remains_updatable(temp_backlog):
    """Only the episodic log is frozen — the work queue must keep working."""
    from backlog import TaskRecord
    temp_backlog.add_task(TaskRecord(task_id="t-q", goal="g"))
    temp_backlog.update_status("t-q", "in_progress")
    assert temp_backlog.get_task("t-q").status == "in_progress"
