"""
NOVA — External-action audit logging (project guide §4).

One function, `log_action()`, called by every node/tool at the moment an
external effect happens (LLM API call, file write, app launch, email fetch/
send). It writes to the append-only episodic_log in the shared backlog DB.

Kept separate from backlog.py's task-queue concerns so tools can log without
knowing anything about queue semantics.
"""

from __future__ import annotations

from typing import Optional

from backlog import Backlog, EpisodicEntry

_backlog: Optional[Backlog] = None


def configure(backlog: Backlog) -> None:
    """Wire the audit log to a Backlog instance (done once at startup;
    tests inject a temp-file DB here)."""
    global _backlog
    _backlog = backlog


def log_action(task_id: Optional[str], actor: str, action_type: str,
               detail: dict, reason: str) -> None:
    """Never raises: an audit-log failure must not mask or abort the action's
    own error handling — but it is loudly visible on stderr."""
    entry = EpisodicEntry(task_id=task_id, actor=actor, action_type=action_type,
                          detail=detail, reason=reason)
    if _backlog is None:
        import sys
        print(f"[action_log] UNCONFIGURED, dropping to stderr: {entry.model_dump()}",
              file=sys.stderr)
        return
    try:
        _backlog.log(entry)
    except Exception as e:  # noqa: BLE001
        import sys
        print(f"[action_log] write failed ({e}): {entry.model_dump()}", file=sys.stderr)
