"""Graceful shutdown under REAL signals in a real loop process: SIGTERM
mid-task must leave a durable, resumable checkpoint, and
a fresh process must pick the task up where it stopped — the same guarantee
test_graph_interrupt.py proves in-process, here proven across an actual
process boundary with actual signal delivery.

These tests spawn tests/loop_harness.py as a subprocess; the LLM and inbox
are faked inside it, everything else (signals, SQLite, checkpoints) is real.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from backlog import Backlog

HARNESS = Path(__file__).parent / "loop_harness.py"

VALID_EXECUTOR_REPLY = json.dumps(
    {"tool_calls": [], "summary": "triaged: revenue noted"})


def _spawn(tmp_path, reply: str, delay: str = "0", max_ticks: str | None = None,
           mode: str | None = None, inbox: str | None = None):
    env = os.environ.copy()
    env.update({
        "NOVA_DB": str(tmp_path / "nova.db"),
        "NOVA_CHECKPOINT_DB": str(tmp_path / "checkpoints.db"),
        "NOVA_WORKSPACE": str(tmp_path / "ws"),
        "NOVA_CHROMA_DIR": str(tmp_path / "chroma"),
        "HARNESS_REPLY": reply,
        "HARNESS_LLM_DELAY": delay,
    })
    if max_ticks is not None:
        env["HARNESS_MAX_TICKS"] = max_ticks
    if mode is not None:
        env["HARNESS_MODE"] = mode
    if inbox is not None:
        env["HARNESS_INBOX"] = inbox
    proc = subprocess.Popen([sys.executable, str(HARNESS)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines: list[str] = []
    reader = threading.Thread(
        target=lambda: [lines.append(line) for line in proc.stdout], daemon=True)
    reader.start()
    return proc, lines


def _wait_for(lines: list[str], marker: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(marker in line for line in lines):
            return
        time.sleep(0.05)
    raise AssertionError(f"never saw {marker!r} in output:\n{''.join(lines)}")


def test_sigterm_mid_task_checkpoints_and_a_new_process_resumes(tmp_path):
    # --- process 1: the model call is slow and returns garbage (would retry).
    # SIGTERM lands while the executor's LLM call is in flight.
    proc, lines = _spawn(tmp_path, reply="never valid json", delay="2")
    _wait_for(lines, "LLM_CALL")
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 0            # clean exit, not a crash
    _wait_for(lines, "CLEAN_EXIT", timeout=5)
    assert sum("LLM_CALL" in l for l in lines) == 1  # stopped after the step in flight

    backlog = Backlog(tmp_path / "nova.db")
    try:
        stranded = backlog.tasks_with_status("in_progress")
        assert len(stranded) == 1                # mid-flight and resumable — not failed
        assert backlog.tasks_with_status("done", "failed") == []
    finally:
        backlog.close()

    # --- process 2: fresh process, same DBs; the model now answers correctly.
    # resume_incomplete must continue from the checkpoint (retry attempt 2),
    # and the same email must not be re-ingested.
    proc2, lines2 = _spawn(tmp_path, reply=VALID_EXECUTOR_REPLY, max_ticks="1")
    assert proc2.wait(timeout=30) == 0
    _wait_for(lines2, "CLEAN_EXIT", timeout=5)
    assert any("resuming task" in l for l in lines2)
    assert sum("LLM_CALL" in l for l in lines2) == 1   # resumed, not restarted from zero

    backlog = Backlog(tmp_path / "nova.db")
    try:
        assert len(backlog.tasks_with_status("done")) == 1
        assert backlog.tasks_with_status("in_progress", "pending") == []
        llm_calls = [e for e in backlog.log_entries() if e["action_type"] == "llm_call"]
        assert len(llm_calls) == 2               # 1 wasted attempt + 1 successful, audited
    finally:
        backlog.close()


def test_sigkill_mid_task_recovery_needs_only_the_checkpoint(tmp_path):
    """kill -9: unlike SIGTERM, NOTHING runs on the way down — no graceful
    stop, no final checkpoint write, no CLEAN_EXIT. Recovery must come purely
    from whatever checkpoint was already durable before the kill. This is
    the proof that the checkpoint itself is sufficient, not that graceful
    shutdown works."""
    proc, lines = _spawn(tmp_path, reply=VALID_EXECUTOR_REPLY, delay="3")
    _wait_for(lines, "LLM_CALL")
    proc.send_signal(signal.SIGKILL)
    assert proc.wait(timeout=15) == -signal.SIGKILL  # hard death
    assert not any("CLEAN_EXIT" in l for l in lines)  # nothing cleaned up

    backlog = Backlog(tmp_path / "nova.db")
    try:
        assert len(backlog.tasks_with_status("in_progress")) == 1
        # the in-flight call died before its audit entry was written
        assert [e for e in backlog.log_entries()
                if e["action_type"] == "llm_call"] == []
    finally:
        backlog.close()

    # fresh process: resume from the durable checkpoint alone
    proc2, lines2 = _spawn(tmp_path, reply=VALID_EXECUTOR_REPLY, max_ticks="1")
    assert proc2.wait(timeout=30) == 0
    _wait_for(lines2, "CLEAN_EXIT", timeout=5)
    assert any("resuming task" in l for l in lines2)

    backlog = Backlog(tmp_path / "nova.db")
    try:
        assert len(backlog.tasks_with_status("done")) == 1
        llm_calls = [e for e in backlog.log_entries() if e["action_type"] == "llm_call"]
        assert len(llm_calls) == 1                    # the killed attempt reran once
    finally:
        backlog.close()


def test_day_rollover_survives_a_real_process_restart(tmp_path):
    """The guarantee the rollover actually needs to provide: yesterday's
    spend sitting in a DURABLE checkpoint is reset by the next real request
    in a NEW process, and that reset itself survives an immediate kill.
    (Email tasks never touch the cloud, so the stranded task is a manual one
    whose next node is the Director — a real cloud-accounted request.)"""
    from datetime import datetime, timedelta, timezone

    from langgraph.checkpoint.sqlite import SqliteSaver

    from graph import _snapshot_state, build_graph
    from loop_harness import MANUAL_TASK_ID
    from nova_state import QuotaTracker

    checkpoint_db = str(tmp_path / "checkpoints.db")
    cfg = {"configurable": {"thread_id": MANUAL_TASK_ID}}

    # --- process 1: strand a manual task mid-flight (SIGTERM during the
    # Director's slow call), leaving a checkpoint that carries quota state.
    proc, lines = _spawn(tmp_path, reply="never valid json", delay="2",
                         mode="seed_manual")
    _wait_for(lines, "LLM_CALL")
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 0
    _wait_for(lines, "CLEAN_EXIT", timeout=5)

    # --- tamper with the checkpoint DB directly (no NOVA process running):
    # make the persisted quota look like heavy spend from YESTERDAY.
    stale = QuotaTracker(gemini_requests_today=200, groq_requests_today=900,
                         last_reset=datetime.now(timezone.utc) - timedelta(days=1))
    with SqliteSaver.from_conn_string(checkpoint_db) as saver:
        app = build_graph(checkpointer=saver)
        app.update_state(cfg, {"quota": stale}, as_node="director")
        assert _snapshot_state(app, cfg).quota.gemini_requests_today == 200

    # --- process 2: fresh process resumes the task; its very next request
    # must persist the rollover. Kill immediately after that one request.
    proc2, lines2 = _spawn(tmp_path, reply="never valid json", delay="2",
                           inbox="empty")
    _wait_for(lines2, "resuming task")
    _wait_for(lines2, "LLM_CALL")
    proc2.send_signal(signal.SIGTERM)
    assert proc2.wait(timeout=30) == 0
    _wait_for(lines2, "CLEAN_EXIT", timeout=5)
    assert sum("LLM_CALL" in l for l in lines2) == 1

    # --- reopen the DB cold: the reset was CHECKPOINTED, not just in memory.
    with SqliteSaver.from_conn_string(checkpoint_db) as saver:
        app = build_graph(checkpointer=saver)
        quota = _snapshot_state(app, cfg).quota
    assert quota.gemini_requests_today == 1        # 200 → reset → +1 (the one call)
    assert quota.groq_requests_today == 0          # 900 → reset, untouched since
    assert quota.last_reset.date() == datetime.now(timezone.utc).date()


def test_sigint_during_idle_wait_exits_promptly_and_cleanly(tmp_path):
    proc, lines = _spawn(tmp_path, reply=VALID_EXECUTOR_REPLY)  # no max_ticks: runs forever
    _wait_for(lines, "tick 1")                   # first tick done, loop is idling
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=15) == 0
    _wait_for(lines, "CLEAN_EXIT", timeout=5)
    assert any("tick 1: 1 unread fetched, 1 new task(s)" in l for l in lines)
