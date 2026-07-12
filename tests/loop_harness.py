"""Real-process driver for the loop shutdown tests (run as a script, never
imported by pytest). Everything external is faked — LLM and inbox — but the
process, the signal delivery, the SQLite checkpoints and the backlog are
real. Configuration comes from the environment:

  NOVA_DB / NOVA_CHECKPOINT_DB / NOVA_WORKSPACE / NOVA_CHROMA_DIR  — tmp paths
  HARNESS_REPLY      — what the fake LLM answers (every call)
  HARNESS_LLM_DELAY  — seconds each fake LLM call takes (SIGTERM window)
  HARNESS_MAX_TICKS  — stop after N ticks (unset = run until signalled)
  HARNESS_MODE       — "loop" (default) or "seed_manual": create one manual
                       (non-privacy — its next node is the CLOUD Director)
                       task and drive it under signal control, no loop
  HARNESS_INBOX      — "empty" for a mail-less loop (resume-only scenarios)

Markers on stdout: "LLM_CALL" before each model call, "CLEAN_EXIT" last.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import litellm


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Usage:
    total_tokens = 5


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


def _fake_completion(**kwargs):
    print("LLM_CALL", flush=True)
    time.sleep(float(os.environ.get("HARNESS_LLM_DELAY", "0")))
    return _Response(os.environ["HARNESS_REPLY"])


litellm.completion = _fake_completion

import threading  # noqa: E402

import secrets_store  # noqa: E402


class _NoKeyring:
    """Subprocess tests must never touch the real Secret Service."""

    def get_password(self, service, name):
        return None

    def set_password(self, service, name, value):
        pass

    def delete_password(self, service, name):
        pass


secrets_store.configure(_NoKeyring())
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GROQ_API_KEY", None)

import action_log  # noqa: E402
import main as manager  # noqa: E402

# Subprocess tests must not fire real desktop notifications either.
manager._dispatch_notification = lambda title, body: print(f"NOTIFY {title}", flush=True)
from backlog import Backlog, TaskRecord  # noqa: E402
from graph import build_graph  # noqa: E402
from nova_state import NovaState, QuotaTracker  # noqa: E402
from outlook import EmailMessage  # noqa: E402

MANUAL_TASK_ID = "t-manual-seed"


class StubInbox:
    """The same benign message forever — dedup decides what's new."""

    def fetch_unread(self, limit=10):
        return [EmailMessage(message_id="m-signal", subject="Q2 numbers",
                             sender="cfo@company.example", received="",
                             body_preview="", body="Revenue grew 4% in Q2.")]


class EmptyInbox:
    def fetch_unread(self, limit=10):
        return []


def _seed_manual(app, backlog) -> None:
    """One fresh manual task driven under signal control: its first node is
    the cloud Director, so a SIGTERM here strands a task whose resumption
    makes a REAL cloud-accounted request."""
    stop = threading.Event()
    manager._install_signal_handlers(stop)
    state = NovaState(task_id=MANUAL_TASK_ID, goal="summarize the market outlook",
                      source="manual")
    backlog.add_task(TaskRecord(task_id=state.task_id, goal=state.goal))
    manager.drive_task(app, backlog, state.task_id, QuotaTracker(),
                       stop=stop, fresh_state=state)


def run() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="[nova] %(message)s")
    backlog = Backlog()  # honors NOVA_DB
    action_log.configure(backlog)

    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string(os.environ["NOVA_CHECKPOINT_DB"]) as saver:
        app = build_graph(checkpointer=saver)
        if os.environ.get("HARNESS_MODE") == "seed_manual":
            _seed_manual(app, backlog)
        else:
            inbox = (EmptyInbox() if os.environ.get("HARNESS_INBOX") == "empty"
                     else StubInbox())
            max_ticks = int(os.environ.get("HARNESS_MAX_TICKS", "0")) or None
            manager.loop(app, backlog, QuotaTracker(), interval=0.2, limit=5,
                         client=inbox, max_ticks=max_ticks)

    backlog.close()
    print("CLEAN_EXIT", flush=True)


if __name__ == "__main__":
    run()
