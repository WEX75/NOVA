"""
NOVA — manager entry point (single-user CLI).

Wires the tested pieces together; contains no business logic of its own:
  preflight  -> fail fast if Ollama is down (privacy vault is load-bearing);
                a missing cloud API key just disables that provider so the
                router falls back to the local Executor instead of the node
                crashing mid-task.
  poll       -> one-shot: fetch unread Outlook mail, quarantine-ingest each
                message (nodes.email_ingest — the §7 path the injection tests
                pin), then run each resulting task through the graph.
  run        -> one-shot: run one manually-entered goal through the graph.
  loop       -> persistent: poll on an interval with a durable (SQLite)
                checkpointer. SIGINT/SIGTERM shut down cleanly — the graph
                step in flight finishes, its checkpoint survives, and the
                next start resumes exactly there.

Approvals happen HERE: when the graph interrupts at human_gate the manager
prompts the owner on the console and resumes with their explicit decision.
No other code path ever sets `approved`.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import action_log
import secrets_store
from backlog import Backlog, TaskRecord
from graph import (_snapshot_state, build_graph, interrupted,
                   pending_decision, resume_with_decision)
from llm import assert_local_backend_ready
from nodes.email_ingest import ingest_email, ingest_unread
from nova_state import NovaState, QuotaTracker, TaskStatus

CLOUD_KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
CHECKPOINT_DB = Path(os.environ.get("NOVA_CHECKPOINT_DB",
                                    Path(__file__).parent / "checkpoints.db"))

# How long preflight keeps retrying an unreachable Ollama before giving up.
# Sized against systemd's StartLimitBurst=5 / StartLimitIntervalSec=300: with
# a 90s in-process window each service start survives a slow Ollama boot, and
# 5 starts x (90s + RestartSec) can never burn through the 300s budget the
# way five instant exits would.
PREFLIGHT_RETRY_WINDOW = float(os.environ.get("NOVA_PREFLIGHT_WINDOW", "90"))
PREFLIGHT_RETRY_DELAY = float(os.environ.get("NOVA_PREFLIGHT_DELAY", "3"))

log = logging.getLogger("nova")


def preflight(retry_window: float = PREFLIGHT_RETRY_WINDOW,
              retry_delay: float = PREFLIGHT_RETRY_DELAY) -> QuotaTracker:
    """Ollama must be up (hard requirement), but not necessarily FIRST: at
    boot, nova and Ollama race, so an unreachable backend is retried within
    a bounded window instead of exiting instantly and burning the systemd
    restart budget. Missing cloud keys degrade to local-only by zeroing that
    provider's daily LIMIT — not its counter, which the QuotaTracker
    day-rollover resets at UTC midnight. A keyless provider must stay
    disabled forever, not until the next calendar day."""
    deadline = time.monotonic() + retry_window
    while True:
        try:
            assert_local_backend_ready()
            break
        except RuntimeError as e:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            log.warning("preflight: %s — retrying for another %.0fs", e, remaining)
            time.sleep(min(retry_delay, remaining))
    quota = QuotaTracker()
    if not secrets_store.get_secret(CLOUD_KEYS["gemini"]):
        quota.gemini_daily_limit = 0
        print("[nova] no GEMINI_API_KEY in the keyring — Director/Researcher run locally",
              file=sys.stderr)
    if not secrets_store.get_secret(CLOUD_KEYS["groq"]):
        quota.groq_daily_limit = 0
        print("[nova] no GROQ_API_KEY in the keyring — Critic reviews unavailable; "
              "destructive-tier actions will stay blocked", file=sys.stderr)
    return quota


def ask_owner(summary: str) -> bool:
    """Explicit console confirmation — the only source of approved=True.
    A closed stdin is a rejection, never a silent pass."""
    try:
        answer = input(f"\n[NOVA needs approval] {summary}\nApprove? [y/N] ")
    except EOFError:
        log.warning("stdin closed while awaiting approval — treating as rejection")
        return False
    return answer.strip().lower() in ("y", "yes")


def _sanitize_notification(text: str, limit: int = 120) -> str:
    """Untrusted content reaches gate summaries (email subjects live in task
    goals), so the notification body is flattened to printable single-line
    text, truncated, then markup-escaped — several notification daemons
    render the body as Pango markup."""
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _dispatch_notification(title: str, body: str) -> None:
    """The actual notify-send call, isolated so tests can silence it."""
    try:
        subprocess.run(["notify-send", "--app-name=nova", title, body],
                       check=False, timeout=5, capture_output=True)
    except Exception as e:  # noqa: BLE001 — a failed alert must not affect the task
        log.warning("notify-send failed: %s", e)


def _send_notification(task_id: str, title: str, summary: str, reason: str) -> None:
    """Fire-and-forget desktop alert (§13). Deliberately NOT an os_action:
    a transient system notification, not an app launch — but still an
    external effect, so audit-logged (§4)."""
    body = _sanitize_notification(summary)
    _dispatch_notification(title, body)
    action_log.log_action(task_id, actor="system", action_type="notification",
                          detail={"title": title, "body": body}, reason=reason)


def notify_gate(task_id: str, summary: str) -> None:
    _send_notification(task_id, f"NOVA: approval needed ({task_id})", summary,
                       reason="task waiting at human gate (headless mode)")


def notify_result(task_id: str, status: str, summary: str) -> None:
    _send_notification(task_id, f"NOVA: task {status} ({task_id})",
                       summary or status,
                       reason=f"task reached terminal state ({status}) in headless mode")


def _stream_until_stop(app, payload, cfg, stop: threading.Event | None) -> None:
    """Advance the graph node by node. A stop request exits after the node in
    flight completes — LangGraph has already checkpointed that node's output,
    so the task resumes from exactly here on the next start."""
    for _ in app.stream(payload, cfg, stream_mode="values"):
        if stop is not None and stop.is_set():
            break


def drive_task(app, backlog: Backlog, task_id: str, quota: QuotaTracker,
               stop: threading.Event | None = None,
               fresh_state: NovaState | None = None,
               headless: bool = False) -> QuotaTracker:
    """Drive one task from wherever it stands (a fresh state, or its existing
    checkpoint when fresh_state is None) toward a terminal state, pausing for
    owner decisions at every interrupt. Honors `stop` between graph steps:
    on shutdown the task lands checkpointed and resumable, never half-dead.

    Interrupt handling has two modes (§13):
      console (headless=False)  — block on ask_owner(); EOF = rejection.
      headless (headless=True)  — apply a decision already recorded via
        `nova approve/reject` if one exists; otherwise park the task as
        needs_approval, fire a desktop notification, and return. No stdin
        dependency anywhere on this path.

    Returns the updated day-quota (shared across tasks; each task's state
    carries a copy seeded here)."""
    cfg = {"configurable": {"thread_id": task_id}}
    backlog.update_status(task_id, "in_progress")

    if fresh_state is not None:
        _stream_until_stop(app, fresh_state.model_copy(update={"quota": quota}), cfg, stop)
    elif not interrupted(app, cfg):
        # Mid-flight checkpoint from a previous process: continue it. (An
        # interrupted thread must NOT be streamed — that would run straight
        # through the gate without a decision; the loop below owns that.)
        _stream_until_stop(app, None, cfg, stop)

    while interrupted(app, cfg):
        if stop is not None and stop.is_set():
            backlog.update_status(task_id, "needs_approval")
            log.info("task %s parked at the human gate (shutdown requested)", task_id)
            break
        if headless:
            decision = backlog.unapplied_decision(task_id)
            if decision is None:
                backlog.update_status(task_id, "needs_approval")
                notify_gate(task_id, pending_decision(app, cfg))
                log.info("task %s waiting at the human gate — decide with: "
                         "nova approve %s | nova reject %s",
                         task_id, task_id, task_id)
                break
            resume_with_decision(app, cfg, approved=decision)
            backlog.mark_decision_applied(task_id)
            log.info("task %s: applied recorded owner decision (%s)",
                     task_id, "approved" if decision else "rejected")
            continue
        resume_with_decision(app, cfg, approved=ask_owner(pending_decision(app, cfg)))

    final = _snapshot_state(app, cfg)
    if final.status is TaskStatus.DONE:
        backlog.update_status(task_id, "done")
        if headless:  # console users are already watching; alert the absent owner
            notify_result(task_id, "done", final.final_output or "")
    elif final.status is TaskStatus.FAILED:
        backlog.update_status(task_id, "failed")
        if headless:
            notify_result(task_id, "failed",
                          final.error_log[-1] if final.error_log else "")
    # Any other status means a shutdown landed mid-flight: the backlog row
    # stays in_progress/needs_approval and resume_incomplete() picks it up.
    return final.quota


def run_task(app, backlog: Backlog, state: NovaState, quota: QuotaTracker) -> QuotaTracker:
    """One-shot commands: drive a brand-new task to a terminal state."""
    return drive_task(app, backlog, state.task_id, quota, fresh_state=state)


def resume_incomplete(app, backlog: Backlog, quota: QuotaTracker,
                      stop: threading.Event | None = None,
                      headless: bool = False) -> QuotaTracker:
    """Pick up tasks a previous process left mid-flight or parked at the gate.
    Only a checkpoint can be resumed safely — it carries the approval flags
    and artifact provenance that the tasks table deliberately does not — so a
    task without one is marked failed, loudly, for the owner to re-issue.
    In headless mode a still-undecided parked task re-notifies and stays
    parked; a recorded decision is applied immediately."""
    for task in backlog.tasks_with_status("in_progress", "needs_approval"):
        if stop is not None and stop.is_set():
            break
        cfg = {"configurable": {"thread_id": task.task_id}}
        if not app.get_state(cfg).values:
            log.warning("task %s has no checkpoint to resume from — marking failed. "
                        "Goal was: %s", task.task_id, task.goal[:120])
            backlog.update_status(task.task_id, "failed")
            continue
        log.info("resuming task %s from its checkpoint", task.task_id)
        quota = _drive_guarded(app, backlog, task.task_id, quota,
                               stop=stop, headless=headless)
    return quota


def apply_pending_decisions(app, backlog: Backlog, quota: QuotaTracker,
                            stop: threading.Event | None = None) -> QuotaTracker:
    """Apply owner decisions recorded by one-shot `nova approve/reject`
    processes since the last tick (§13). A decision for a task not yet at
    the gate stays pending (it may arrive there soon); a decision for a
    terminal or unknown task is discarded loudly."""
    for task_id, _approved in backlog.unapplied_decisions():
        if stop is not None and stop.is_set():
            break
        task = backlog.get_task(task_id)
        if task is None or task.status in ("done", "failed"):
            log.warning("decision for task %s discarded — %s", task_id,
                        "no such task" if task is None else f"task is already {task.status}")
            backlog.mark_decision_applied(task_id)
            continue
        cfg = {"configurable": {"thread_id": task_id}}
        if not app.get_state(cfg).values or not interrupted(app, cfg):
            continue  # not waiting at the gate (yet) — keep the decision pending
        quota = _drive_guarded(app, backlog, task_id, quota, stop=stop, headless=True)
    return quota


def _drive_guarded(app, backlog: Backlog, task_id: str, quota: QuotaTracker,
                   stop: threading.Event | None = None,
                   fresh_state: NovaState | None = None,
                   headless: bool = False) -> QuotaTracker:
    """Daemon-mode wrapper: a raising task must never take the loop down.
    Transport failures are already absorbed inside the nodes; anything that
    still raises here is a genuine bug — log it loudly, leave the task's
    checkpoint for the next start, keep serving other tasks."""
    try:
        return drive_task(app, backlog, task_id, quota,
                          stop=stop, fresh_state=fresh_state, headless=headless)
    except Exception:  # noqa: BLE001 — last resort, deliberately broad
        log.exception("task %s raised out of the graph — leaving its checkpoint "
                      "in place and moving on", task_id)
        return quota


def _install_signal_handlers(stop: threading.Event) -> dict:
    def handler(signum, frame):
        log.info("received %s — finishing the current step, then shutting down "
                 "(send again to force-quit)", signal.Signals(signum).name)
        stop.set()
        signal.signal(signum, signal.SIG_DFL)  # second signal terminates immediately

    previous: dict = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, handler)
    except ValueError:
        pass  # not the main thread (in-process tests): the stop Event still works
    return previous


class _NullInbox:
    """Stand-in when Outlook cannot be constructed: the loop keeps serving
    approvals/resumes, just with no mail to ingest."""

    def fetch_unread(self, limit: int = 10) -> list:
        return []


def _sleep_unless_stopped(stop: threading.Event, seconds: float) -> None:
    """Wait in small slices: a signal handler sets the event from this same
    thread, which one long Event.wait would not reliably notice."""
    deadline = time.monotonic() + seconds
    while not stop.is_set() and (remaining := deadline - time.monotonic()) > 0:
        stop.wait(min(0.2, remaining))


def loop(app, backlog: Backlog, quota: QuotaTracker, interval: float,
         limit: int = 10, client=None, stop: threading.Event | None = None,
         max_ticks: int | None = None,
         headless: bool | None = None) -> QuotaTracker:
    """Persistent mode. Each tick: apply any recorded approval decisions
    (§13), then poll the inbox, quarantine-ingest and run new mail one
    message at a time — a task is only created immediately before it starts
    running, so a shutdown can never strand an ingested-but-never-started
    task (its mail simply comes back next poll, un-deduped). The tick log
    line is the liveness/ops signal: unread seen, tasks created, gates
    waiting, quota standing.

    headless=None auto-detects: no tty on stdin (the systemd case) means
    approvals come from `nova approve/reject`, never from input()."""
    stop = stop or threading.Event()
    if headless is None:
        headless = not sys.stdin.isatty()
    previous_handlers = _install_signal_handlers(stop)
    try:
        if client is None:
            # An unusable Outlook client (e.g. NOVA_MS_CLIENT_ID not yet in
            # the keyring) must not kill the service: the loop still serves
            # approval decisions and checkpoint resumes without mail.
            try:
                from outlook import OutlookClient  # deferred: needs msal
                client = OutlookClient()
            except Exception as e:  # noqa: BLE001
                log.error("outlook client unavailable (%s) — running without "
                          "mail ingestion; approvals and resumes still served", e)
                client = _NullInbox()

        quota = resume_incomplete(app, backlog, quota, stop, headless=headless)

        tick = 0
        while not stop.is_set() and (max_ticks is None or tick < max_ticks):
            tick += 1
            quota = apply_pending_decisions(app, backlog, quota, stop)
            try:
                messages = client.fetch_unread(limit=limit)
            except Exception as e:  # noqa: BLE001 — one bad poll must not kill the loop
                log.error("tick %d: outlook fetch failed: %s", tick, e)
                messages = []

            new_tasks = 0
            for msg in messages:
                if stop.is_set():
                    break
                if backlog.email_already_ingested(msg.message_id):
                    continue
                state = ingest_email(msg, backlog)
                new_tasks += 1
                quota = _drive_guarded(app, backlog, state.task_id, quota,
                                       stop=stop, fresh_state=state,
                                       headless=headless)

            waiting = len(backlog.tasks_with_status("needs_approval"))
            log.info("tick %d: %d unread fetched, %d new task(s), "
                     "%d awaiting approval; quota gemini %d/%d, groq %d/%d",
                     tick, len(messages), new_tasks, waiting,
                     quota.gemini_requests_today, quota.gemini_daily_limit,
                     quota.groq_requests_today, quota.groq_daily_limit)
            backlog.write_heartbeat(tick, interval,
                                    quota.gemini_requests_today, quota.gemini_daily_limit,
                                    quota.groq_requests_today, quota.groq_daily_limit)
            _sleep_unless_stopped(stop, interval)

        log.info("loop stopped after %d tick(s) — in-flight work is checkpointed", tick)
        return quota
    finally:
        for sig, h in previous_handlers.items():
            signal.signal(sig, h)


def poll(app, backlog: Backlog, quota: QuotaTracker, limit: int = 10) -> QuotaTracker:
    from outlook import OutlookClient  # deferred: needs NOVA_MS_CLIENT_ID + msal

    states = ingest_unread(OutlookClient(), backlog, limit=limit)
    print(f"[nova] {len(states)} new email task(s)")
    for state in states:
        quota = run_task(app, backlog, state, quota)
    return quota


def run_manual_goal(app, backlog: Backlog, goal: str, quota: QuotaTracker) -> QuotaTracker:
    state = NovaState(task_id=f"task-{uuid.uuid4().hex[:12]}", goal=goal, source="manual")
    backlog.add_task(TaskRecord(task_id=state.task_id, goal=goal, source="manual"))
    return run_task(app, backlog, state, quota)


def record_owner_decision(backlog: Backlog, task_id: str, approved: bool) -> int:
    """`nova approve/reject <task_id>` — one-shot, no Ollama/preflight needed:
    writes the decision row; the running loop applies it on its next tick."""
    task = backlog.get_task(task_id)
    if task is None:
        print(f"[nova] no such task: {task_id}", file=sys.stderr)
        return 1
    if task.status in ("done", "failed"):
        print(f"[nova] task {task_id} is already {task.status} — nothing to decide",
              file=sys.stderr)
        return 1
    backlog.record_decision(task_id, approved)
    print(f"[nova] {'approval' if approved else 'rejection'} recorded for {task_id}; "
          "the loop applies it on its next tick")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nova")
    sub = parser.add_subparsers(dest="command", required=True)
    p_poll = sub.add_parser("poll", help="one-shot: ingest unread Outlook mail and run the tasks")
    p_poll.add_argument("--limit", type=int, default=10)
    p_run = sub.add_parser("run", help="one-shot: run one goal through the graph")
    p_run.add_argument("goal")
    p_loop = sub.add_parser("loop", help="persistent: poll the inbox on an interval")
    p_loop.add_argument("--interval", type=float, default=300.0,
                        help="seconds between polls (default 300)")
    p_loop.add_argument("--limit", type=int, default=10)
    p_approve = sub.add_parser("approve", help="approve a task waiting at the human gate")
    p_approve.add_argument("task_id")
    p_reject = sub.add_parser("reject", help="reject a task waiting at the human gate")
    p_reject.add_argument("task_id")
    p_dash = sub.add_parser("dashboard", help="local web dashboard (127.0.0.1 only)")
    p_dash.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="[nova %(asctime)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S%z")

    backlog = Backlog()
    action_log.configure(backlog)

    if args.command in ("approve", "reject"):
        # Deliberately BEFORE preflight: deciding must not require Ollama.
        return record_owner_decision(backlog, args.task_id,
                                     approved=(args.command == "approve"))

    if args.command == "dashboard":
        # Also before preflight: the dashboard reads the DB and records
        # decisions — it never talks to a model.
        import dashboard
        return dashboard.main(backlog.db_path, port=args.port)

    quota = preflight()

    if args.command == "loop":
        # Durable checkpoints: interrupted/parked tasks survive restarts.
        from langgraph.checkpoint.sqlite import SqliteSaver
        with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
            loop(build_graph(checkpointer=saver), backlog, quota,
                 interval=args.interval, limit=args.limit)
    elif args.command == "poll":
        poll(build_graph(), backlog, quota, limit=args.limit)
    else:
        run_manual_goal(build_graph(), backlog, args.goal, quota)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
