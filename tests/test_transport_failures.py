"""Infrastructure failures (network outage, provider unreachable) route
through the same retry_count -> human_gate escalation as task failures —
never an unhandled exception out of the graph, never a phantom quota count.

This is the §4/§5 gap the hardening phase surfaced: llm.complete() was
documented as raising on transport errors with nodes owning the
bookkeeping, but no node caught anything.
"""

import json
import logging

import pytest

import main as manager
from conftest import make_state, make_subtask
from graph import _snapshot_state, build_graph, interrupted, pending_decision
from nodes.critic import critic_node
from nodes.director import director_node
from nova_state import AgentRole, QuotaTracker, TaskStatus
from outlook import EmailMessage
from test_outlook import StubInbox


def _network_down(kwargs):
    raise ConnectionError("connection refused: network is down")


# --- node level ----------------------------------------------------------------

def test_director_transport_failure_becomes_a_retry_not_a_crash(fake_llm,
                                                                temp_backlog):
    fake_llm.reply = _network_down
    state = make_state()

    update = director_node(state)                   # must not raise

    assert update["retry_count"] == 1
    assert any("transport failure" in e for e in update["error_log"])
    assert "quota" not in update                    # a call that never returned
    assert any(e["action_type"] == "llm_transport_failure"
               for e in temp_backlog.log_entries())


def test_critic_transport_failure_does_not_consume_its_single_review_pass(
        fake_llm, temp_backlog):
    """The router keys the one-pass rule on a Critic step whose input_summary
    equals the subtask description — a failed call must not match, or the
    subtask would silently lose its review."""
    fake_llm.reply = _network_down
    subtask = make_subtask("review the deletion", AgentRole.CRITIC)
    state = make_state(status=TaskStatus.IN_PROGRESS, subtasks=[subtask])

    update = critic_node(state)

    assert update["retry_count"] == 1
    assert "quota" not in update
    step = update["history"][0]
    assert step.input_summary != subtask.description  # pass NOT consumed


# --- graph level: escalation to the human gate ---------------------------------

def test_persistent_outage_escalates_to_the_gate_with_zero_quota_spent(fake_llm):
    fake_llm.reply = _network_down
    state = make_state(task_id="t-outage", goal="plan the quarter")
    app = build_graph()
    cfg = {"configurable": {"thread_id": state.task_id}}

    app.invoke(state, cfg)                          # must not raise

    assert interrupted(app, cfg)                    # genuine halt at the gate
    paused = _snapshot_state(app, cfg)
    assert paused.retry_count == 3                  # ceiling, then escalation
    assert "retry ceiling" in pending_decision(app, cfg)
    assert paused.quota.gemini_requests_today == 0  # phantom calls never counted
    assert paused.quota.groq_requests_today == 0


# --- loop level: the daemon keeps running --------------------------------------

def test_loop_survives_a_mid_task_outage_and_escalates(temp_backlog, fake_llm,
                                                       caplog, monkeypatch):
    """Ollama (or the network) dies while an email task runs: three transport
    failures, gate escalation, owner rejects, loop ticks on."""
    fake_llm.reply = _network_down
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    inbox = StubInbox([EmailMessage(
        message_id="m-outage", subject="Q2", sender="cfo@company.example",
        received="", body_preview="", body="Revenue grew 4% in Q2.")])

    with caplog.at_level(logging.INFO, logger="nova"):
        manager.loop(build_graph(), temp_backlog, QuotaTracker(),
                     interval=0, client=inbox, max_ticks=2,
                     headless=False)               # console mode: prompt + reject

    assert temp_backlog.get_task(
        temp_backlog.tasks_with_status("failed")[0].task_id).status == "failed"
    assert "tick 2" in caplog.text                  # the loop outlived the outage


def test_loop_survives_a_genuine_bug_in_a_task(temp_backlog, fake_llm, caplog,
                                               monkeypatch):
    """Last-resort guard: even a raise that is NOT a transport failure must
    not take the daemon down — log it, keep the checkpoint, keep ticking."""
    def explode(app, backlog, task_id, quota, stop=None, fresh_state=None):
        raise RuntimeError("unexpected bug in the graph")
    monkeypatch.setattr(manager, "drive_task", explode)
    inbox = StubInbox([EmailMessage(
        message_id="m-bug", subject="Q2", sender="cfo@company.example",
        received="", body_preview="", body="Revenue grew 4% in Q2.")])

    with caplog.at_level(logging.INFO, logger="nova"):
        manager.loop(build_graph(), temp_backlog, QuotaTracker(),
                     interval=0, client=inbox, max_ticks=2)

    assert "raised out of the graph" in caplog.text
    assert "tick 2" in caplog.text                  # still alive
