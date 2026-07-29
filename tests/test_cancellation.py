"""§14: one cancellation mechanism in llm.complete() with two triggers —
the wall-clock timeout and the owner's abort signal — plus the executor's
handling of an abort (terminal, never a retry) and the NOVA identity that
now lives in the executor's own system prompt.

The GPU-level effect (Ollama really stops generating when the connection
closes) cannot be proven with fakes; that live proof ran against the real
server — see project guide §14 for the measured result.
"""

from __future__ import annotations

import json
import time

import litellm
import pytest

from backlog import TaskRecord
from conftest import make_state
from llm import ChatMessage, LLMAborted, LLMRequest, complete
from nodes.common import ABORT_REASON
from nodes.executor import executor_node
from nova_state import AgentRole, TaskStatus

_REQ = LLMRequest(agent=AgentRole.EXECUTOR, force_local=True, messages=[
    ChatMessage(role="user", content="count to a million")])


class _Closeable:
    closed = False

    def close(self):
        self.closed = True


def _fake_stream(monkeypatch, chunk_delay: float = 0.0):
    """Install a litellm.completion fake whose stream yields forever; returns
    the closeable sentinel so tests can prove the connection was shut."""
    sentinel = _Closeable()

    class _Delta:
        content = "x"

    class _Choice:
        delta = _Delta()

    class _Chunk:
        choices = [_Choice()]
        usage = None

    class _Inner:
        streaming_response = sentinel
        response_iterator = None

    class _Stream:
        completion_stream = _Inner()

        def __iter__(self):
            while True:
                if chunk_delay:
                    time.sleep(chunk_delay)
                yield _Chunk()

    monkeypatch.setattr(litellm, "completion", lambda **kw: _Stream())
    return sentinel


# --- llm-level: the shared mechanism -------------------------------------------

def test_abort_signal_cancels_mid_stream_and_really_closes(monkeypatch):
    sentinel = _fake_stream(monkeypatch)
    calls = {"n": 0}

    def abort_on_second_check() -> bool:
        calls["n"] += 1
        return calls["n"] > 1          # entry check passes; first chunk aborts

    with pytest.raises(LLMAborted):
        complete(_REQ, should_abort=abort_on_second_check)
    assert sentinel.closed             # the connection is what stops the GPU


def test_abort_before_the_call_never_opens_a_connection(monkeypatch):
    def boom(**kw):  # litellm must not be reached at all
        raise AssertionError("connection was opened despite a pending abort")
    monkeypatch.setattr(litellm, "completion", boom)

    with pytest.raises(LLMAborted):
        complete(_REQ, should_abort=lambda: True)


def test_deadline_raises_plain_timeout_for_the_transport_path(monkeypatch):
    """The SAME mechanism, other trigger: past the wall-clock deadline the
    raise is a plain TimeoutError — NOT LLMAborted — so it rides the existing
    llm_transport_failure route, and the stream is still really closed."""
    sentinel = _fake_stream(monkeypatch, chunk_delay=0.03)

    with pytest.raises(TimeoutError):
        complete(_REQ, timeout=0.01, should_abort=lambda: False)
    assert sentinel.closed


# --- executor-level: an abort is terminal, never a retry -----------------------

def test_executor_abort_is_terminal_failed_not_a_retry(fake_llm, temp_backlog):
    temp_backlog.add_task(TaskRecord(task_id="t-abort", goal="long analysis",
                                     status="in_progress"))
    temp_backlog.request_abort("t-abort")          # what POST /api/task/abort sets
    state = make_state(task_id="t-abort", goal="long analysis")

    update = executor_node(state)

    assert update["status"] is TaskStatus.FAILED   # terminal — router ends the graph
    assert "retry_count" not in update             # an owner decision is not retried
    assert update["error_log"] == [ABORT_REASON]
    entries = temp_backlog.log_entries(task_id="t-abort")
    assert any(e["action_type"] == "task_aborted" for e in entries)
    assert not any(e["action_type"] == "llm_transport_failure" for e in entries)


# --- identity hardening (§14) --------------------------------------------------

def test_executor_system_prompt_carries_the_nova_identity(fake_llm, temp_backlog):
    fake_llm.reply = json.dumps({"tool_calls": [], "summary": "ok"})

    executor_node(make_state())

    system = fake_llm.calls[-1]["messages"][0]
    assert system["role"] == "system"
    assert "You are NOVA (Network Operations & Virtual Assistant)" in system["content"]
    normalized = " ".join(system["content"].split())   # prompt line-wraps aren't load-bearing
    assert ("I'm NOVA, an autonomous assistant built to run entirely on "
           "this machine.") in normalized
