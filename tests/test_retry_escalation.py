"""Retry -> mandatory human escalation: three failures
route to human_gate; the gate's decision fully controls what happens next."""

from langgraph.graph import END

from conftest import make_state, make_subtask
from nodes.common import failure_update
from nodes.executor import executor_node
from nodes.human_gate import human_gate_node
from nova_state import AgentRole, AgentStep, ModelProvider, NovaState, TaskStatus
from route_next_node import route_next_node


def merge(state: NovaState, update: dict) -> NovaState:
    """Mimic LangGraph reducer semantics for tests: operator.add lists
    append, everything else replaces."""
    data = dict(update)
    for key in ("history", "error_log"):
        if key in data:
            data[key] = getattr(state, key) + data[key]
    return state.model_copy(update=data)


def test_failure_update_bumps_retry_count():
    state = make_state(retry_count=1)
    step = AgentStep(agent=AgentRole.EXECUTOR, provider=ModelProvider.OLLAMA_LOCAL,
                     model_name="qwen2.5-coder:14b", input_summary="x", output_summary="boom")
    update = failure_update(state, step, "boom")
    assert update["retry_count"] == 2
    assert update["error_log"] == ["boom"]
    assert update["status"] == TaskStatus.IN_PROGRESS


def test_executor_failure_increments_retry_and_logs_error(fake_llm):
    fake_llm.reply = "this is not a json tool plan"
    state = make_state(status=TaskStatus.IN_PROGRESS,
                       subtasks=[make_subtask("write the report")])
    update = executor_node(state)
    assert update["retry_count"] == 1
    assert any("unparseable" in e for e in update["error_log"])
    assert len(update["history"]) == 1  # AgentStep logged even on failure (§4)


def test_three_failures_escalate_to_human_gate(fake_llm):
    """The full loop: fail, retry, fail, retry, fail -> human_gate. Never a
    fourth autonomous attempt."""
    fake_llm.reply = "garbage every time"
    state = make_state(status=TaskStatus.IN_PROGRESS,
                       subtasks=[make_subtask("write the report")])

    for expected_retry in (1, 2):
        state = merge(state, executor_node(state))
        assert state.retry_count == expected_retry
        assert route_next_node(state) == "executor"  # still retrying

    state = merge(state, executor_node(state))
    assert state.retry_count == 3
    assert route_next_node(state) == "human_gate"
    assert len(fake_llm.calls) == 3  # exactly max_retries attempts, no more


def test_escalation_beats_everything_but_terminal_status():
    state = make_state(retry_count=3, privacy_sensitive=True,
                       requires_human_approval=True, approved=True)
    assert route_next_node(state) == "human_gate"


def test_gate_approval_resets_retry_and_resumes():
    state = make_state(retry_count=3, approved=True,
                       error_log=["e1", "e2", "e3"],
                       subtasks=[make_subtask("write the report")])
    update = human_gate_node(state)
    assert update["retry_count"] == 0
    assert update["status"] == TaskStatus.IN_PROGRESS
    assert update["requires_human_approval"] is False
    resumed = merge(state, update)
    assert route_next_node(resumed) == "executor"  # back to work, not the gate


def test_gate_rejection_fails_the_task_terminally(temp_backlog):
    state = make_state(retry_count=3, approved=False)
    update = human_gate_node(state)
    assert update["status"] == TaskStatus.FAILED
    assert any("human rejected" in e for e in update["error_log"])
    assert route_next_node(merge(state, update)) == END


def test_gate_without_decision_stays_blocked():
    """Resuming with no decision must not unblock execution: the router
    sends the task straight back to the gate."""
    state = make_state(requires_human_approval=True, approved=None)
    update = human_gate_node(state)
    assert update["status"] == TaskStatus.BLOCKED
    blocked = merge(state, update)
    assert route_next_node(blocked) == "human_gate"


def test_gate_decision_is_audit_logged(temp_backlog):
    state = make_state(task_id="t-audit", retry_count=3, approved=True)
    human_gate_node(state)
    entries = temp_backlog.log_entries(task_id="t-audit")
    assert any(e["action_type"] == "human_gate" for e in entries)
