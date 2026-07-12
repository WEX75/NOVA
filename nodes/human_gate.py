"""
NOVA — Human gate node.

The graph is compiled with interrupt_before=["human_gate"], so execution
GENUINELY HALTS before this node runs. The owner inspects the paused state
and resumes with graph.update_state(config, {"approved": True/False},
as_node=...) — see graph.request_human_decision(). This node only runs
after that resume, and its whole job is to translate the decision into
state changes. No node anywhere sets approved=True; only the human does.

Two situations arrive here (route_next_node #2 and #3):
  - an action flagged requires_human_approval awaits approval
  - retry_count hit max_retries (mandatory escalation)
"""

from __future__ import annotations

from action_log import log_action
from nova_state import AgentRole, AgentStep, ModelProvider, NovaState, TaskStatus


def _gate_step(state: NovaState, decision: str) -> AgentStep:
    return AgentStep(agent=AgentRole.MANAGER, provider=ModelProvider.OLLAMA_LOCAL,
                     model_name="human_gate", input_summary=_pending_summary(state),
                     output_summary=decision)


def _pending_summary(state: NovaState) -> str:
    if state.retry_count >= state.max_retries:
        return (f"retry ceiling ({state.max_retries}) reached; "
                f"last error: {state.error_log[-1] if state.error_log else 'n/a'}")
    if state.subtasks:
        return f"approval requested: {state.subtasks[state.current_subtask_index].description}"
    return f"approval requested for task goal: {state.goal}"


def human_gate_node(state: NovaState) -> dict:
    summary = _pending_summary(state)
    log_action(state.task_id, actor="human", action_type="human_gate",
               detail={"approved": state.approved, "pending": summary,
                       "retry_count": state.retry_count},
               reason="human-in-the-loop decision point")

    if state.approved is True:
        # Keep approved=True: the Executor consumes it on the very next run
        # of the pending subtask, then resets it to None (single-use).
        return {"history": [_gate_step(state, "human APPROVED")],
                "requires_human_approval": False,
                "retry_count": 0,  # approval also clears a retry escalation
                "status": TaskStatus.IN_PROGRESS}

    if state.approved is False:
        return {"history": [_gate_step(state, "human REJECTED — task failed")],
                "requires_human_approval": False,
                "status": TaskStatus.FAILED,
                "error_log": [f"human rejected: {summary}"]}

    # Resumed without a decision — stay blocked; the router sends us straight
    # back here and the interrupt fires again. Nothing executes meanwhile.
    return {"history": [_gate_step(state, "no decision — still blocked")],
            "status": TaskStatus.BLOCKED}
