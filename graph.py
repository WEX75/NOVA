"""
NOVA — LangGraph assembly.

Every node's outgoing edge is the single conditional router
(route_next_node), and the graph is compiled with
interrupt_before=["human_gate"]: anything routed to the gate stops the
graph BEFORE the gate node runs. Approval is not a logged flag — it is the
owner resuming a checkpointed, halted graph.

Usage:
    app = build_graph()
    cfg = {"configurable": {"thread_id": task_id}}
    result = app.invoke(initial_state, cfg)      # may halt at human_gate
    if interrupted(app, cfg):
        print(pending_decision(app, cfg))
        resume_with_decision(app, cfg, approved=True)
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from nodes import (critic_node, director_node, executor_node,
                   human_gate_node, researcher_node, tutor_node)
from nova_state import NovaState
from route_next_node import route_next_node

_ROUTES = {
    "director": "director",
    "researcher": "researcher",
    "critic": "critic",
    "executor": "executor",
    "human_gate": "human_gate",
    "tutor": "tutor",
    END: END,
}


def build_graph(checkpointer=None):
    """Compile the NOVA graph. A checkpointer is mandatory for the interrupt
    to work; defaults to in-memory (swap in SqliteSaver for persistence
    across restarts).

    TODO(before next LangGraph upgrade): checkpoint deserialization warns
    "Deserializing unregistered type nova_state.* from checkpoint" — a future
    LangGraph will block this unless the nova_state types are registered via
    allowed_msgpack_modules (or LANGGRAPH_STRICT_MSGPACK is handled). Harmless
    on 1.2.9; must be addressed when bumping langgraph/langgraph-checkpoint."""
    workflow = StateGraph(NovaState)

    workflow.add_node("director", director_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("human_gate", human_gate_node)
    workflow.add_node("tutor", tutor_node)

    workflow.set_conditional_entry_point(route_next_node, _ROUTES)
    for node in ("director", "researcher", "critic", "executor", "human_gate",
                 "tutor"):
        workflow.add_conditional_edges(node, route_next_node, _ROUTES)

    return workflow.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=["human_gate"],  # project guide §6 — genuine halt
    )


def interrupted(app, config) -> bool:
    """True if the thread is paused at the human gate awaiting a decision."""
    snapshot = app.get_state(config)
    return bool(snapshot.next) and "human_gate" in snapshot.next


def _snapshot_state(app, config) -> NovaState:
    values = app.get_state(config).values
    return NovaState(**values) if isinstance(values, dict) else values


def pending_decision(app, config) -> str:
    """Human-readable description of what is waiting for approval."""
    from nodes.human_gate import _pending_summary
    return _pending_summary(_snapshot_state(app, config))


def resume_with_decision(app, config, approved: bool):
    """The ONLY path by which `approved` ever becomes True/False: an explicit
    call made by the owner.

    Applies the decision AS the human_gate node (standard LangGraph
    interrupt_before pattern): human_gate_node computes the consequences
    (history entry, retry reset on approve, FAILED on reject), the update is
    checkpointed as that node's output, and the graph resumes through the
    normal router. Without as_node="human_gate" the router — which keys on
    `approved is None` — would recompute edges and skip the gate's
    bookkeeping, leaving a rejection looping forever."""
    state = _snapshot_state(app, config).model_copy(update={"approved": approved})
    gate_update = human_gate_node(state)
    app.update_state(config, {"approved": approved, **gate_update}, as_node="human_gate")
    return app.invoke(None, config)
