"""
NOVA — Central Routing Function

This is registered as a LangGraph conditional edge, e.g.:

    workflow.add_conditional_edges(
        "director",
        route_next_node,
        {
            "director": "director",
            "researcher": "researcher",
            "critic": "critic",
            "executor": "executor",
            "human_gate": "human_gate",
            END: END,
        },
    )

IMPORTANT: This function is a pure reader — it inspects state and returns a
node name string. It does NOT mutate state. All state changes (incrementing
quota.gemini_requests_today, appending to error_log, bumping retry_count,
changing status) happen inside the node functions themselves (director_node,
critic_node, executor_node, human_gate_node), which return partial updates
that LangGraph merges via the reducers defined on NovaState. Mutating state
inside a routing function is a common mistake — LangGraph's checkpointing
model assumes nodes own state changes, not edges.
"""

from langgraph.graph import END
from nova_state import NovaState, AgentRole, TaskStatus


def route_next_node(state: NovaState) -> str:
    # 1. Terminal states — stop the graph.
    if state.status in (TaskStatus.DONE, TaskStatus.FAILED):
        return END

    # 2. Retry ceiling hit — stop autonomous looping, escalate to human
    #    rather than burning quota on a task that keeps failing.
    if state.retry_count >= state.max_retries:
        return "human_gate"

    # 3. Human-in-the-loop gate. OS actions and outbound emails/sends are
    #    flagged requires_human_approval=True by whichever node proposes them.
    #    approved is None while waiting; the human_gate node itself sets
    #    status=FAILED on rejection, which check #1 catches on the next pass.
    if state.requires_human_approval and state.approved is None:
        return "human_gate"

    # 4. Privacy-sensitive tasks never touch the cloud Director/Critic.
    #    Everything — planning included — runs on the local Executor.
    if state.privacy_sensitive:
        return "executor"

    # 5. Nothing decomposed yet — needs the Director. Fall back to local
    #    if Gemini's free-tier daily quota is nearly exhausted.
    if state.status == TaskStatus.PENDING or not state.subtasks:
        return "director" if state.quota.gemini_available() else "executor"

    # 6. In progress — route by whatever agent the Director assigned to the
    #    current subtask, respecting per-provider quota availability.
    current = state.subtasks[state.current_subtask_index]

    if current.assigned_agent == AgentRole.RESEARCHER:
        return "researcher" if state.quota.gemini_available() else "executor"

    if current.assigned_agent == AgentRole.CRITIC:
        # Critic is capped at exactly one pass per subtask — no debate loops.
        already_reviewed = any(
            step.agent == AgentRole.CRITIC and step.input_summary == current.description
            for step in state.history
        )
        if already_reviewed:
            return "executor"
        return "critic" if state.quota.groq_available() else "executor"

    # Default: code/file/git work always runs local regardless of quota.
    return "executor"
