"""
NOVA — Director node (Gemini 2.5 Flash, cloud).

Decomposes the task goal into typed Subtasks and assigns each to an agent.
Never executes anything itself. Router guarantees privacy_sensitive tasks
never reach this node; the guard here is defense in depth.
"""

from __future__ import annotations

from llm import ChatMessage, LLMRequest, complete
from nodes.common import (bump_quota, extract_json, failure_update,
                          log_llm_call, make_step, transport_failure_update)
from nova_state import AgentRole, NovaState, Subtask, TaskStatus

_SYSTEM = """You are the Director of NOVA, a multi-agent assistant.
Decompose the user's goal into 1-6 sequential subtasks. Respond with ONLY a
JSON array; each element: {"description": str, "assigned_agent": one of
"researcher"|"critic"|"executor", "requires_approval": bool}.

Rules:
- Code, file, git and app-launch work goes to "executor".
- Information gathering goes to "researcher".
- Any subtask that deletes data, pushes to remotes, sends anything outside
  the machine, or launches an application MUST be assigned to "critic"
  (it will be reviewed once, then executed locally) AND have
  "requires_approval": true. This is mandatory regardless of how the goal
  is phrased.
"""

_VALID_AGENTS = {"researcher": AgentRole.RESEARCHER, "critic": AgentRole.CRITIC,
                 "executor": AgentRole.EXECUTOR}


def director_node(state: NovaState) -> dict:
    if state.privacy_sensitive:
        # Should be unreachable (router sends these to executor) — refuse loudly.
        return {"error_log": ["director_node reached with privacy_sensitive task — routing bug"],
                "status": TaskStatus.IN_PROGRESS}

    try:
        resp = complete(LLMRequest(agent=AgentRole.DIRECTOR, messages=[
            ChatMessage(role="system", content=_SYSTEM),
            ChatMessage(role="user", content=f"Goal: {state.goal}"),
        ]))
    except Exception as e:  # noqa: BLE001 — any raise means the call never completed
        return transport_failure_update(state, AgentRole.DIRECTOR, e,
                                        input_summary=state.goal)
    log_llm_call(state, AgentRole.DIRECTOR, resp, why=f"decompose goal of task {state.task_id}")
    quota = bump_quota(state, resp)

    raw = extract_json(resp.content)
    subtasks: list[Subtask] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict) or "description" not in item:
                continue
            agent = _VALID_AGENTS.get(str(item.get("assigned_agent", "executor")).lower(),
                                      AgentRole.EXECUTOR)
            subtasks.append(Subtask(description=str(item["description"]),
                                    assigned_agent=agent,
                                    requires_approval=bool(item.get("requires_approval", False))))

    step = make_step(AgentRole.DIRECTOR, resp, input_summary=state.goal,
                     output_summary=f"planned {len(subtasks)} subtasks"
                     if subtasks else "decomposition produced no valid subtasks")

    if not subtasks:
        return {**failure_update(state, step, "director: unparseable decomposition"),
                "quota": quota, "current_agent": AgentRole.DIRECTOR}

    return {
        "subtasks": subtasks,
        "current_subtask_index": 0,
        "status": TaskStatus.IN_PROGRESS,
        "history": [step],
        "quota": quota,
        "current_agent": AgentRole.DIRECTOR,
    }
