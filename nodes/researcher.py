"""
NOVA — Researcher node (Gemini 2.5 Flash, cloud).

Gathers information for the current subtask and records findings in history.
Read-only by construction: it has no access to tools.run_tool at all.
"""

from __future__ import annotations

from llm import ChatMessage, LLMRequest, complete
from nodes.common import (advance_update, bump_quota, failure_update,
                          log_llm_call, make_step, transport_failure_update)
from nova_state import AgentRole, NovaState

_SYSTEM = ("You are the Researcher of NOVA. Answer the research subtask "
           "concisely and factually. You cannot run code or take actions; "
           "produce findings as text only.")


def researcher_node(state: NovaState) -> dict:
    subtask = state.subtasks[state.current_subtask_index]

    try:
        resp = complete(LLMRequest(agent=AgentRole.RESEARCHER, messages=[
            ChatMessage(role="system", content=_SYSTEM),
            ChatMessage(role="user",
                        content=f"Overall goal: {state.goal}\n\nResearch subtask: {subtask.description}"),
        ]))
    except Exception as e:  # noqa: BLE001 — any raise means the call never completed
        return transport_failure_update(state, AgentRole.RESEARCHER, e,
                                        input_summary=subtask.description)
    log_llm_call(state, AgentRole.RESEARCHER, resp,
                 why=f"research subtask {state.current_subtask_index}: {subtask.description[:80]}")
    quota = bump_quota(state, resp)

    step = make_step(AgentRole.RESEARCHER, resp, input_summary=subtask.description,
                     output_summary=resp.content[:500])

    if not resp.content.strip():
        return {**failure_update(state, step, "researcher: empty response"),
                "quota": quota, "current_agent": AgentRole.RESEARCHER}

    return {**advance_update(state, step),
            "quota": quota, "current_agent": AgentRole.RESEARCHER}
