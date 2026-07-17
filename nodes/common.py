"""
NOVA — Shared node plumbing (not orchestration logic).

Every node uses these three pieces so the project guide invariants hold
uniformly: an AgentStep logged per run (§4), the quota counter bumped after
every cloud call (§4/§8), and JSON coaxed safely out of model output.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from action_log import log_action
from llm import LLMResponse
from nova_state import (AgentRole, AgentStep, NovaState, QuotaTracker,
                        TaskStatus, ToolCallResult)


def make_step(agent: AgentRole, resp: LLMResponse, input_summary: str,
              output_summary: str, tool_calls: list[ToolCallResult] | None = None) -> AgentStep:
    return AgentStep(agent=agent, provider=resp.provider, model_name=resp.model_name,
                     input_summary=input_summary, output_summary=output_summary,
                     tool_calls=tool_calls or [], tokens_used=resp.tokens_used)


def bump_quota(state: NovaState, resp: LLMResponse) -> QuotaTracker:
    """Return the QuotaTracker for the node's partial update. Cloud calls go
    through QuotaTracker.record_request — the only writer of the UTC day
    rollover; local calls hand back an unchanged copy. Nodes never mutate
    state.quota."""
    if resp.cloud_call and resp.provider.value in ("gemini", "groq"):
        return state.quota.record_request(resp.provider.value)
    return state.quota.model_copy(deep=True)


def log_llm_call(state: NovaState, agent: AgentRole, resp: LLMResponse, why: str) -> None:
    log_action(state.task_id, actor=agent.value, action_type="llm_call",
               detail={"model": resp.model_name, "provider": resp.provider.value,
                       "tokens": resp.tokens_used, "cloud": resp.cloud_call},
               reason=why)


def extract_json(text: str) -> Optional[object]:
    """Parse the JSON object/array out of model output. A ```json fence wins
    outright; otherwise scan for complete top-level JSON values and return the
    LAST one — a model that disobeys "respond with ONLY JSON" typically
    restates the format as an example first and ends with the real answer.
    Nested values are never returned: a successful parse skips past the whole
    value before scanning resumes."""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    decoder = json.JSONDecoder()
    value: Optional[object] = None
    found = False
    pos = 0
    while (match := re.compile(r"[\[{]").search(candidate, pos)) is not None:
        try:
            value, pos = decoder.raw_decode(candidate, match.start())
            found = True
        except json.JSONDecodeError:
            pos = match.start() + 1
    return value if found else None


def failure_update(state: NovaState, step: AgentStep, error: str) -> dict:
    """Uniform 'this run failed' partial update: bump retry_count, log the
    error; the router escalates to human_gate at max_retries."""
    return {
        "history": [step],
        "error_log": [error],
        "retry_count": state.retry_count + 1,
        "status": TaskStatus.IN_PROGRESS,
    }


def transport_failure_update(state: NovaState, agent: AgentRole, exc: Exception,
                             input_summary: str, force_local: bool = False) -> dict:
    """INFRASTRUCTURE failure: the model call itself raised (network outage,
    provider unreachable) — nothing came back to parse. Routed through the
    same retry_count -> human_gate escalation as any task failure (§4), so a
    flapping network degrades to a human decision instead of crashing the
    loop. No quota is recorded: QuotaTracker cannot see a request that never
    returned, and counting phantom requests would starve the router for
    nothing."""
    from llm import resolve_model

    model, provider = resolve_model(agent, force_local)
    step = AgentStep(agent=agent, provider=provider, model_name=model,
                     input_summary=input_summary,
                     output_summary=f"TRANSPORT FAILURE: {exc}")
    log_action(state.task_id, actor=agent.value, action_type="llm_transport_failure",
               detail={"model": model, "provider": provider.value,
                       "error": f"{type(exc).__name__}: {exc}"},
               reason="model call raised before returning — infrastructure, not task, failure")
    return {**failure_update(state, step, f"{agent.value}: transport failure: {exc}"),
            "current_agent": agent}


ABORT_REASON = "Aborted by user via dashboard"


def aborted_update(state: NovaState, agent: AgentRole, input_summary: str,
                   force_local: bool = False) -> dict:
    """Owner-initiated cancellation (§14): TERMINAL by design. The owner
    already decided this task should stop — retrying would countermand
    them, so this deliberately does not ride failure_update's
    retry_count -> human_gate path the way transport failures do."""
    from llm import resolve_model

    model, provider = resolve_model(agent, force_local)
    step = AgentStep(agent=agent, provider=provider, model_name=model,
                     input_summary=input_summary,
                     output_summary="ABORTED: owner cancelled mid-flight")
    log_action(state.task_id, actor=agent.value, action_type="task_aborted",
               detail={"model": model, "provider": provider.value, "in_flight": True},
               reason="owner abort signal cancelled the in-flight call")
    return {"history": [step], "error_log": [ABORT_REASON],
            "status": TaskStatus.FAILED, "current_agent": agent}


def advance_update(state: NovaState, step: AgentStep, final_output: str | None = None) -> dict:
    """Move to the next subtask; finish the task when none remain."""
    next_index = state.current_subtask_index + 1
    done = next_index >= len(state.subtasks)
    update: dict = {
        "history": [step],
        "current_subtask_index": state.current_subtask_index if done else next_index,
        "status": TaskStatus.DONE if done else TaskStatus.IN_PROGRESS,
        "retry_count": 0,
    }
    if done:
        update["final_output"] = final_output or step.output_summary
    return update
