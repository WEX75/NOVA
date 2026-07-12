"""
NOVA — Permission tiers and enforcement (project guide §5–§7).

This is checked at the TOOL boundary inside the Executor, not merely at
planning time — so a mis-planned or prompt-injected request still cannot
execute. The check is a pure function of state; it never mutates anything.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from nova_state import AgentRole, NovaState


class PermissionTier(str, Enum):
    READ_ONLY = "read_only"
    FILE_WRITE = "file_write"
    OS_ACTION = "os_action"
    DESTRUCTIVE = "destructive"


# Every tool the Executor can invoke, mapped to its tier. A tool not listed
# here does not exist as far as enforcement is concerned (deny by default).
TOOL_TIERS: dict[str, PermissionTier] = {
    "read_file": PermissionTier.READ_ONLY,
    "list_dir": PermissionTier.READ_ONLY,
    "memory_recall": PermissionTier.READ_ONLY,
    "write_file": PermissionTier.FILE_WRITE,
    "git_commit": PermissionTier.FILE_WRITE,
    "launch_app": PermissionTier.OS_ACTION,
    "delete_file": PermissionTier.DESTRUCTIVE,
    "git_push": PermissionTier.DESTRUCTIVE,
    "send_email": PermissionTier.DESTRUCTIVE,
}


class PermissionDecision(BaseModel):
    allowed: bool
    tier: PermissionTier
    reason: str


def _critic_passed_current_subtask(state: NovaState) -> bool:
    """A Critic step exists for the current subtask AND its verdict was
    'pass'. A 'block' verdict counts as reviewed for the router's one-pass
    rule, but never satisfies the destructive-tier gate. Keyed on the same
    input_summary == description convention the router uses; critic_node
    writes 'verdict=<pass|block>; ...' as its output_summary."""
    if not state.subtasks:
        return False
    current = state.subtasks[state.current_subtask_index]
    return any(
        step.agent == AgentRole.CRITIC
        and step.input_summary == current.description
        and step.output_summary.startswith("verdict=pass")
        for step in state.history
    )


def known_artifacts(state: NovaState) -> set[str]:
    """Values that legitimately exist in NovaState: saved attachments /
    memory refs, and outputs of prior successful tool calls (e.g. a report
    the Executor just wrote). Raw email body text is never in this set —
    that is the project guide §7 provenance boundary."""
    artifacts: set[str] = set(state.memory_refs)
    for step in state.history:
        for tc in step.tool_calls:
            if tc.success and tc.output:
                artifacts.add(tc.output.strip())
    return artifacts


def arg_provenance_ok(state: NovaState, arg: str | None) -> bool:
    """os_action args (URL / file path) must already exist in NovaState."""
    if arg is None:
        return True  # launching an app with no target needs no provenance
    return arg.strip() in known_artifacts(state)


def check_permission(state: NovaState, tool_name: str, arg: str | None = None) -> PermissionDecision:
    tier = TOOL_TIERS.get(tool_name)
    if tier is None:
        return PermissionDecision(
            allowed=False, tier=PermissionTier.DESTRUCTIVE,
            reason=f"unknown tool '{tool_name}' — deny by default",
        )

    if tier in (PermissionTier.READ_ONLY, PermissionTier.FILE_WRITE):
        return PermissionDecision(allowed=True, tier=tier, reason="within autonomous tiers")

    # os_action and destructive both require explicit, current human approval.
    # `approved is True` only ever gets set by the human via the interrupt —
    # no node sets it (see human_gate). Framing of the request is irrelevant.
    if state.approved is not True:
        return PermissionDecision(
            allowed=False, tier=tier,
            reason=f"{tier.value} tool '{tool_name}' requires explicit human approval; approved={state.approved}",
        )

    if tier is PermissionTier.OS_ACTION and not arg_provenance_ok(state, arg):
        return PermissionDecision(
            allowed=False, tier=tier,
            reason=f"os_action arg {arg!r} has no provenance in NovaState "
                   "(not a saved attachment or generated artifact) — project guide §7",
        )

    if tier is PermissionTier.DESTRUCTIVE and not _critic_passed_current_subtask(state):
        return PermissionDecision(
            allowed=False, tier=tier,
            reason=f"destructive tool '{tool_name}' requires a Critic pass on the "
                   "current subtask before human confirmation counts — project guide §5",
        )

    return PermissionDecision(allowed=True, tier=tier, reason="approved by human"
                              + ("; critic reviewed" if tier is PermissionTier.DESTRUCTIVE else ""))
