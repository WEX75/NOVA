"""
NOVA — Executor tool implementations.

Single entry point: `run_tool()`. Every call — no matter which agent asked,
no matter how the request was phrased — goes through `check_permission()`
first and `log_action()` always. There is no side door: executor_node calls
nothing but this function, and os_actions.launch_app is invoked only from
here after the permission + provenance checks pass.

File operations are confined to the workspace directory (path-traversal
guarded); git runs with explicit argv lists, never shell strings.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

import os_actions
from action_log import log_action
from nova_state import NovaState, ToolCallResult
from permissions import check_permission

WORKSPACE = Path(os.environ.get("NOVA_WORKSPACE", Path(__file__).parent / "workspace"))


class ToolInput(BaseModel):
    tool_name: str
    path: Optional[str] = None      # file tools
    content: Optional[str] = None   # write_file
    app_name: Optional[str] = None  # launch_app
    arg: Optional[str] = None       # launch_app URL/file target, git commit msg
    reason: str = ""                # why this action is happening (goes to audit log)


def _safe_path(raw: str) -> Path:
    p = (WORKSPACE / raw).resolve()
    if not p.is_relative_to(WORKSPACE.resolve()):
        raise PermissionError(f"path {raw!r} escapes the workspace")
    return p


def _git(args: list[str]) -> str:
    out = subprocess.run(["git", "-C", str(WORKSPACE), *args],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"git {args[0]} failed")
    return out.stdout.strip()


def run_tool(state: NovaState, tool: ToolInput) -> ToolCallResult:
    decision = check_permission(state, tool.tool_name,
                                arg=tool.arg if tool.tool_name == "launch_app" else None)
    log_action(state.task_id, actor="executor", action_type=f"tool:{tool.tool_name}",
               detail={"input": tool.model_dump(), "tier": decision.tier.value,
                       "allowed": decision.allowed, "decision_reason": decision.reason},
               reason=tool.reason or f"subtask {state.current_subtask_index} of task {state.task_id}")

    if not decision.allowed:
        return ToolCallResult(tool_name=tool.tool_name, input=tool.model_dump(),
                              success=False, error=f"PERMISSION DENIED: {decision.reason}")

    try:
        output = _dispatch(state, tool)
        return ToolCallResult(tool_name=tool.tool_name, input=tool.model_dump(),
                              success=True, output=output)
    except Exception as e:  # noqa: BLE001 — every tool failure becomes a typed result
        return ToolCallResult(tool_name=tool.tool_name, input=tool.model_dump(),
                              success=False, error=str(e))


def _dispatch(state: NovaState, tool: ToolInput) -> str:
    name = tool.tool_name

    if name == "read_file":
        return _safe_path(tool.path or "").read_text()

    if name == "list_dir":
        target = _safe_path(tool.path or ".")
        return "\n".join(sorted(p.name for p in target.iterdir()))

    if name == "write_file":
        # content=None (missing/malformed — e.g. a hallucinated tool call)
        # must be a loud failure, never a silent `or ""` — that would
        # truncate a real, possibly valuable, existing file to empty with
        # no error at all. An explicit content="" (a deliberate empty
        # file) is still allowed; only non-string content is rejected.
        if not isinstance(tool.content, str):
            raise ValueError(
                f"write_file requires string content, got {tool.content!r} — "
                "refusing to write (null/missing content would silently "
                "truncate an existing file)")
        p = _safe_path(tool.path or "")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tool.content)
        return str(p)  # returned path becomes a known artifact (provenance)

    if name == "delete_file":
        p = _safe_path(tool.path or "")
        p.unlink()
        return f"deleted {p}"

    if name == "git_commit":
        _git(["add", "-A"])
        return _git(["commit", "-m", tool.arg or "nova: automated commit"])

    if name == "git_push":
        return _git(["push"])

    if name == "launch_app":
        result = os_actions.launch_app(tool.app_name or "", tool.arg, state.task_id)
        log_action(state.task_id, actor="executor", action_type="app_launch",
                   detail=result.model_dump(), reason=tool.reason or "approved os_action")
        if not result.success:
            raise RuntimeError(result.error or "launch failed")
        return f"launched {result.app_name}" + (f" with {tool.arg}" if tool.arg else "")

    if name == "memory_recall":
        from memory import recall
        return "\n".join(recall(tool.arg or state.goal))

    if name == "send_email":
        # Outbound send is destructive-tier and deliberately not implemented
        # until Mail.Send scope is approved by the owner.
        raise NotImplementedError("send_email requires Mail.Send scope — not yet enabled")

    raise ValueError(f"unhandled tool {name!r}")  # unreachable: permission check denies unknowns
