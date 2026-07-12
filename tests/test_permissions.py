"""Permission-tier enforcement at the tool boundary (project guide §5/§7):
autonomous tiers pass, gated tiers require real human approval, os_action
args need provenance, destructive needs a Critic pass, and the allow-list
in os_actions is a hard wall."""

import subprocess

import os_actions
from conftest import critic_step, make_state, make_subtask
from nova_state import AgentRole, AgentStep, ModelProvider, ToolCallResult
from permissions import PermissionTier, check_permission, known_artifacts
from tools import ToolInput, run_tool


# --- autonomous tiers --------------------------------------------------------

def test_read_only_allowed_without_approval():
    d = check_permission(make_state(), "read_file")
    assert d.allowed and d.tier == PermissionTier.READ_ONLY


def test_file_write_allowed_without_approval():
    d = check_permission(make_state(), "write_file")
    assert d.allowed and d.tier == PermissionTier.FILE_WRITE


def test_unknown_tool_denied_even_with_approval():
    d = check_permission(make_state(approved=True), "format_disk")
    assert not d.allowed
    assert "deny by default" in d.reason


# --- os_action tier ----------------------------------------------------------

def test_os_action_denied_without_approval():
    for approved in (None, False):
        d = check_permission(make_state(approved=approved), "launch_app")
        assert not d.allowed
        assert "human approval" in d.reason


def test_os_action_allowed_when_approved_and_no_arg():
    d = check_permission(make_state(approved=True), "launch_app", arg=None)
    assert d.allowed


def test_os_action_arg_with_provenance_allowed():
    state = make_state(approved=True, memory_refs=["/ws/email/body.txt"])
    assert check_permission(state, "launch_app", arg="/ws/email/body.txt").allowed


def test_os_action_arg_without_provenance_denied():
    """The §7 corollary: a URL/path not already in NovaState never launches."""
    state = make_state(approved=True, memory_refs=["/ws/email/body.txt"])
    d = check_permission(state, "launch_app", arg="https://evil.example/pay")
    assert not d.allowed
    assert "provenance" in d.reason


def test_prior_tool_output_counts_as_provenance():
    step = AgentStep(agent=AgentRole.EXECUTOR, provider=ModelProvider.OLLAMA_LOCAL,
                     model_name="qwen2.5-coder:14b", input_summary="write report",
                     output_summary="ok",
                     tool_calls=[ToolCallResult(tool_name="write_file", input={},
                                                output="/ws/report.pdf", success=True)])
    state = make_state(approved=True, history=[step])
    assert "/ws/report.pdf" in known_artifacts(state)
    assert check_permission(state, "launch_app", arg="/ws/report.pdf").allowed


def test_failed_tool_output_is_not_provenance():
    step = AgentStep(agent=AgentRole.EXECUTOR, provider=ModelProvider.OLLAMA_LOCAL,
                     model_name="qwen2.5-coder:14b", input_summary="x", output_summary="x",
                     tool_calls=[ToolCallResult(tool_name="write_file", input={},
                                                output="/ws/half-written.pdf", success=False)])
    state = make_state(approved=True, history=[step])
    assert not check_permission(state, "launch_app", arg="/ws/half-written.pdf").allowed


# --- destructive tier --------------------------------------------------------

def _destructive_state(approved, with_critic=True, verdict="pass"):
    subtask = make_subtask("purge old exports")
    history = [critic_step("purge old exports", verdict=verdict)] if with_critic else []
    return make_state(approved=approved, subtasks=[subtask], history=history)


def test_destructive_denied_without_approval_even_with_critic_pass():
    assert not check_permission(_destructive_state(approved=None), "delete_file").allowed


def test_destructive_denied_without_critic_pass():
    d = check_permission(_destructive_state(approved=True, with_critic=False), "delete_file")
    assert not d.allowed
    assert "Critic pass" in d.reason


def test_destructive_denied_when_critic_blocked():
    d = check_permission(_destructive_state(approved=True, verdict="block"), "delete_file")
    assert not d.allowed


def test_destructive_allowed_with_critic_pass_and_approval():
    d = check_permission(_destructive_state(approved=True), "delete_file")
    assert d.allowed and d.tier == PermissionTier.DESTRUCTIVE


def test_send_email_is_destructive_tier():
    assert not check_permission(make_state(), "send_email").allowed
    assert not check_permission(make_state(approved=True), "send_email").allowed  # no critic pass


# --- run_tool: the single enforcement point ----------------------------------

def test_run_tool_denies_and_reports_permission_error():
    result = run_tool(make_state(), ToolInput(tool_name="git_push", reason="test"))
    assert not result.success
    assert "PERMISSION DENIED" in (result.error or "")


def test_run_tool_audit_logs_denied_and_allowed_calls(temp_backlog, temp_workspace):
    state = make_state(task_id="t-audit")
    run_tool(state, ToolInput(tool_name="git_push", reason="should be denied"))
    run_tool(state, ToolInput(tool_name="write_file", path="a.txt", content="hi",
                              reason="should be allowed"))
    entries = {e["action_type"] for e in temp_backlog.log_entries(task_id="t-audit")}
    assert {"tool:git_push", "tool:write_file"} <= entries


def test_path_traversal_escaping_workspace_fails(temp_workspace):
    result = run_tool(make_state(), ToolInput(tool_name="read_file",
                                              path="../../etc/passwd", reason="test"))
    assert not result.success
    assert "escapes the workspace" in (result.error or "")


# --- os_actions allow-list (hard wall below the permission check) ------------

def test_launch_app_refuses_unlisted_app():
    result = os_actions.launch_app("spotify", None, "t-test")
    assert not result.success
    assert "not in ALLOWED_APPS" in (result.error or "")


def test_forbidden_substring_blocks_even_allowlisted_entry(monkeypatch):
    """Even if someone edits ALLOWED_APPS to add a terminal, the hard check
    refuses to grant shell access."""
    monkeypatch.setitem(os_actions.ALLOWED_APPS, "totally_safe", ["gnome-terminal"])
    result = os_actions.launch_app("totally_safe", None, "t-test")
    assert not result.success
    assert "shell-granting" in (result.error or "")


def test_launch_app_never_uses_a_shell(monkeypatch):
    recorded = {}

    def fake_popen(cmd, **kwargs):
        recorded["cmd"], recorded["kwargs"] = cmd, kwargs

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    result = os_actions.launch_app("pdf_viewer", "/ws/report.pdf", "t-test")
    assert result.success
    assert recorded["cmd"] == ["xdg-open", "/ws/report.pdf"]  # argv list, arg not interpolated
    assert recorded["kwargs"].get("shell") is False
