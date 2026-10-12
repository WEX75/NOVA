"""The first test that proves the pieces work TOGETHER: one manual goal
through the real compiled graph, end to end —

  Director (Gemini)   decomposes into three subtasks
  Researcher (Gemini) gathers figures
  Executor (local)    writes the report file
  Critic (Groq)       reviews the flagged subtask, exactly once
  human gate          interrupts; the owner approves on the console
  Executor (local)    launches the viewer on the report it just wrote
                      (provenance: a generated artifact, not free text)

Only litellm.completion, input() and subprocess.Popen are faked; routing,
quota bookkeeping, permission checks, checkpointing and audit logging are
all the production code paths.
"""

from __future__ import annotations

import json
import subprocess

import main as manager
from graph import _snapshot_state, build_graph
from nova_state import AgentRole, QuotaTracker, TaskStatus

GOAL = "Summarize Q2 numbers, write the report, and open it for review"

RESEARCH_FINDING = "Q2 revenue grew 4% versus Q1."


def _director_plan() -> str:
    return json.dumps([
        {"description": "gather Q2 revenue figures",
         "assigned_agent": "researcher", "requires_approval": False},
        {"description": "write the report file",
         "assigned_agent": "executor", "requires_approval": False},
        {"description": "open the finished report for the owner",
         "assigned_agent": "critic", "requires_approval": True},
    ])


def _write_plan() -> str:
    return json.dumps({
        "tool_calls": [{"tool_name": "write_file", "path": "report.md",
                        "content": f"# Q2 report\n\n{RESEARCH_FINDING}\n",
                        "reason": "persist the findings"}],
        "summary": "report written"})


def _launch_plan(report_path: str) -> str:
    return json.dumps({
        "tool_calls": [{"tool_name": "launch_app", "app_name": "pdf_viewer",
                        "arg": report_path, "reason": "open for owner review"}],
        "summary": "opened report for review"})


def test_manual_goal_runs_the_whole_pipeline(temp_backlog, temp_workspace,
                                             fake_llm, monkeypatch):
    report_path = str((temp_workspace / "report.md").resolve())

    def reply(kwargs):
        system = kwargs["messages"][0]["content"]
        if "Director" in system:
            return _director_plan()
        if "Researcher" in system:
            return RESEARCH_FINDING
        if "Critic" in system:
            return json.dumps({"verdict": "pass", "issues": []})
        # Executor — two different subtasks reach it. Match the Subtask line,
        # not the goal text (which repeats in every executor prompt).
        user = kwargs["messages"][-1]["content"]
        if "Subtask: write the report file" in user:
            return _write_plan()
        assert "Subtask: open the finished report" in user
        return _launch_plan(report_path)
    fake_llm.reply = reply

    prompts: list[str] = []

    def owner_console(prompt=""):
        prompts.append(prompt)
        return "y"
    monkeypatch.setattr("builtins.input", owner_console)

    launched: list[list[str]] = []
    monkeypatch.setattr(subprocess, "Popen",
                        lambda cmd, **kw: launched.append(cmd))

    app = build_graph()
    quota = manager.run_manual_goal(app, temp_backlog, GOAL, QuotaTracker())

    # --- terminal state, through the real router/checkpointer ---
    task = temp_backlog.tasks_with_status("done")[0]
    final = _snapshot_state(app, {"configurable": {"thread_id": task.task_id}})
    assert final.status is TaskStatus.DONE
    assert final.final_output == "opened report for review"
    assert final.error_log == []

    # --- every agent actually ran, in a sane order ---
    agents = [step.agent for step in final.history]
    assert agents.index(AgentRole.DIRECTOR) < agents.index(AgentRole.RESEARCHER)
    assert agents.index(AgentRole.RESEARCHER) < agents.index(AgentRole.CRITIC)
    assert agents.count(AgentRole.CRITIC) == 1          # exactly one review pass
    assert agents.count(AgentRole.EXECUTOR) == 2        # write, then launch
    assert any(s.model_name == "human_gate" and "APPROVED" in s.output_summary
               for s in final.history)

    # --- quota: the counters incremented through the REAL response paths ---
    assert quota.gemini_requests_today == 2             # director + researcher
    assert quota.groq_requests_today == 1               # single critic pass

    # --- side effects are real ---
    assert (temp_workspace / "report.md").read_text().rstrip().endswith(RESEARCH_FINDING)
    assert launched == [["xdg-open", report_path]]      # allow-listed argv, no shell
    assert len(prompts) == 1 and "open the finished report" in prompts[0]

    # --- and the audit trail can reconstruct all of it ---
    entries = temp_backlog.log_entries(task_id=task.task_id)
    kinds = [e["actor"] for e in entries if e["action_type"] == "llm_call"]
    assert kinds == ["director", "researcher", "executor", "critic", "executor"]
    types = {e["action_type"] for e in entries}
    assert {"task_created", "tool:write_file", "human_gate",
            "tool:launch_app", "app_launch"} <= types


def test_pipeline_stops_dead_on_owner_rejection(temp_backlog, temp_workspace,
                                                fake_llm, monkeypatch):
    """Same pipeline, owner says no at the gate: the flagged subtask never
    executes, nothing launches, the task fails terminally."""
    def reply(kwargs):
        system = kwargs["messages"][0]["content"]
        if "Director" in system:
            return _director_plan()
        if "Researcher" in system:
            return RESEARCH_FINDING
        if "Critic" in system:
            return json.dumps({"verdict": "pass", "issues": []})
        user = kwargs["messages"][-1]["content"]
        assert "Subtask: open the finished report" not in user, \
            "flagged subtask reached the executor despite rejection"
        return _write_plan()
    fake_llm.reply = reply

    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    launched: list[list[str]] = []
    monkeypatch.setattr(subprocess, "Popen",
                        lambda cmd, **kw: launched.append(cmd))

    app = build_graph()
    manager.run_manual_goal(app, temp_backlog, GOAL, QuotaTracker())

    task = temp_backlog.tasks_with_status("failed")[0]
    final = _snapshot_state(app, {"configurable": {"thread_id": task.task_id}})
    assert final.status is TaskStatus.FAILED
    assert launched == []
    assert any("human rejected" in e for e in final.error_log)
