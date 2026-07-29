"""§15 tutor mode: a separate prose node, routed directly by source, local
always, read-only by construction, with the chat-context window that makes
quiz mode possible (investigation: every task otherwise starts stateless).
Also the num_ctx fix — Ollama's silent-truncation default (measured live:
a ~6000-token prompt capped to 2050 evaluated tokens, no error)."""

from __future__ import annotations

import llm
from backlog import TaskRecord
from conftest import make_state
from nodes.director import director_node
from nodes.executor import executor_node
from nodes.tutor import tutor_node
from nova_state import AgentRole, NovaState, TaskStatus
from route_next_node import route_next_node

PROSE = ("Good question. Before I sketch anything: what do you already know "
         "about hash collisions? Start by writing the bucket lookup yourself.")


# --- routing (§15: one conditional, same pattern as privacy_sensitive) ---------

def test_tutor_source_routes_straight_to_the_tutor_node():
    assert route_next_node(make_state(source="tutor")) == "tutor"


def test_tutor_routing_still_respects_terminal_retry_and_gate_checks():
    assert route_next_node(make_state(source="tutor",
                                      status=TaskStatus.DONE)) == "__end__"
    assert route_next_node(make_state(source="tutor",
                                      retry_count=3)) == "human_gate"
    assert route_next_node(make_state(source="tutor",
                                      requires_human_approval=True)) == "human_gate"


def test_non_tutor_sources_are_unaffected():
    assert route_next_node(make_state(source="manual",
                                      privacy_sensitive=True)) == "executor"


# --- the node --------------------------------------------------------------------

def test_tutor_node_returns_prose_done_and_local_only(fake_llm, temp_backlog):
    fake_llm.reply = PROSE
    update = tutor_node(make_state(source="tutor", goal="teach me hash maps"))

    assert update["status"] is TaskStatus.DONE
    assert update["final_output"] == PROSE          # prose, no JSON contract
    assert update["current_agent"] is AgentRole.TUTOR
    assert fake_llm.calls[-1]["model"].startswith("ollama/")   # force_local
    system = fake_llm.calls[-1]["messages"][0]
    assert system["role"] == "system"
    assert "You are NOVA (Network Operations & Virtual Assistant)" in system["content"]
    assert "TUTOR MODE" in system["content"]


def test_tutor_gets_the_recent_chat_window(fake_llm, temp_backlog):
    """Investigation (c): tasks start stateless, so the node must assemble
    prior manual/tutor turns itself — goal as user, final_output as
    assistant, oldest-first, current message last."""
    temp_backlog.add_task(TaskRecord(task_id="t-old", goal="What is a B-tree?",
                                     source="manual", status="done"))
    temp_backlog.update_status("t-old", "done",
                               final_output="A B-tree keeps keys sorted across wide nodes.")
    fake_llm.reply = PROSE

    tutor_node(make_state(task_id="t-now", source="tutor",
                          goal="How does that compare to a hash map?"))

    msgs = fake_llm.calls[-1]["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == "What is a B-tree?"
    assert "keys sorted across wide nodes" in msgs[2]["content"]
    assert msgs[-1]["content"].startswith("How does that compare")


def test_quiz_continuity_the_previous_question_is_in_context(fake_llm,
                                                             temp_backlog):
    """The reason the window exists: the question the tutor asked lives in
    the previous task's final_output — without it, grading is impossible."""
    temp_backlog.add_task(TaskRecord(task_id="t-quiz1", goal="quiz me on Big-O",
                                     source="tutor", status="done"))
    temp_backlog.update_status(
        "t-quiz1", "done",
        final_output="Question 1: what is the amortized cost of list.append?")
    fake_llm.reply = "Correct — amortized O(1). Question 2: …"

    tutor_node(make_state(task_id="t-quiz2", source="tutor", goal="O(1) amortized"))

    joined = " ".join(m["content"] for m in fake_llm.calls[-1]["messages"])
    assert "amortized cost of list.append" in joined   # the question survived


def test_failed_turns_are_excluded_from_the_window(fake_llm, temp_backlog):
    temp_backlog.add_task(TaskRecord(task_id="t-bad", goal="broken turn",
                                     source="manual", status="failed"))
    temp_backlog.update_status("t-bad", "failed", final_output="Transport failure")
    fake_llm.reply = PROSE

    tutor_node(make_state(task_id="t-now", source="tutor", goal="hello"))

    assert len(fake_llm.calls[-1]["messages"]) == 2    # system + current only


# --- read-only file prefetch -----------------------------------------------------

def test_workspace_files_named_in_the_goal_are_attached_and_audited(
        fake_llm, temp_backlog, temp_workspace):
    (temp_workspace / "algo.py").write_text(
        "def slow(xs):\n    return [x*2 for x in xs]\n")
    fake_llm.reply = PROSE

    update = tutor_node(make_state(source="tutor", goal="review algo.py please"))

    last = fake_llm.calls[-1]["messages"][-1]["content"]
    assert "FILE: algo.py" in last and "def slow(xs)" in last
    assert any(e["action_type"] == "tool:read_file"    # audit-logged (§4/§5)
               for e in temp_backlog.log_entries())
    assert update["history"][0].tool_calls[0].tool_name == "read_file"


def test_paths_outside_the_workspace_are_never_attached(fake_llm, temp_backlog,
                                                        tmp_path):
    secret = tmp_path / "secret.txt"                   # exists, but OUTSIDE
    secret.write_text("keyring contents")
    fake_llm.reply = PROSE

    tutor_node(make_state(source="tutor", goal=f"review {secret} please"))

    last = fake_llm.calls[-1]["messages"][-1]["content"]
    assert "keyring contents" not in last
    assert not any(e["action_type"] == "tool:read_file"
                   for e in temp_backlog.log_entries())


def test_tutor_empty_reply_is_a_retry_not_done(fake_llm, temp_backlog):
    fake_llm.reply = ""
    update = tutor_node(make_state(source="tutor", goal="teach me"))
    assert update["retry_count"] == 1
    assert update["status"] is TaskStatus.IN_PROGRESS


def test_file_attachment_accounts_for_header_overhead(fake_llm, temp_backlog,
                                                       temp_workspace, monkeypatch):
    """The minor accounting gap flagged in review: the '--- FILE: ... ---'
    header must count against FILE_CHAR_BUDGET too, not just the body."""
    import nodes.tutor as tutor_mod
    monkeypatch.setattr(tutor_mod, "FILE_CHAR_BUDGET", 200)
    (temp_workspace / "big.py").write_text("x = 1\n" * 200)   # 1200 chars, over budget
    fake_llm.reply = PROSE

    tutor_node(make_state(source="tutor", goal="review big.py please"))

    last = fake_llm.calls[-1]["messages"][-1]["content"]
    section = last[last.index("--- FILE: big.py"):]
    assert len(section) <= 200                          # header + body + note, all counted


# --- context budget (§15 follow-up, 2026-07-20): caps re-measured live -----------

def test_goal_itself_is_capped_with_a_visible_note(fake_llm, temp_backlog):
    """Item 1: state.goal was previously uncapped — a single large paste
    bypassed every other budget in this file."""
    from nodes.tutor import GOAL_CHAR_CAP
    huge = "x" * (GOAL_CHAR_CAP + 5000)
    fake_llm.reply = PROSE

    tutor_node(make_state(source="tutor", goal=huge))

    last = fake_llm.calls[-1]["messages"][-1]["content"]
    assert len(last) < len(huge)
    assert "truncated" in last and "ask to focus" in last


def test_graceful_degradation_drops_oldest_history_first_with_a_visible_note(
        fake_llm, temp_backlog, monkeypatch):
    """Item 2 decision: if the budget would still be exceeded, drop the
    OLDEST turns and say so — never repeat Ollama's own silent truncation.
    Forces the branch deterministically (two turns of known, equal size;
    a budget sized to fit exactly one) instead of needing gigantic content."""
    import nodes.tutor as tutor_mod

    for i, tag in enumerate(("old", "new")):
        tid = f"t-hist-{tag}"
        temp_backlog.add_task(TaskRecord(task_id=tid, goal=f"Q{tag} " + "Q" * 500,
                                         source="tutor", status="done"))
        temp_backlog.update_status(tid, "done", final_output=f"A{tag} " + "A" * 500)

    reserved = len(tutor_mod._SYSTEM) + len("hi")   # goal="hi", no attachment
    monkeypatch.setattr(tutor_mod, "SAFE_PROMPT_CHAR_BUDGET", reserved + 1500)
    fake_llm.reply = PROSE

    update = tutor_node(make_state(task_id="t-now", source="tutor", goal="hi"))

    msgs = fake_llm.calls[-1]["messages"]
    history_text = " ".join(m["content"] for m in msgs[1:-1])
    assert "Qold" not in history_text and "Aold" not in history_text   # oldest dropped
    assert "Qnew" in history_text                                     # newest kept
    assert "1 earlier turn(s)" in msgs[-1]["content"]                  # visible, not silent
    assert update["status"] is TaskStatus.DONE


# --- graph level -----------------------------------------------------------------

def test_tutor_task_runs_end_to_end_without_director_or_cloud(fake_llm,
                                                              temp_backlog):
    from graph import build_graph
    fake_llm.reply = PROSE
    app = build_graph()
    state = make_state(task_id="t-e2e-tutor", source="tutor",
                       goal="teach me dynamic programming")

    final = app.invoke(state, {"configurable": {"thread_id": state.task_id}})

    assert final["status"] == TaskStatus.DONE
    assert final["final_output"] == PROSE
    assert all(c["model"].startswith("ollama/") for c in fake_llm.calls)


# --- tutor timeout/max_tokens (§15 follow-up: the latency landmine the ------------
# token-budget fix exposed — a large-context first call after model load
# measured 89-95s, separate from the ~120s DEFAULT_TIMEOUT sized for the
# executor's short calls)

def test_tutor_uses_its_own_measured_timeout_and_bounded_max_tokens(
        fake_llm, temp_backlog):
    from nodes.tutor import TUTOR_MAX_TOKENS, TUTOR_TIMEOUT
    fake_llm.reply = PROSE

    tutor_node(make_state(source="tutor", goal="teach me"))

    call = fake_llm.calls[-1]
    assert call["timeout"] == TUTOR_TIMEOUT
    assert call["max_tokens"] == TUTOR_MAX_TOKENS
    assert TUTOR_TIMEOUT > llm.DEFAULT_TIMEOUT   # deliberately more generous


# --- num_ctx (§15: the silent-truncation landmine) -------------------------------

def test_local_calls_pass_num_ctx_cloud_calls_do_not(fake_llm, temp_backlog):
    import json as _json
    fake_llm.reply = _json.dumps({"tool_calls": [], "summary": "ok"})
    executor_node(make_state())
    assert fake_llm.calls[-1]["num_ctx"] == llm.NUM_CTX

    fake_llm.reply = _json.dumps({"subtasks": []})
    director_node(make_state())
    assert "num_ctx" not in fake_llm.calls[-1]         # provider-specific knob

    fake_llm.reply = PROSE
    tutor_node(make_state(source="tutor", goal="teach me"))
    assert fake_llm.calls[-1]["num_ctx"] == llm.NUM_CTX
