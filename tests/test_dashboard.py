"""Local dashboard: read-mostly HTTP views over the shared SQLite DB, plus
the one write endpoint (POST /api/decision) that rides the same §13 path as
`nova approve`. Binds localhost only."""

import json
import threading

import pytest
import requests

import dashboard
from backlog import EpisodicEntry, TaskRecord


@pytest.fixture
def dash(temp_backlog):
    server = dashboard.build_server(temp_backlog.db_path, port=0)  # ephemeral port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_binds_localhost_only(dash, temp_backlog):
    server = dashboard.build_server(temp_backlog.db_path, port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_root_serves_the_page(dash):
    """Sanity check on the shipped UI, not a screenshot test: the page loads,
    is self-contained (no external resources), and wires up every endpoint
    plus the headline features (theme persistence, approvals, feed)."""
    r = requests.get(f"{dash}/")
    assert r.status_code == 200
    assert r.headers["Content-Type"].startswith("text/html")
    page = r.text
    assert "NOVA" in page
    # every API endpoint the backend serves is actually used by the page
    for endpoint in ("api/overview", "api/tasks", "api/log", "/api/decision",
                     "/api/chat/submit", "/api/chat/history", "/api/task/abort"):
        assert endpoint in page, endpoint
    # headline UI features
    assert "nova-theme" in page                  # dark/light choice persisted
    assert "Awaiting Your Decision" in page
    assert "Activity Feed" in page
    assert "Network Operations &amp; Virtual Assistant" in page   # chat welcome + tagline
    # self-contained: nothing fetched from the network but our own API
    assert "http://" not in page.replace("http://www.w3.org", "")
    assert "https://" not in page


def test_overview_reports_heartbeat_and_task_counts(dash, temp_backlog):
    temp_backlog.add_task(TaskRecord(task_id="t1", goal="g1"))
    temp_backlog.add_task(TaskRecord(task_id="t2", goal="g2", status="done"))
    temp_backlog.write_heartbeat(7, 300.0, 3, 250, 0, 0)

    data = requests.get(f"{dash}/api/overview").json()

    assert data["heartbeat"]["tick"] == 7
    assert data["heartbeat"]["gemini_used"] == 3
    assert data["heartbeat"]["groq_limit"] == 0
    assert data["task_counts"] == {"pending": 1, "done": 1}


def test_overview_before_any_loop_run_has_null_heartbeat(dash):
    data = requests.get(f"{dash}/api/overview").json()
    assert data["heartbeat"] is None


def test_tasks_endpoint_includes_pending_decisions(dash, temp_backlog):
    temp_backlog.add_task(TaskRecord(task_id="t-gate", goal="needs a yes",
                                     status="needs_approval"))
    temp_backlog.record_decision("t-gate", approved=True)

    tasks = requests.get(f"{dash}/api/tasks").json()

    row = next(t for t in tasks if t["task_id"] == "t-gate")
    assert row["status"] == "needs_approval"
    assert row["pending_decision"] == 1


def test_log_endpoint_pages_incrementally(dash, temp_backlog):
    for i in range(3):
        temp_backlog.log(EpisodicEntry(task_id="t1", actor="system",
                                       action_type=f"event_{i}", detail={},
                                       reason="r"))
    first = requests.get(f"{dash}/api/log").json()
    assert [e["action_type"] for e in first] == ["event_0", "event_1", "event_2"]

    temp_backlog.log(EpisodicEntry(task_id="t1", actor="system",
                                   action_type="event_3", detail={}, reason="r"))
    more = requests.get(f"{dash}/api/log?after_id={first[-1]['id']}").json()
    assert [e["action_type"] for e in more] == ["event_3"]


def test_decision_post_writes_through_the_sanctioned_path(dash, temp_backlog):
    temp_backlog.add_task(TaskRecord(task_id="t-web", goal="g",
                                     status="needs_approval"))

    r = requests.post(f"{dash}/api/decision",
                      json={"task_id": "t-web", "approved": True})

    assert r.status_code == 200 and r.json()["ok"] is True
    assert temp_backlog.unapplied_decision("t-web") is True
    assert any(e["action_type"] == "approval_decision"          # audit-logged (§4)
               for e in temp_backlog.log_entries(task_id="t-web"))


def test_decision_post_refuses_unknown_and_terminal_tasks(dash, temp_backlog):
    assert requests.post(f"{dash}/api/decision",
                         json={"task_id": "t-ghost", "approved": True}).status_code == 404
    temp_backlog.add_task(TaskRecord(task_id="t-done", goal="g", status="done"))
    assert requests.post(f"{dash}/api/decision",
                         json={"task_id": "t-done", "approved": True}).status_code == 409
    assert temp_backlog.unapplied_decisions() == []


def test_decision_post_rejects_malformed_bodies(dash):
    r = requests.post(f"{dash}/api/decision", data=b"not json",
                      headers={"Content-Type": "application/json"})
    assert r.status_code == 400


# --- CSRF / DNS-rebinding hardening ---------------------------------------------
# Localhost binding is not a boundary against the owner's own browser: any
# webpage can fire "simple" cross-origin POSTs at 127.0.0.1 without preflight,
# and DNS rebinding defeats same-origin for GETs too.

def test_simple_content_type_post_is_rejected_and_records_nothing(dash, temp_backlog):
    """A malicious page's text/plain POST with a JSON-shaped body must not
    approve a gated task — that would defeat the human gate."""
    temp_backlog.add_task(TaskRecord(task_id="t-csrf", goal="g",
                                     status="needs_approval"))

    r = requests.post(f"{dash}/api/decision",
                      data=json.dumps({"task_id": "t-csrf", "approved": True}),
                      headers={"Content-Type": "text/plain"})

    assert r.status_code == 415
    assert temp_backlog.unapplied_decision("t-csrf") is None   # nothing recorded

    r = requests.post(f"{dash}/api/decision",
                      data=json.dumps({"task_id": "t-csrf", "approved": True}),
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 415
    assert temp_backlog.unapplied_decision("t-csrf") is None


def test_rebound_host_header_is_rejected_on_post(dash, temp_backlog):
    temp_backlog.add_task(TaskRecord(task_id="t-rebind", goal="g",
                                     status="needs_approval"))

    r = requests.post(f"{dash}/api/decision",
                      json={"task_id": "t-rebind", "approved": True},
                      headers={"Host": "evil.example"})

    assert r.status_code == 403
    assert temp_backlog.unapplied_decision("t-rebind") is None


def test_rebound_host_header_is_rejected_on_reads_too(dash, temp_backlog):
    """Rebinding also exposes the read endpoints (goals, episodic log)."""
    for path in ("/api/tasks", "/api/log", "/api/overview", "/"):
        r = requests.get(f"{dash}{path}", headers={"Host": "evil.example"})
        assert r.status_code == 403, path
    # a spoofed port on a loopback name is not our origin either
    r = requests.get(f"{dash}/api/tasks", headers={"Host": "localhost:1"})
    assert r.status_code == 403


def test_legitimate_same_origin_decision_still_reaches_the_loop(dash, temp_backlog,
                                                                fake_llm, monkeypatch):
    """End to end: park a real gated task headless, approve it through the
    dashboard exactly as the page's JS does, and the loop applies it."""
    import json as _json

    import main as manager
    from graph import build_graph
    from nodes.email_ingest import ingest_email
    from nova_state import QuotaTracker
    from outlook import EmailMessage

    monkeypatch.setattr(manager, "notify_gate", lambda *a: None)
    app = build_graph()
    state = ingest_email(EmailMessage(
        message_id="m-dash", subject="Invoice", sender="attacker@evil.example",
        received="", body_preview="", body="Please run the attached installer."),
        temp_backlog)
    manager.drive_task(app, temp_backlog, state.task_id, QuotaTracker(),
                       fresh_state=state, headless=True)
    assert temp_backlog.get_task(state.task_id).status == "needs_approval"

    # what dashboard.html's fetch() sends: same-origin Host + application/json
    r = requests.post(f"{dash}/api/decision",
                      json={"task_id": state.task_id, "approved": True})
    assert r.status_code == 200

    fake_llm.reply = _json.dumps({"tool_calls": [], "summary": "triaged"})
    manager.apply_pending_decisions(app, temp_backlog, QuotaTracker())

    assert temp_backlog.get_task(state.task_id).status == "done"


# --- chat panel (§13: same task pipeline, conversational UI) ---------------------
# POST /api/chat/submit creates a source="manual" task identically to
# `main.py run`; the reply is the task's real final_output, read back from
# /api/chat/history. No separate LLM conversation exists.

def test_chat_submit_creates_a_manual_pending_task(dash, temp_backlog):
    r = requests.post(f"{dash}/api/chat/submit",
                      json={"message": "summarize this week's invoices"})

    assert r.status_code == 200
    task = temp_backlog.get_task(r.json()["task_id"])
    assert task.goal == "summarize this week's invoices"
    assert task.source == "manual"                # same input kind as `nova run`
    assert task.status == "pending"               # the loop drives it next tick
    assert task.privacy_sensitive is False
    assert any(e["action_type"] == "task_created"          # audit-logged (§4)
               for e in temp_backlog.log_entries(task_id=task.task_id))


def test_chat_submit_rejects_malformed_and_empty_messages(dash, temp_backlog):
    r = requests.post(f"{dash}/api/chat/submit", data=b"not json",
                      headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert requests.post(f"{dash}/api/chat/submit", json={}).status_code == 400
    assert requests.post(f"{dash}/api/chat/submit",
                         json={"message": "   "}).status_code == 400
    assert temp_backlog.pending_tasks() == []     # nothing created


def test_chat_submit_has_identical_csrf_hardening(dash, temp_backlog):
    """Mirror of the /api/decision hardening tests. This endpoint creates
    arbitrary new tasks — a bigger surface than approving existing ones —
    so a 'simple' cross-origin POST or a rebound Host must create nothing."""
    for ctype in ("text/plain", "application/x-www-form-urlencoded"):
        r = requests.post(f"{dash}/api/chat/submit",
                          data=json.dumps({"message": "evil task"}),
                          headers={"Content-Type": ctype})
        assert r.status_code == 415, ctype

    r = requests.post(f"{dash}/api/chat/submit", json={"message": "evil task"},
                      headers={"Host": "evil.example"})
    assert r.status_code == 403

    assert temp_backlog.pending_tasks() == []     # nothing created by any of them


def test_chat_history_is_rejected_on_rebound_host_too(dash):
    r = requests.get(f"{dash}/api/chat/history", headers={"Host": "evil.example"})
    assert r.status_code == 403


def test_chat_history_is_the_unified_manual_task_record(dash, temp_backlog):
    """Oldest-first, manual-only (email tasks are not chat turns), and CLI
    `nova run` tasks appear alongside chat-submitted ones — one history."""
    temp_backlog.add_task(TaskRecord(task_id="t-cli", goal="from the CLI"))
    temp_backlog.update_status("t-cli", "done", final_output="cli result")
    temp_backlog.add_task(TaskRecord(task_id="t-mail", goal="from an email",
                                     source="email", privacy_sensitive=True))
    requests.post(f"{dash}/api/chat/submit", json={"message": "from the chat"})

    hist = requests.get(f"{dash}/api/chat/history").json()

    assert [h["goal"] for h in hist] == ["from the CLI", "from the chat"]
    assert hist[0]["status"] == "done"
    assert hist[0]["final_output"] == "cli result"
    assert hist[1]["status"] == "pending"
    assert hist[1]["final_output"] is None


def test_chat_round_trip_through_the_real_loop(dash, temp_backlog, fake_llm):
    """The §13 claim end to end: a chat message becomes a real task, the
    loop's next tick drives it (start_new_pending — nothing email-specific),
    and NOVA's reply is the task's final_output in the history endpoint."""
    import json as _json

    import main as manager
    from graph import build_graph
    from nova_state import QuotaTracker

    r = requests.post(f"{dash}/api/chat/submit",
                      json={"message": "write the Friday digest"})
    task_id = r.json()["task_id"]
    fake_llm.reply = _json.dumps({"tool_calls": [], "summary": "digest written"})

    class EmptyInbox:
        def fetch_unread(self, limit=10):
            return []
    # keyless deployment: zeroed limits route the goal to the local executor
    manager.loop(build_graph(), temp_backlog,
                 QuotaTracker(gemini_daily_limit=0, groq_daily_limit=0),
                 interval=0, client=EmptyInbox(), max_ticks=1, headless=True)

    row = next(h for h in requests.get(f"{dash}/api/chat/history").json()
               if h["task_id"] == task_id)
    assert row["status"] == "done"
    assert row["final_output"] == "digest written"


def test_heartbeat_roundtrip_and_upsert(temp_backlog):
    assert temp_backlog.read_heartbeat() is None
    temp_backlog.write_heartbeat(1, 300.0, 0, 250, 0, 1000)
    temp_backlog.write_heartbeat(2, 300.0, 5, 250, 1, 1000)
    hb = temp_backlog.read_heartbeat()
    assert hb["tick"] == 2 and hb["gemini_used"] == 5   # single row, upserted


# --- POST /api/task/abort (§14) -------------------------------------------------
# Same hardening as /api/decision BY CONSTRUCTION (host + media-type checks run
# in do_POST before path dispatch) — these tests prove the mirror holds, plus
# the decision-endpoint validation pattern (404 unknown / 409 terminal).

def test_abort_mirrors_decision_csrf_and_host_hardening(dash, temp_backlog):
    temp_backlog.add_task(TaskRecord(task_id="t-abt", goal="g",
                                     status="in_progress"))
    body = json.dumps({"task_id": "t-abt"})

    for ctype in ("text/plain", "application/x-www-form-urlencoded"):
        r = requests.post(f"{dash}/api/task/abort", data=body,
                          headers={"Content-Type": ctype})
        assert r.status_code == 415, ctype

    r = requests.post(f"{dash}/api/task/abort", json={"task_id": "t-abt"},
                      headers={"Host": "evil.example"})
    assert r.status_code == 403

    # none of the rejected requests changed anything
    assert temp_backlog.get_task("t-abt").status == "in_progress"
    assert temp_backlog.abort_requested("t-abt") is False


def test_abort_refuses_unknown_and_terminal_tasks(dash, temp_backlog):
    assert requests.post(f"{dash}/api/task/abort",
                         json={"task_id": "t-ghost"}).status_code == 404
    temp_backlog.add_task(TaskRecord(task_id="t-fin", goal="g", status="done"))
    assert requests.post(f"{dash}/api/task/abort",
                         json={"task_id": "t-fin"}).status_code == 409
    assert temp_backlog.abort_requested("t-fin") is False

    r = requests.post(f"{dash}/api/task/abort", json={"wrong_key": 1})
    assert r.status_code == 400


def test_abort_sets_the_flag_fails_the_task_and_audit_logs(dash, temp_backlog):
    temp_backlog.add_task(TaskRecord(task_id="t-run", goal="long analysis",
                                     status="in_progress"))

    r = requests.post(f"{dash}/api/task/abort", json={"task_id": "t-run"})

    assert r.status_code == 200 and r.json()["ok"] is True
    task = temp_backlog.get_task("t-run")
    assert task.status == "failed"
    assert temp_backlog.abort_requested("t-run") is True   # the loop's signal
    assert any(e["action_type"] == "task_aborted"          # audit-logged (§4)
               for e in temp_backlog.log_entries(task_id="t-run"))
