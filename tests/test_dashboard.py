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
    r = requests.get(f"{dash}/")
    assert r.status_code == 200
    assert "NOVA" in r.text and "api/overview" in r.text


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


def test_heartbeat_roundtrip_and_upsert(temp_backlog):
    assert temp_backlog.read_heartbeat() is None
    temp_backlog.write_heartbeat(1, 300.0, 0, 250, 0, 1000)
    temp_backlog.write_heartbeat(2, 300.0, 5, 250, 1, 1000)
    hb = temp_backlog.read_heartbeat()
    assert hb["tick"] == 2 and hb["gemini_used"] == 5   # single row, upserted
