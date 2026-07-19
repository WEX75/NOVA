"""
NOVA — local dashboard (§13 companion).

A read-mostly web UI over the same SQLite DB the loop uses: live activity
feed (the episodic log), task states, loop heartbeat/quota, approve/reject
buttons that write through Backlog.record_decision — the same sanctioned
§13 path as `nova approve` — and a chat panel whose messages become real
manual tasks (POST /api/chat/submit, identical to `main.py run`) that the
running loop drives on its next tick. It never talks to the running loop
process and never calls a model.

SECURITY MODEL: binds 127.0.0.1 ONLY (hard-coded, not configurable), but
localhost binding alone is NOT a boundary against the owner's own browser:
any webpage can make the browser POST to 127.0.0.1 ("simple" cross-origin
requests skip CORS preflight), and DNS rebinding can point an attacker's
domain at 127.0.0.1. Two header checks close this:
  - every request must carry a Host of 127.0.0.1:<port>/localhost:<port>
    (rejects DNS rebinding, GETs included — the log/tasks are private);
  - POST requires Content-Type: application/json, which is never a
    "simple" type — forcing a CORS preflight this server never answers
    (no CORS headers, no OPTIONS handler, deliberately).
Reads use per-request read-only SQLite connections; each write endpoint
(POST /api/decision, /api/chat/submit, /api/task/abort) opens a fresh
Backlog per request, so no connection ever crosses threads.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from backlog import ABORT_REASON, Backlog, EpisodicEntry, TaskRecord

_PAGE = Path(__file__).parent / "dashboard.html"
BIND_HOST = "127.0.0.1"  # never expose beyond the local machine

TERMINAL = ("done", "failed")


class _Handler(BaseHTTPRequestHandler):
    server_version = "nova-dashboard"

    def log_message(self, fmt, *args):  # noqa: A003 — quiet: journald noise
        pass

    # --- plumbing ---

    def _host_ok(self) -> bool:
        """Only the loopback names for the port actually bound. Anything
        else is a DNS-rebinding attempt or a misdirected request."""
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in (f"127.0.0.1:{port}",
                                                f"localhost:{port}")

    def _ro(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.server.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _json(self, code: int, payload) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    # --- GET ---

    def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler API
        if not self._host_ok():
            self._json(403, {"error": "bad Host header"})
            return
        url = urlparse(self.path)
        params = parse_qs(url.query)
        if url.path == "/":
            raw = _PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif url.path == "/api/overview":
            self._json(200, self._overview())
        elif url.path == "/api/tasks":
            limit = int(params.get("limit", ["50"])[0])
            self._json(200, self._tasks(limit))
        elif url.path == "/api/log":
            after_id = int(params.get("after_id", ["0"])[0])
            limit = int(params.get("limit", ["100"])[0])
            self._json(200, self._log(after_id, limit))
        elif url.path == "/api/chat/history":
            limit = int(params.get("limit", ["200"])[0])
            self._json(200, self._chat_history(limit))
        else:
            self._json(404, {"error": "not found"})

    def _overview(self) -> dict:
        with self._ro() as conn:
            hb_row = conn.execute("SELECT * FROM runtime_status WHERE id=1").fetchone()
            counts = {r["status"]: r["n"] for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status")}
        heartbeat = dict(hb_row) if hb_row is not None else None
        return {"heartbeat": heartbeat, "task_counts": counts}

    def _tasks(self, limit: int) -> list[dict]:
        with self._ro() as conn:
            rows = conn.execute(
                "SELECT t.task_id, t.goal, t.source, t.privacy_sensitive, t.status, "
                "t.created_at, t.updated_at, "
                "(SELECT approved FROM approval_decisions d "
                " WHERE d.task_id = t.task_id AND d.applied_at IS NULL) AS pending_decision "
                "FROM tasks t ORDER BY t.updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def _log(self, after_id: int, limit: int) -> list[dict]:
        with self._ro() as conn:
            if after_id > 0:
                rows = conn.execute(
                    "SELECT * FROM episodic_log WHERE id > ? ORDER BY id ASC LIMIT ?",
                    (after_id, limit)).fetchall()
            else:  # initial load: the most recent entries, oldest-first
                rows = conn.execute(
                    "SELECT * FROM (SELECT * FROM episodic_log ORDER BY id DESC LIMIT ?) "
                    "ORDER BY id ASC", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def _chat_history(self, limit: int) -> list[dict]:
        """The chat transcript is real task data: manual- and tutor-source
        tasks oldest-first (goal = the owner's message, final_output =
        NOVA's reply). CLI `nova run` tasks are the same kind of input, so
        one unified history. The subquery keeps the most RECENT rows when
        the limit bites."""
        with self._ro() as conn:
            rows = conn.execute(
                "SELECT * FROM ("
                "  SELECT t.task_id, t.goal, t.status, t.source, t.created_at, "
                "         t.updated_at, t.final_output, "
                "         (SELECT approved FROM approval_decisions d "
                "          WHERE d.task_id = t.task_id AND d.applied_at IS NULL) "
                "         AS pending_decision "
                "  FROM tasks t WHERE t.source IN ('manual', 'tutor') "
                "  ORDER BY t.created_at DESC, t.task_id DESC LIMIT ?"
                ") ORDER BY created_at ASC, task_id ASC", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # --- POST ---
    # Both write endpoints share the SAME hardening by construction: the Host
    # check and the application/json requirement below run before any path
    # dispatch. /api/chat/submit is a bigger attack surface than /api/decision
    # (it creates arbitrary new tasks), so it must never get a weaker gate.

    def do_POST(self):  # noqa: N802
        if not self._host_ok():
            self._json(403, {"error": "bad Host header"})
            return
        media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            # A cross-origin "simple" POST (text/plain, form-encoded) never
            # carries this type; requiring it forces a CORS preflight that
            # this server never approves.
            self._json(415, {"error": "Content-Type must be application/json"})
            return
        path = urlparse(self.path).path
        if path == "/api/decision":
            self._post_decision()
        elif path == "/api/chat/submit":
            self._post_chat_submit()
        elif path == "/api/task/abort":
            self._post_task_abort()
        else:
            self._json(404, {"error": "not found"})

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def _post_decision(self) -> None:
        try:
            data = self._read_body()
            task_id = str(data["task_id"])
            approved = bool(data["approved"])
        except (KeyError, ValueError, json.JSONDecodeError):
            self._json(400, {"error": "expected JSON {task_id, approved}"})
            return

        backlog = Backlog(self.server.db_path)  # fresh rw connection per request
        try:
            task = backlog.get_task(task_id)
            if task is None:
                self._json(404, {"error": f"no such task: {task_id}"})
                return
            if task.status in TERMINAL:
                self._json(409, {"error": f"task is already {task.status}"})
                return
            backlog.record_decision(task_id, approved)
            self._json(200, {"ok": True, "task_id": task_id, "approved": approved})
        finally:
            backlog.close()

    def _post_task_abort(self) -> None:
        """Owner cancellation (§14). Sets the durable abort flag FIRST so the
        loop's in-flight model call (which polls it between stream chunks)
        sees it as early as possible, then fails the row and audit-logs. The
        in-flight cancellation really stops local generation — llm.complete()
        closes the connection and Ollama cancels server-side (verified; see
        llm._close_stream). Runs behind the same Host/media-type gate as
        every other POST, by construction (do_POST dispatches after both)."""
        try:
            data = self._read_body()
            task_id = str(data["task_id"])
        except (KeyError, ValueError, json.JSONDecodeError):
            self._json(400, {"error": "expected JSON {task_id}"})
            return

        backlog = Backlog(self.server.db_path)  # fresh rw connection per request
        try:
            task = backlog.get_task(task_id)
            if task is None:
                self._json(404, {"error": f"no such task: {task_id}"})
                return
            if task.status in TERMINAL:
                self._json(409, {"error": f"task is already {task.status}"})
                return
            backlog.request_abort(task_id)
            backlog.update_status(task_id, "failed", final_output=ABORT_REASON)
            backlog.log(EpisodicEntry(task_id=task_id, actor="human",
                                      action_type="task_aborted",
                                      detail={"via": "dashboard"},
                                      reason=ABORT_REASON))
            self._json(200, {"ok": True, "task_id": task_id})
        finally:
            backlog.close()

    def _post_chat_submit(self) -> None:
        """A chat message becomes a real manual task — the same defaults as
        `main.py run "<goal>"` (source='manual', status='pending', not
        privacy-sensitive). With {"tutor": true} the task is created with
        source='tutor' instead, which the router sends straight to the
        local tutor node (§15). The running loop drives it on its next tick
        (main.start_new_pending); this process never touches a model, so
        there is no path from here to an LLM that isn't an audited task."""
        try:
            data = self._read_body()
            message = str(data["message"]).strip()
            tutor = bool(data.get("tutor", False))
        except (KeyError, ValueError, json.JSONDecodeError):
            self._json(400, {"error": "expected JSON {message}"})
            return
        if not message:
            self._json(400, {"error": "message must not be empty"})
            return
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        backlog = Backlog(self.server.db_path)  # fresh rw connection per request
        try:
            backlog.add_task(TaskRecord(task_id=task_id, goal=message,
                                        source="tutor" if tutor else "manual"))
        finally:
            backlog.close()
        self._json(200, {"task_id": task_id})


def build_server(db_path, port: int = 8765) -> ThreadingHTTPServer:
    Backlog(db_path).close()  # ensure the schema exists before read-only opens
    server = ThreadingHTTPServer((BIND_HOST, port), _Handler)
    server.db_path = str(db_path)
    return server


def main(db_path, port: int = 8765) -> int:
    server = build_server(db_path, port)
    print(f"[nova] dashboard on http://{BIND_HOST}:{server.server_address[1]} "
          f"(db: {db_path}) — Ctrl-C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
