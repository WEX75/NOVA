"""tools.py dispatch-layer safety: write_file must never silently accept
non-string content. A hallucinated or malformed tool call with
content=null (observed live from qwen2.5-coder under an unrelated
long-form-prompt failure mode — project guide long-form investigation) must
be a loud failure, not a silent truncation of a real, possibly valuable,
existing file down to empty."""

from __future__ import annotations

from conftest import make_state
from tools import ToolInput, run_tool


def test_write_file_rejects_null_content_and_does_not_touch_the_file(
        temp_workspace):
    target = temp_workspace / "important_report.md"
    target.write_text("# Q2 Report\n\nRevenue grew 12%. Do not lose this.\n")

    result = run_tool(make_state(), ToolInput(
        tool_name="write_file", path="important_report.md", content=None,
        reason="hallucinated tool call"))

    assert result.success is False
    assert "content" in result.error.lower()
    assert target.read_text() == "# Q2 Report\n\nRevenue grew 12%. Do not lose this.\n"


def test_write_file_still_allows_a_deliberate_empty_string(temp_workspace):
    """content="" is a real, valid instruction (create/clear a file) —
    distinct from content=None (missing/malformed) — and must still work."""
    result = run_tool(make_state(), ToolInput(
        tool_name="write_file", path="blank.txt", content="",
        reason="deliberately empty file"))

    assert result.success is True
    assert (temp_workspace / "blank.txt").read_text() == ""


def test_write_file_normal_content_is_unaffected(temp_workspace):
    result = run_tool(make_state(), ToolInput(
        tool_name="write_file", path="notes.txt", content="hello world",
        reason="normal write"))

    assert result.success is True
    assert (temp_workspace / "notes.txt").read_text() == "hello world"
