"""Regression tests for extract_json: it must return the FIRST JSON value
in the text. The original version tried the [...] span before {...}, which
mis-parsed the Executor's {"tool_calls": [...]} object as its inner array
and made every tool plan 'unparseable'."""

from nodes.common import extract_json


def test_object_containing_array_parses_as_the_object():
    plan = extract_json('{"tool_calls": [{"tool_name": "read_file"}], "summary": "ok"}')
    assert isinstance(plan, dict)
    assert plan["summary"] == "ok"


def test_bare_array_still_parses():
    assert extract_json('[{"description": "step 1"}]') == [{"description": "step 1"}]


def test_fenced_json_with_prose():
    text = 'Here is the plan:\n```json\n{"tool_calls": [], "summary": "done"}\n```\nHope that helps!'
    assert extract_json(text) == {"tool_calls": [], "summary": "done"}


def test_fenced_executor_reply_with_surrounding_chatter():
    """Realistic local-model shape: chatter, then the plan in a ```json fence."""
    text = ('Sure! I will read the file first. The plan follows the required '
            'format {"tool_calls": [...], "summary": str}:\n\n'
            '```json\n'
            '{"tool_calls": [{"tool_name": "read_file", "path": "notes.txt",\n'
            '                 "reason": "triage"}],\n'
            ' "summary": "reading the saved email body"}\n'
            '```')
    plan = extract_json(text)
    assert isinstance(plan, dict)
    assert plan["tool_calls"][0]["tool_name"] == "read_file"


def test_example_blob_before_the_real_plan():
    """Realistic disobedient shape: the model restates the format as an
    example JSON blob in prose, then gives the actual tool plan."""
    text = ('Each call must look like {"tool_name": "...", "path": "...", '
            '"reason": "..."} as specified. Here is my plan:\n'
            '{"tool_calls": [{"tool_name": "read_file", "path": "notes.txt", '
            '"reason": "triage"}], "summary": "reading notes"}')
    plan = extract_json(text)
    assert isinstance(plan, dict)
    assert plan["summary"] == "reading notes"
    assert plan["tool_calls"][0]["path"] == "notes.txt"


def test_array_embedded_in_prose():
    assert extract_json("The subtasks are: [1, 2, 3] as requested.") == [1, 2, 3]


def test_garbage_returns_none():
    assert extract_json("no json here at all") is None
