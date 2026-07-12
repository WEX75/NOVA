"""Shared fixtures: temp DB + workspace, litellm fake, state factory."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root imports

import pytest

import action_log
import secrets_store
import tools
from backlog import Backlog
from nova_state import (AgentRole, AgentStep, ModelProvider, NovaState,
                        QuotaTracker, Subtask, TaskStatus)


@pytest.fixture(autouse=True)
def temp_backlog(tmp_path):
    """Every test gets a fresh DB, and the audit log is wired to it."""
    bl = Backlog(tmp_path / "nova_test.db")
    action_log.configure(bl)
    yield bl
    bl.close()


@pytest.fixture(autouse=True)
def temp_workspace(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(tools, "WORKSPACE", ws)
    return ws


class FakeKeyring:
    """In-memory stand-in for the Secret Service backend."""

    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service, name):
        return self.store.get((service, name))

    def set_password(self, service, name, value):
        self.store[(service, name)] = value

    def delete_password(self, service, name):
        del self.store[(service, name)]


@pytest.fixture(autouse=True)
def temp_secrets(monkeypatch):
    """Tests never touch the user's real keyring, and never inherit real
    API keys from the developer's shell."""
    fake = FakeKeyring()
    monkeypatch.setattr(secrets_store, "_keyring", fake)
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return fake


@pytest.fixture(autouse=True)
def quiet_notifications(monkeypatch):
    """No test ever fires a real desktop notification; the recorded
    (title, body) pairs are returned for tests that assert on them."""
    import main as manager
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(manager, "_dispatch_notification",
                        lambda title, body: sent.append((title, body)))
    return sent


class _FakeUsage:
    total_tokens = 10


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class FakeLLMResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


@pytest.fixture
def fake_llm(monkeypatch):
    """Patch litellm.completion. Returns a controller: set .reply (str or
    callable(kwargs)->str); .calls records every request."""
    import litellm

    class Controller:
        reply: object = "ok"
        calls: list[dict] = []

    ctl = Controller()

    def fake_completion(**kwargs):
        ctl.calls.append(kwargs)
        content = ctl.reply(kwargs) if callable(ctl.reply) else ctl.reply
        return FakeLLMResponse(content)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    return ctl


def make_state(**overrides) -> NovaState:
    defaults = dict(task_id="t-test", goal="test goal")
    defaults.update(overrides)
    return NovaState(**defaults)


def make_subtask(description="do the thing", agent=AgentRole.EXECUTOR,
                 requires_approval=False) -> Subtask:
    return Subtask(description=description, assigned_agent=agent,
                   requires_approval=requires_approval)


def critic_step(description: str, verdict: str = "pass") -> AgentStep:
    """History entry matching what critic_node writes — used to satisfy (or
    fail) the destructive-tier critic-pass requirement in tests."""
    return AgentStep(agent=AgentRole.CRITIC, provider=ModelProvider.GROQ,
                     model_name="llama-3.3-70b-versatile",
                     input_summary=description,
                     output_summary=f"verdict={verdict}; issues=none")


__all__ = ["make_state", "make_subtask", "critic_step", "FakeLLMResponse"]
