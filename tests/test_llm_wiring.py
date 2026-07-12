"""The unified LLM interface and the §4/§8 bookkeeping around it: correct
backend per role, api_base only for the local model, cloud_call flag
accuracy, and — through the REAL node code — the right QuotaTracker counter
bumped after every cloud call, success or failure."""

import json

import pytest

from conftest import make_state, make_subtask
from llm import (EMBED_MODEL, OLLAMA_BASE, ChatMessage, LLMRequest,
                 assert_local_backend_ready, complete, embed, resolve_model)
from nodes.critic import critic_node
from nodes.director import director_node
from nodes.executor import executor_node
from nodes.researcher import researcher_node
from nova_state import AgentRole, ModelProvider, TaskStatus


# --- llm.complete ------------------------------------------------------------

def _req(agent, **kw):
    return LLMRequest(agent=agent, messages=[ChatMessage(role="user", content="hi")], **kw)


def test_resolve_model_covers_every_role():
    for role in AgentRole:
        model, provider = resolve_model(role)
        assert model and isinstance(provider, ModelProvider)


def test_force_local_overrides_any_cloud_role():
    model, provider = resolve_model(AgentRole.DIRECTOR, force_local=True)
    assert provider is ModelProvider.OLLAMA_LOCAL
    assert model.startswith("ollama/")


def test_local_call_gets_api_base_and_is_not_cloud(fake_llm):
    resp = complete(_req(AgentRole.EXECUTOR))
    assert fake_llm.calls[-1]["api_base"] == OLLAMA_BASE
    assert resp.cloud_call is False


def test_cloud_call_gets_no_api_base_and_is_flagged(fake_llm):
    for agent, provider in ((AgentRole.DIRECTOR, ModelProvider.GEMINI),
                            (AgentRole.CRITIC, ModelProvider.GROQ)):
        resp = complete(_req(agent))
        assert "api_base" not in fake_llm.calls[-1]
        assert resp.cloud_call is True and resp.provider is provider


def test_cloud_api_keys_come_from_the_secret_store(fake_llm):
    """litellm never reads os.environ for a key: llm.complete passes the
    SecretStore value explicitly (or nothing, if the keyring has none)."""
    import secrets_store

    complete(_req(AgentRole.DIRECTOR))
    assert "api_key" not in fake_llm.calls[-1]        # nothing in the keyring

    secrets_store.set_secret("GEMINI_API_KEY", "gem-key")
    secrets_store.set_secret("GROQ_API_KEY", "groq-key")
    complete(_req(AgentRole.DIRECTOR))
    assert fake_llm.calls[-1]["api_key"] == "gem-key"
    complete(_req(AgentRole.CRITIC))
    assert fake_llm.calls[-1]["api_key"] == "groq-key"
    complete(_req(AgentRole.EXECUTOR))
    assert "api_key" not in fake_llm.calls[-1]        # local never needs one


def test_max_tokens_forwarded_only_when_set(fake_llm):
    complete(_req(AgentRole.EXECUTOR))
    assert "max_tokens" not in fake_llm.calls[-1]
    complete(_req(AgentRole.EXECUTOR, max_tokens=64))
    assert fake_llm.calls[-1]["max_tokens"] == 64


# --- llm.embed (long-term memory embeddings, always local) --------------------

def test_embed_uses_the_local_ollama_backend(monkeypatch):
    import litellm

    recorded = {}

    class FakeEmbeddingResponse:
        data = [{"embedding": [0.3, 0.4], "index": 1},
                {"embedding": [0.1, 0.2], "index": 0}]

    def fake_embedding(**kwargs):
        recorded.update(kwargs)
        return FakeEmbeddingResponse()

    monkeypatch.setattr(litellm, "embedding", fake_embedding)
    vectors = embed(["first", "second"])

    assert recorded["model"] == EMBED_MODEL
    assert recorded["model"].startswith("ollama/")     # never a cloud embedder
    assert recorded["api_base"] == OLLAMA_BASE
    assert recorded["input"] == ["first", "second"]
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]         # index order, not arrival order


# --- startup readiness (server up AND embedding model pulled) ------------------

def _fake_ollama(monkeypatch, models: list[str]):
    class R:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"models": [{"name": m} for m in models]}

    monkeypatch.setattr("requests.get", lambda url, timeout=0: R())


def test_readiness_passes_when_embed_model_is_pulled(monkeypatch):
    _fake_ollama(monkeypatch, ["qwen2.5-coder:14b", "nomic-embed-text:latest"])
    assert_local_backend_ready()  # must not raise


def test_readiness_fails_with_pull_instruction_when_embed_model_missing(monkeypatch):
    _fake_ollama(monkeypatch, ["qwen2.5-coder:14b"])
    with pytest.raises(RuntimeError, match="ollama pull nomic-embed-text"):
        assert_local_backend_ready()


# --- per-node quota bumps through the real node code ---------------------------

def test_director_bumps_gemini_counter(fake_llm):
    fake_llm.reply = json.dumps([{"description": "step", "assigned_agent": "executor"}])
    update = director_node(make_state())
    assert update["quota"].gemini_requests_today == 1
    assert update["quota"].groq_requests_today == 0


def test_director_counts_the_call_even_when_output_is_garbage(fake_llm):
    """§4: increment after EVERY cloud call — a wasted request still spent
    free-tier quota."""
    fake_llm.reply = "no json at all"
    update = director_node(make_state())
    assert update["quota"].gemini_requests_today == 1
    assert update["retry_count"] == 1  # and the failure is still a failure


def test_researcher_bumps_gemini_counter(fake_llm):
    fake_llm.reply = "findings: the market grew"
    state = make_state(status=TaskStatus.IN_PROGRESS,
                       subtasks=[make_subtask("find info", AgentRole.RESEARCHER)])
    update = researcher_node(state)
    assert update["quota"].gemini_requests_today == 1
    assert update["quota"].groq_requests_today == 0


def test_critic_bumps_groq_counter(fake_llm):
    fake_llm.reply = json.dumps({"verdict": "pass", "issues": []})
    state = make_state(status=TaskStatus.IN_PROGRESS,
                       subtasks=[make_subtask("review this", AgentRole.CRITIC)])
    update = critic_node(state)
    assert update["quota"].groq_requests_today == 1
    assert update["quota"].gemini_requests_today == 0


def test_executor_local_call_bumps_nothing(fake_llm):
    fake_llm.reply = json.dumps({"tool_calls": [], "summary": "done"})
    state = make_state(status=TaskStatus.IN_PROGRESS, subtasks=[make_subtask()])
    update = executor_node(state)
    assert update["quota"].gemini_requests_today == 0
    assert update["quota"].groq_requests_today == 0


def test_counters_accumulate_across_calls(fake_llm):
    fake_llm.reply = json.dumps([{"description": "s", "assigned_agent": "executor"}])
    state = make_state()
    state.quota.gemini_requests_today = 5
    update = director_node(state)
    assert update["quota"].gemini_requests_today == 6
