"""
NOVA — Unified LLM interface (LiteLLM).

Every node calls `complete()` regardless of backend. This is the ONLY module
that knows provider strings; nodes know AgentRole, nothing else.

Local backend note (project guide §3): Ollama on this machine (Radeon 780M /
gfx1103) must run with OLLAMA_VULKAN=1 — gfx1103 is not ROCm-supported and
without Vulkan it silently falls back to CPU. That env var belongs on the
`ollama serve` process, not this client, but `assert_local_backend_ready()`
checks for the common misconfiguration at startup.
"""

from __future__ import annotations

import os
from typing import Optional

import litellm
from pydantic import BaseModel, Field

import secrets_store
from nova_state import AgentRole, ModelProvider

OLLAMA_BASE = os.environ.get("NOVA_OLLAMA_BASE", "http://localhost:11434")

# Long-term memory embeddings (project guide §9). Local by design: memory content
# is business updates/preferences — privacy-sensitive by default — so it never
# leaves the machine, and it rides the Vulkan GPU path Ollama already provides
# (§3). One-time setup: `ollama pull nomic-embed-text`.
EMBED_MODEL = "ollama/nomic-embed-text"

# provider -> SecretStore name. Passed to litellm explicitly: litellm would
# otherwise read os.environ itself, which is exactly what the SecretStore
# migration removes.
PROVIDER_KEYS: dict[ModelProvider, str] = {
    ModelProvider.GEMINI: "GEMINI_API_KEY",
    ModelProvider.GROQ: "GROQ_API_KEY",
}

# role -> (litellm model string, provider tag)
MODEL_MAP: dict[AgentRole, tuple[str, ModelProvider]] = {
    AgentRole.DIRECTOR: ("gemini/gemini-2.5-flash", ModelProvider.GEMINI),
    AgentRole.RESEARCHER: ("gemini/gemini-2.5-flash", ModelProvider.GEMINI),
    AgentRole.CRITIC: ("groq/llama-3.3-70b-versatile", ModelProvider.GROQ),
    AgentRole.EXECUTOR: ("ollama/qwen2.5-coder:14b", ModelProvider.OLLAMA_LOCAL),
    AgentRole.MANAGER: ("ollama/qwen2.5-coder:14b", ModelProvider.OLLAMA_LOCAL),
}


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMRequest(BaseModel):
    agent: AgentRole
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    # Force the local model regardless of the agent's default (privacy vault).
    force_local: bool = False


class LLMResponse(BaseModel):
    content: str
    model_name: str
    provider: ModelProvider
    tokens_used: Optional[int] = None
    cloud_call: bool = Field(
        description="True if this consumed cloud free-tier quota; the calling "
        "node MUST increment the matching QuotaTracker counter."
    )


def resolve_model(agent: AgentRole, force_local: bool = False) -> tuple[str, ModelProvider]:
    if force_local:
        return MODEL_MAP[AgentRole.EXECUTOR]
    return MODEL_MAP[agent]


def complete(request: LLMRequest) -> LLMResponse:
    """One completion, any backend. Raises on transport errors — callers
    (nodes) own retry/escalation bookkeeping per project guide §4."""
    model, provider = resolve_model(request.agent, request.force_local)

    kwargs: dict = {
        "model": model,
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
    }
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if provider is ModelProvider.OLLAMA_LOCAL:
        kwargs["api_base"] = OLLAMA_BASE
    elif (api_key := secrets_store.get_secret(PROVIDER_KEYS[provider])) is not None:
        kwargs["api_key"] = api_key

    resp = litellm.completion(**kwargs)

    usage = getattr(resp, "usage", None)
    return LLMResponse(
        content=resp.choices[0].message.content or "",
        model_name=model,
        provider=provider,
        tokens_used=getattr(usage, "total_tokens", None),
        cloud_call=provider is not ModelProvider.OLLAMA_LOCAL,
    )


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts for the memory store — always the local model, never cloud.
    Raises on transport errors; memory.py owns nothing beyond passing them up."""
    resp = litellm.embedding(model=EMBED_MODEL, input=list(texts),
                             api_base=OLLAMA_BASE)
    items = sorted(resp.data, key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in items]


def assert_local_backend_ready() -> None:
    """Fail fast at startup if the local Ollama server is unreachable or the
    embedding model isn't pulled. The privacy vault is load-bearing: without
    it, privacy_sensitive tasks have no legal execution path."""
    import requests

    try:
        r = requests.get(f"{OLLAMA_BASE}/api/version", timeout=3)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001 — any failure means "not ready"
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_BASE}. Start it with "
            "OLLAMA_VULKAN=1 (gfx1103 needs the Vulkan backend, see project guide §3)."
        ) from e

    embed_name = EMBED_MODEL.split("/", 1)[1]
    tags = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3).json()
    pulled = {m.get("name", "") for m in tags.get("models", [])}
    if not any(n == embed_name or n.startswith(f"{embed_name}:") for n in pulled):
        raise RuntimeError(
            f"Embedding model missing — long-term memory recall cannot work. "
            f"Run: ollama pull {embed_name}")
