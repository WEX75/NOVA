"""
NOVA — Unified LLM interface (LiteLLM).

Every node calls `complete()` regardless of backend. This is the ONLY module
that knows provider strings; nodes know AgentRole, nothing else.

Local backend note (project guide §3): Ollama on this machine (Radeon 780M /
gfx1103, not ROCm-supported) can silently fall back to slow CPU if the iGPU
isn't picked up. The verified-working service config is OLLAMA_IGPU_ENABLE=1
on the `ollama serve` process (see §3 — re-verify with `ollama ps` after any
Ollama upgrade). `assert_local_backend_ready()` checks reachability and the
embed model at startup.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import litellm
from pydantic import BaseModel, Field

import secrets_store
from nova_state import AgentRole, ModelProvider

OLLAMA_BASE = os.environ.get("NOVA_OLLAMA_BASE", "http://localhost:11434")

# Wall-clock ceiling for one completion. Grounded in observation, not guessed:
# live single qwen2.5-coder:14b calls on this box measured ~13s and ~27s, and
# a cold model load adds ~6s — 120s is ~4x the worst observed steady call with
# cold-load headroom. A timeout closes the connection, which Ollama honors by
# cancelling generation server-side (verified 2026-07-17, Ollama 0.31.1:
# gpu_busy_percent 98 -> 0 within seconds of the client giving up).
DEFAULT_TIMEOUT = float(os.environ.get("NOVA_LLM_TIMEOUT", "120"))

# How often the cancellable path re-checks the abort signal between chunks.
_ABORT_POLL_SECONDS = 0.3


class LLMAborted(RuntimeError):
    """An external abort signal cancelled the call. NOT a transport failure:
    this is an owner decision, so callers must treat it as terminal — never
    retry it, never route it through llm_transport_failure."""

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


def _close_stream(stream) -> None:
    """Really close a litellm stream's underlying HTTP connection.

    litellm's CustomStreamWrapper has no close() of its own, and simply
    abandoning the iterator leaves the connection open — measured on this
    box: Ollama then generates the discarded answer to completion with the
    GPU pinned ~97%. Closing the wrapped httpx line-generator
    (completion_stream.streaming_response) closes the socket, and Ollama
    honors the disconnect by cancelling generation (gpu_busy_percent -> 0
    within seconds; verified 2026-07-17 on Ollama 0.31.1). Best-effort by
    design: a close failure must not mask the reason we are closing."""
    inner = getattr(stream, "completion_stream", None)
    for obj in (getattr(inner, "streaming_response", None),
                getattr(inner, "response_iterator", None),
                inner):
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001, S110 — cleanup only
                pass


def complete(request: LLMRequest, *, timeout: Optional[float] = None,
             should_abort: Optional[Callable[[], bool]] = None) -> LLMResponse:
    """One completion, any backend. Raises on transport errors — callers
    (nodes) own retry/escalation bookkeeping per project guide §4.

    ONE cancellation mechanism, two triggers (§14): `timeout` (wall-clock,
    DEFAULT_TIMEOUT unless overridden) and `should_abort` (an external
    signal, polled between stream chunks). Both end the call by closing the
    underlying connection, which really stops Ollama's generation. A timeout
    raise is deliberately a plain exception so it rides the EXISTING
    llm_transport_failure path — retry then human gate, no quota charge, no
    new failure category. An abort raises LLMAborted, which is terminal.

    Without `should_abort` the call is non-streaming and litellm's own
    timeout closes the connection (same measured effect). With it, the call
    streams so there is a connection to close mid-generation."""
    model, provider = resolve_model(request.agent, request.force_local)
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout

    if should_abort is not None and should_abort():
        raise LLMAborted(f"{model}: abort requested before the call started")

    kwargs: dict = {
        "model": model,
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "timeout": timeout,
    }
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if provider is ModelProvider.OLLAMA_LOCAL:
        kwargs["api_base"] = OLLAMA_BASE
    elif (api_key := secrets_store.get_secret(PROVIDER_KEYS[provider])) is not None:
        kwargs["api_key"] = api_key

    if should_abort is None:
        resp = litellm.completion(**kwargs)
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model_name=model,
            provider=provider,
            tokens_used=getattr(usage, "total_tokens", None),
            cloud_call=provider is not ModelProvider.OLLAMA_LOCAL,
        )

    # Cancellable path: stream, and always close — see _close_stream for why
    # abandoning the wrapper would leave the GPU computing a discarded answer.
    deadline = time.monotonic() + timeout
    stream = litellm.completion(**kwargs, stream=True)
    parts: list[str] = []
    usage = None
    last_poll = 0.0
    try:
        for chunk in stream:
            now = time.monotonic()
            if now > deadline:
                raise TimeoutError(
                    f"{model}: generation exceeded {timeout:.0f}s wall clock")
            if now - last_poll >= _ABORT_POLL_SECONDS:
                last_poll = now
                if should_abort():
                    raise LLMAborted(f"{model}: abort requested mid-generation")
            choices = getattr(chunk, "choices", None)
            if choices:
                delta = getattr(choices[0].delta, "content", None)
                if delta:
                    parts.append(delta)
            usage = getattr(chunk, "usage", None) or usage
    finally:
        _close_stream(stream)

    return LLMResponse(
        content="".join(parts),
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
            f"Ollama not reachable at {OLLAMA_BASE}. Start the service with "
            "OLLAMA_IGPU_ENABLE=1 (gfx1103 iGPU, see project guide §3; verify GPU "
            "use with `ollama ps`)."
        ) from e

    embed_name = EMBED_MODEL.split("/", 1)[1]
    tags = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3).json()
    pulled = {m.get("name", "") for m in tags.get("models", [])}
    if not any(n == embed_name or n.startswith(f"{embed_name}:") for n in pulled):
        raise RuntimeError(
            f"Embedding model missing — long-term memory recall cannot work. "
            f"Run: ollama pull {embed_name}")
