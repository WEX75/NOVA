"""
NOVA — Long-term memory (Chroma, local embedded, project guide §9).

RAG recall of past business updates and preferences. Fully local: persistent
client + nomic-embed-text on the local Ollama (via llm.embed — only llm.py
knows provider strings), so memory content never leaves the machine and
embedding rides the Vulkan GPU path (project guide §3). NovaState carries only
the returned ids/snippets in memory_refs, never full documents.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

_PERSIST_DIR = Path(os.environ.get("NOVA_CHROMA_DIR", Path(__file__).parent / "chroma_db"))
_COLLECTION = "nova_memory"

_client = None  # lazy singleton — chromadb import is heavy
_embedding_function = None  # None -> local Ollama embedder (production default)


def configure(persist_dir: Path | str | None = None, embedding_function=None) -> None:
    """Re-point the store. Production never calls this; tests inject a tmp
    dir and a deterministic embedder (the real embedder needs a running
    Ollama, which tests must not depend on)."""
    global _client, _PERSIST_DIR, _embedding_function
    _client = None
    if persist_dir is not None:
        _PERSIST_DIR = Path(persist_dir)
    _embedding_function = embedding_function


def _default_embedder():
    """Chroma adapter for llm.embed(). NOTE: an existing collection keeps the
    vectors it was written with — swapping embedding models later means
    re-embedding (or starting a fresh persist dir)."""
    from chromadb.api.types import EmbeddingFunction

    class _OllamaEmbedder(EmbeddingFunction):
        def __call__(self, input):
            import llm
            return llm.embed(list(input))

        @staticmethod
        def name():
            return "nova-local-ollama"

    return _OllamaEmbedder()


def _collection():
    global _client
    if _client is None:
        import chromadb
        _client = chromadb.PersistentClient(path=str(_PERSIST_DIR))
    ef = _embedding_function if _embedding_function is not None else _default_embedder()
    return _client.get_or_create_collection(_COLLECTION, embedding_function=ef)


class MemoryHit(BaseModel):
    memory_id: str
    text: str
    distance: Optional[float] = None


def remember(text: str, kind: str = "business_update",
             task_id: Optional[str] = None) -> str:
    """Store one fact/update; returns its id (goes in NovaState.memory_refs)."""
    memory_id = f"mem-{uuid.uuid4().hex[:12]}"
    _collection().add(
        ids=[memory_id],
        documents=[text],
        metadatas=[{"kind": kind, "task_id": task_id or "",
                    "stored_at": datetime.now(timezone.utc).isoformat()}],
    )
    return memory_id


def recall(query: str, n: int = 4) -> list[str]:
    """Top-n relevant memories as plain snippets (Executor's memory_recall tool)."""
    return [hit.text for hit in recall_detailed(query, n)]


def recall_detailed(query: str, n: int = 4) -> list[MemoryHit]:
    col = _collection()
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(n, col.count()))
    return [
        MemoryHit(memory_id=i, text=d, distance=dist)
        for i, d, dist in zip(res["ids"][0], res["documents"][0],
                              (res.get("distances") or [[None] * len(res["ids"][0])])[0])
    ]
