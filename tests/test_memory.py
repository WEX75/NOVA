"""Long-term memory (project guide §9): remember/recall roundtrip against a real
embedded Chroma store, and the Executor's read-only memory_recall tool path.
A deterministic bag-of-words embedder replaces the default ONNX one so tests
stay offline and stable."""

import hashlib

import pytest
from chromadb.api.types import EmbeddingFunction

import memory
from conftest import make_state
from tools import ToolInput, run_tool

# Wide enough that no two words in the test vocabulary share a bucket; a
# single cross-text collision can flip a ranking decided by a 2-word margin.
_DIM = 2048

# bucket -> the word that claimed it, across the whole test run: if fixture
# text added later ever collides, the run fails loudly naming both words
# instead of silently ranking wrong.
_BUCKET_OWNERS: dict[int, str] = {}


def _bucket(word: str) -> int:
    # md5, not builtin hash(): hash() is salted per process, so rankings
    # flipped between runs depending on PYTHONHASHSEED.
    bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:8], "big") % _DIM
    owner = _BUCKET_OWNERS.setdefault(bucket, word)
    assert owner == word, (
        f"hash collision between {owner!r} and {word!r} at _DIM={_DIM} — "
        f"bump _DIM or reword the fixture text")
    return bucket


class BagOfWordsEmbedder(EmbeddingFunction):
    """Deterministic, dependency-free: hash words into a fixed-size vector so
    texts sharing words land near each other."""

    def __call__(self, input):
        vectors = []
        for text in input:
            v = [0.0] * _DIM
            for word in text.lower().split():
                v[_bucket(word)] += 1.0
            vectors.append(v)
        return vectors

    @staticmethod
    def name():
        return "bag-of-words-test-embedder"


@pytest.fixture(autouse=True)
def temp_memory(tmp_path):
    memory.configure(persist_dir=tmp_path / "chroma",
                     embedding_function=BagOfWordsEmbedder())
    yield
    memory.configure(persist_dir=memory._PERSIST_DIR, embedding_function=None)
    memory._client = None


def test_remember_returns_id_suitable_for_memory_refs():
    memory_id = memory.remember("Q2 revenue grew 4 percent", task_id="t1")
    assert memory_id.startswith("mem-")


def test_recall_finds_the_relevant_memory():
    memory.remember("Q2 revenue grew 4 percent versus Q1")
    memory.remember("the office plants need watering on fridays")
    hits = memory.recall("what was the revenue growth in Q2", n=1)
    assert hits == ["Q2 revenue grew 4 percent versus Q1"]


def test_recall_detailed_carries_ids_and_distances():
    stored_id = memory.remember("owner prefers summaries under five bullet points")
    hit = memory.recall_detailed("owner summary preferences", n=1)[0]
    assert hit.memory_id == stored_id
    assert hit.distance is not None


def test_recall_on_empty_store_returns_nothing():
    assert memory.recall("anything at all") == []


def test_memories_persist_across_client_restarts(tmp_path):
    memory.remember("churn stayed flat in Q2")
    memory._client = None  # simulate process restart; same persist dir
    assert memory.recall("what happened to churn", n=1) == ["churn stayed flat in Q2"]


def test_production_default_embedder_is_local_llm_embed(tmp_path, monkeypatch):
    """With no injected embedder (the production configuration), memory must
    embed through llm.embed — the local Ollama path — and never anything else."""
    import llm

    batches: list[list[str]] = []

    def fake_embed(texts):
        batches.append(list(texts))
        return BagOfWordsEmbedder()(texts)  # deterministic stand-in vectors

    monkeypatch.setattr(llm, "embed", fake_embed)
    memory.configure(persist_dir=tmp_path / "prod_chroma", embedding_function=None)

    memory.remember("Q2 revenue grew 4 percent versus Q1")
    hits = memory.recall("what was the revenue growth in Q2", n=1)

    assert hits == ["Q2 revenue grew 4 percent versus Q1"]
    assert batches != []                       # everything embedded via llm.embed
    assert any("Q2 revenue grew 4 percent versus Q1" in b for b in batches)


def test_executor_memory_recall_tool_is_read_only_and_works(temp_backlog):
    """The RAG path the Executor actually uses: read_only tier, no approval
    needed, audit-logged like every tool call."""
    memory.remember("Q2 revenue grew 4 percent versus Q1")
    state = make_state(task_id="t-recall")
    result = run_tool(state, ToolInput(tool_name="memory_recall",
                                       arg="revenue growth Q2", reason="RAG recall"))
    assert result.success
    assert "revenue grew 4 percent" in (result.output or "")
    assert any(e["action_type"] == "tool:memory_recall"
               for e in temp_backlog.log_entries(task_id="t-recall"))
