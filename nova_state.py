"""
NOVA — LangGraph State Schema
Tracks a single task's lifecycle across the hybrid Director -> Researcher/Critic -> Executor pipeline.
"""

from __future__ import annotations
from enum import Enum
from typing import Annotated, Optional, Literal
from datetime import datetime, timezone
import operator

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    DIRECTOR = "director"       # Gemini 2.5/3 Flash (cloud) — planning, task decomposition
    RESEARCHER = "researcher"   # Gemini Flash (cloud) — info gathering
    CRITIC = "critic"           # Groq Llama 3.3 70B (cloud) — fast review/error-checking
    EXECUTOR = "executor"       # Ollama Qwen2.5-Coder (local) — file/code/git operations
    MANAGER = "manager"         # local — email triage, task backlog
    TUTOR = "tutor"             # local — Socratic tutoring / code review, prose only (§15)


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    NEEDS_APPROVAL = "needs_approval"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"


class ModelProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    OLLAMA_LOCAL = "ollama_local"


class Subtask(BaseModel):
    """A single decomposed unit of work, produced by the Director."""
    description: str
    assigned_agent: AgentRole
    requires_approval: bool = False


class ToolCallResult(BaseModel):
    tool_name: str
    input: dict
    output: Optional[str] = None
    success: bool
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentStep(BaseModel):
    """One entry in the execution history. Every hop through the graph appends one of these."""
    agent: AgentRole
    provider: ModelProvider
    model_name: str  # e.g. "gemini-2.5-flash", "llama-3.3-70b-versatile", "qwen2.5-coder:14b"
    input_summary: str
    output_summary: str
    tool_calls: list[ToolCallResult] = Field(default_factory=list)
    tokens_used: Optional[int] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QuotaTracker(BaseModel):
    """
    Self-monitoring against free-tier limits so Nova can fall back to the local
    Executor model BEFORE hitting a 429, instead of failing mid-task.
    Defaults reflect free-tier caps as of mid-2026 — re-check periodically, these move.
    """
    gemini_requests_today: int = 0
    groq_requests_today: int = 0
    gemini_daily_limit: int = 250    # Gemini 2.5 Flash free tier
    groq_daily_limit: int = 1000     # Llama 3.3 70B free tier
    last_reset: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def _is_stale_day(self) -> bool:
        """True if last_reset falls on an earlier UTC calendar day than now.
        A naive last_reset (possible after a checkpoint round-trip) is
        treated as UTC."""
        last = (self.last_reset if self.last_reset.tzinfo is not None
                else self.last_reset.replace(tzinfo=timezone.utc))
        return last.astimezone(timezone.utc).date() != datetime.now(timezone.utc).date()

    def _effective(self, count: int) -> int:
        """Yesterday's spend counts as zero — without writing anything."""
        return 0 if self._is_stale_day() else count

    def gemini_available(self) -> bool:
        """PURE read (owner decision 2026-07-13): safe to call from
        route_next_node, which must never mutate state. The day rollover is
        only ever PERSISTED by record_request()."""
        return self._effective(self.gemini_requests_today) < self.gemini_daily_limit

    def groq_available(self) -> bool:
        return self._effective(self.groq_requests_today) < self.groq_daily_limit

    def record_request(self, provider: Literal["gemini", "groq"]) -> "QuotaTracker":
        """Account one cloud request: returns a NEW tracker with the counters
        reset first if the UTC day rolled over, then the provider's counter
        incremented. This is the ONLY place a rollover is written; nodes put
        the returned tracker in their partial state update.
        To disable a provider outright (e.g. missing API key), zero its
        daily_limit — counters do not survive midnight, limits do."""
        fresh = self.model_copy(deep=True)
        if fresh._is_stale_day():
            fresh.gemini_requests_today = 0
            fresh.groq_requests_today = 0
            fresh.last_reset = datetime.now(timezone.utc)
        if provider == "gemini":
            fresh.gemini_requests_today += 1
        elif provider == "groq":
            fresh.groq_requests_today += 1
        else:  # typing forbids this, but a runtime guard beats silent miscounting
            raise ValueError(f"unknown cloud provider {provider!r}")
        return fresh


class NovaState(BaseModel):
    """
    Full state object passed between LangGraph nodes.
    Each node reads what it needs and returns a partial update; list fields use
    the `operator.add` reducer so history accumulates across hops instead of
    being overwritten.
    """

    # --- Task identity ---
    task_id: str
    goal: str
    source: Literal["email", "voice", "manual", "tutor"] = "manual"
    privacy_sensitive: bool = False   # True = local Executor only, never leaves the machine

    # --- Planning ---
    status: TaskStatus = TaskStatus.PENDING
    subtasks: list[Subtask] = Field(default_factory=list)
    current_subtask_index: int = 0

    # --- Execution history (append-only) ---
    history: Annotated[list[AgentStep], operator.add] = Field(default_factory=list)

    # --- Routing ---
    current_agent: Optional[AgentRole] = None
    retry_count: int = 0
    max_retries: int = 3

    # --- Human-in-the-loop ---
    requires_human_approval: bool = False
    approved: Optional[bool] = None

    # --- Outputs ---
    final_output: Optional[str] = None
    error_log: Annotated[list[str], operator.add] = Field(default_factory=list)

    # --- Cost/quota self-monitoring (see QuotaTracker) ---
    quota: QuotaTracker = Field(default_factory=QuotaTracker)

    # --- Long-term memory refs (vector DB pointers, not full content) ---
    memory_refs: list[str] = Field(default_factory=list)
