"""Quota self-monitoring: routing falls back to the local
Executor BEFORE a 429, every cloud call bumps the right counter, and the
per-day counters self-heal across the UTC midnight boundary."""

from datetime import datetime, timedelta, timezone

from conftest import make_state, make_subtask
from llm import LLMResponse
from nodes.common import bump_quota
from nova_state import AgentRole, ModelProvider, QuotaTracker, TaskStatus
from route_next_node import route_next_node


def exhausted(gemini: int | None = None, groq: int | None = None) -> QuotaTracker:
    q = QuotaTracker()
    q.gemini_requests_today = q.gemini_daily_limit if gemini is None else gemini
    q.groq_requests_today = q.groq_daily_limit if groq is None else groq
    return q


# --- routing fallback -------------------------------------------------------

def test_gemini_exhausted_pending_falls_back_to_executor():
    state = make_state(status=TaskStatus.PENDING, quota=exhausted(groq=0))
    assert route_next_node(state) == "executor"


def test_gemini_one_below_limit_still_routes_to_director():
    q = QuotaTracker()
    state = make_state(status=TaskStatus.PENDING,
                       quota=exhausted(gemini=q.gemini_daily_limit - 1, groq=0))
    assert route_next_node(state) == "director"


def test_gemini_exhausted_researcher_subtask_falls_back():
    state = make_state(status=TaskStatus.IN_PROGRESS, quota=exhausted(groq=0),
                       subtasks=[make_subtask("find info", AgentRole.RESEARCHER)])
    assert route_next_node(state) == "executor"


def test_groq_exhausted_critic_subtask_falls_back():
    state = make_state(status=TaskStatus.IN_PROGRESS, quota=exhausted(gemini=0),
                       subtasks=[make_subtask("review this", AgentRole.CRITIC)])
    assert route_next_node(state) == "executor"


def test_groq_exhausted_does_not_block_researcher():
    """Per-provider fallback: Groq being out must not ground Gemini agents."""
    state = make_state(status=TaskStatus.IN_PROGRESS, quota=exhausted(gemini=0),
                       subtasks=[make_subtask("find info", AgentRole.RESEARCHER)])
    assert route_next_node(state) == "researcher"


def test_executor_subtask_ignores_quota_entirely():
    state = make_state(status=TaskStatus.IN_PROGRESS, quota=exhausted(),
                       subtasks=[make_subtask(agent=AgentRole.EXECUTOR)])
    assert route_next_node(state) == "executor"


# --- UTC day rollover ---------------------------------------------------------
# gemini/groq_available() are PURE reads (yesterday's spend counts as zero,
# nothing written) so route_next_node stays a pure reader; record_request()
# is the only writer of the rollover.

def _yesterday() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=1)


def test_available_treats_yesterdays_spend_as_zero_without_writing():
    q = exhausted()
    q.last_reset = _yesterday()
    before = q.model_copy(deep=True)
    assert q.gemini_available()                       # effective count is 0...
    assert q.groq_available()
    assert q == before                                # ...and NOTHING was mutated


def test_available_same_day_counts_for_real():
    q = exhausted()
    assert not q.gemini_available()
    assert not q.groq_available()


def test_record_request_persists_the_rollover_then_increments():
    q = exhausted()
    q.last_reset = _yesterday()
    fresh = q.record_request("gemini")
    assert fresh.gemini_requests_today == 1           # reset to 0, then +1
    assert fresh.groq_requests_today == 0             # BOTH counters reset
    assert fresh.last_reset.date() == datetime.now(timezone.utc).date()
    assert q.gemini_requests_today == q.gemini_daily_limit  # original untouched


def test_record_request_same_day_only_increments():
    q = QuotaTracker(gemini_requests_today=5, groq_requests_today=7)
    fresh = q.record_request("groq")
    assert fresh.groq_requests_today == 8
    assert fresh.gemini_requests_today == 5
    assert fresh.last_reset == q.last_reset           # no spurious reset


def test_naive_last_reset_is_treated_as_utc():
    """A checkpoint round-trip may hand back a naive datetime; both the pure
    read and the writer must treat it as UTC."""
    q = exhausted()
    q.last_reset = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    assert q.gemini_available()
    assert q.record_request("gemini").gemini_requests_today == 1


def test_router_recovers_from_yesterdays_exhaustion_without_any_tick():
    """A long-running process that exhausted its quota yesterday routes to
    the Director again after midnight, purely from the pure read."""
    q = exhausted(groq=0)
    q.last_reset = _yesterday()
    state = make_state(status=TaskStatus.PENDING, quota=q)
    assert route_next_node(state) == "director"


def test_zeroed_daily_limit_survives_rollover_in_record_request():
    """preflight() disables a keyless provider via daily_limit=0; midnight
    resets counters, never limits — even accounting a request on a stale day
    must not re-enable the provider."""
    q = QuotaTracker(gemini_daily_limit=0)
    q.last_reset = _yesterday()
    assert not q.gemini_available()
    fresh = q.record_request("gemini")                # defense in depth
    assert fresh.gemini_requests_today == 1
    assert not fresh.gemini_available()               # limit 0 held through reset


# --- counter bookkeeping (§4: increment after EVERY cloud call) -------------

def _resp(provider: ModelProvider, cloud: bool) -> LLMResponse:
    return LLMResponse(content="x", model_name="m", provider=provider,
                       tokens_used=10, cloud_call=cloud)


def test_bump_quota_gemini_increments_only_gemini():
    state = make_state()
    quota = bump_quota(state, _resp(ModelProvider.GEMINI, cloud=True))
    assert quota.gemini_requests_today == 1
    assert quota.groq_requests_today == 0


def test_bump_quota_groq_increments_only_groq():
    state = make_state()
    quota = bump_quota(state, _resp(ModelProvider.GROQ, cloud=True))
    assert quota.groq_requests_today == 1
    assert quota.gemini_requests_today == 0


def test_bump_quota_local_call_is_free():
    state = make_state()
    quota = bump_quota(state, _resp(ModelProvider.OLLAMA_LOCAL, cloud=False))
    assert quota.gemini_requests_today == 0
    assert quota.groq_requests_today == 0


def test_bump_quota_never_mutates_state():
    """Nodes return fresh trackers; reducers merge. Mutation would corrupt
    LangGraph checkpointing."""
    state = make_state()
    bump_quota(state, _resp(ModelProvider.GEMINI, cloud=True))
    assert state.quota.gemini_requests_today == 0
