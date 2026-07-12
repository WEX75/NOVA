"""Every branch of route_next_node (locked file — these tests pin its contract)."""

from langgraph.graph import END

from conftest import make_state, make_subtask, critic_step
from nova_state import AgentRole, QuotaTracker, TaskStatus
from route_next_node import route_next_node


# 1 — terminal states
def test_done_ends():
    assert route_next_node(make_state(status=TaskStatus.DONE)) == END


def test_failed_ends():
    assert route_next_node(make_state(status=TaskStatus.FAILED)) == END


# 2 — retry ceiling
def test_retry_ceiling_escalates_to_human():
    assert route_next_node(make_state(retry_count=3)) == "human_gate"


def test_retry_ceiling_beats_privacy_routing():
    state = make_state(retry_count=3, privacy_sensitive=True)
    assert route_next_node(state) == "human_gate"


# 3 — human gate
def test_pending_approval_goes_to_gate():
    state = make_state(requires_human_approval=True, approved=None)
    assert route_next_node(state) == "human_gate"


def test_approved_true_does_not_loop_on_gate():
    state = make_state(requires_human_approval=True, approved=True,
                       privacy_sensitive=True)
    assert route_next_node(state) == "executor"


# 4 — privacy vault
def test_privacy_sensitive_always_executor():
    state = make_state(privacy_sensitive=True, status=TaskStatus.PENDING)
    assert route_next_node(state) == "executor"


def test_privacy_sensitive_skips_director_even_with_quota():
    state = make_state(privacy_sensitive=True, subtasks=[], status=TaskStatus.PENDING)
    assert state.quota.gemini_available()
    assert route_next_node(state) == "executor"


# 5 — undecomposed -> director (with quota fallback)
def test_pending_goes_to_director():
    assert route_next_node(make_state(status=TaskStatus.PENDING)) == "director"


def test_no_subtasks_goes_to_director():
    state = make_state(status=TaskStatus.IN_PROGRESS, subtasks=[])
    assert route_next_node(state) == "director"


# 6 — per-subtask assignment
def test_researcher_subtask_routes_to_researcher():
    state = make_state(status=TaskStatus.IN_PROGRESS,
                       subtasks=[make_subtask(agent=AgentRole.RESEARCHER)])
    assert route_next_node(state) == "researcher"


def test_critic_subtask_first_pass_routes_to_critic():
    state = make_state(status=TaskStatus.IN_PROGRESS,
                       subtasks=[make_subtask("review this", AgentRole.CRITIC)])
    assert route_next_node(state) == "critic"


def test_critic_single_pass_then_executor():
    """The no-debate-loops rule: once reviewed, same subtask goes local."""
    state = make_state(status=TaskStatus.IN_PROGRESS,
                       subtasks=[make_subtask("review this", AgentRole.CRITIC)],
                       history=[critic_step("review this")])
    assert route_next_node(state) == "executor"


def test_critic_block_verdict_still_counts_as_reviewed():
    state = make_state(status=TaskStatus.IN_PROGRESS,
                       subtasks=[make_subtask("review this", AgentRole.CRITIC)],
                       history=[critic_step("review this", verdict="block")])
    assert route_next_node(state) == "executor"


def test_executor_subtask_routes_to_executor():
    state = make_state(status=TaskStatus.IN_PROGRESS,
                       subtasks=[make_subtask(agent=AgentRole.EXECUTOR)])
    assert route_next_node(state) == "executor"


def test_second_subtask_routing_follows_index():
    state = make_state(status=TaskStatus.IN_PROGRESS, current_subtask_index=1,
                       subtasks=[make_subtask(agent=AgentRole.EXECUTOR),
                                 make_subtask("find info", AgentRole.RESEARCHER)])
    assert route_next_node(state) == "researcher"
