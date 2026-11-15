"""Issue #13 / Arbeitspaket C: pre-call admission control for paid model calls.

The ceiling is enforced BEFORE the provider is called, by holding the next
call's worst-case price against the order budget. A pure post-hoc sum would
already have spent the money it is supposed to prevent.
"""

from decimal import Decimal

import pytest

from agent import cost_budget
from hermes_state import SessionDB


class _FakeAgent:
    def __init__(self, db, order_id=None, session_id="s1", model="test/model", provider="openrouter"):
        self._session_db = db
        self.session_id = session_id
        self._cost_order_id = order_id
        self.model = model
        self.provider = provider
        self.base_url = ""
        self.api_key = ""
        self._pending_call_budget = None


def _priced(monkeypatch, price="0.00001"):
    """Make every route priced with a deterministic per-token rate."""
    entry = cost_budget.PricingEntry(
        input_cost_per_million=Decimal(10), output_cost_per_million=Decimal(10),
        cache_read_cost_per_million=Decimal(10), cache_write_cost_per_million=Decimal(10),
        source="official_docs_snapshot",
    )
    monkeypatch.setattr(cost_budget, "_pricing_entry", lambda *a, **k: entry)
    person = _FakeAgent
    return person


def test_no_order_id_means_no_enforcement(tmp_path, monkeypatch):
    _priced(monkeypatch)
    db = SessionDB(tmp_path / "state.db")
    try:
        agent = _FakeAgent(db, order_id=None)
        assert cost_budget.reserve_for_call(
            agent, prompt_tokens=100, max_output_tokens=100
        ) is None
    finally:
        db.close()


def test_unpriced_route_is_refused_not_treated_as_free(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_budget, "_pricing_entry", lambda *a, **k: None)
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("o1", limit_usd="5.00")
        agent = _FakeAgent(db, order_id="o1")
        with pytest.raises(cost_budget.CostLimitExceeded) as exc:
            cost_budget.reserve_for_call(agent, prompt_tokens=100, max_output_tokens=100)
        assert exc.value.reason == "unpriced"
        assert db.get_cost_order("o1")["reserved_usd"] == "0.000000"
    finally:
        db.close()


def test_unknown_order_stops_instead_of_guessing_a_budget(tmp_path, monkeypatch):
    _priced(monkeypatch)
    db = SessionDB(tmp_path / "state.db")
    try:
        agent = _FakeAgent(db, order_id="never-created")
        with pytest.raises(cost_budget.CostLimitExceeded) as exc:
            cost_budget.reserve_for_call(agent, prompt_tokens=10, max_output_tokens=10)
        assert exc.value.reason == "unknown_order"
    finally:
        db.close()


def test_reservation_is_held_then_settled_to_the_real_cost(tmp_path, monkeypatch):
    _priced(monkeypatch)
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("o2", limit_usd="1.00")
        agent = _FakeAgent(db, order_id="o2")
        budget = cost_budget.reserve_for_call(agent, prompt_tokens=1000, max_output_tokens=1000)
        assert budget is not None
        # 2000 tokens * 10 USD / 1M = 0.02 held before the call
        assert budget.worst_case_usd == Decimal("0.020000")
        assert db.get_cost_order("o2")["reserved_usd"] == "0.020000"
        assert cost_budget.settle_pending_call(agent, Decimal("0.0004")) == "settled"
        order = db.get_cost_order("o2")
        assert order["reserved_usd"] == "0.000000"
        assert order["spent_usd"] == "0.000400"
        # A second settle of the same reservation must not book again.
        assert cost_budget.settle_pending_call(agent, Decimal("0.0004")) == "no_pending"
        assert db.get_cost_order("o2")["spent_usd"] == "0.000400"
    finally:
        db.close()


def test_exhausted_order_blocks_the_next_call(tmp_path, monkeypatch):
    _priced(monkeypatch)
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("o3", limit_usd="0.02")
        agent = _FakeAgent(db, order_id="o3")
        assert cost_budget.reserve_for_call(agent, prompt_tokens=1000, max_output_tokens=1000) is not None
        # The first call is still in flight and already holds the whole ceiling.
        second = _FakeAgent(db, order_id="o3", session_id="s2")
        with pytest.raises(cost_budget.CostLimitExceeded) as exc:
            cost_budget.reserve_for_call(second, prompt_tokens=1000, max_output_tokens=1000)
        assert exc.value.reason == "exhausted"
        cost_budget.settle_pending_call(agent, Decimal("0.02"))
        with pytest.raises(cost_budget.CostLimitExceeded):
            cost_budget.reserve_for_call(second, prompt_tokens=1, max_output_tokens=1)
    finally:
        db.close()


def test_failed_call_releases_the_hold(tmp_path, monkeypatch):
    _priced(monkeypatch)
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("o4", limit_usd="1.00")
        agent = _FakeAgent(db, order_id="o4")
        cost_budget.reserve_for_call(agent, prompt_tokens=1000, max_output_tokens=1000)
        assert cost_budget.release_pending_call(agent) == "released"
        assert db.get_cost_order("o4")["reserved_usd"] == "0.000000"
        assert db.get_cost_order("o4")["spent_usd"] == "0.000000"
    finally:
        db.close()


def test_missing_usage_keeps_the_hold_rather_than_booking_zero(tmp_path, monkeypatch):
    _priced(monkeypatch)
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("o5", limit_usd="1.00")
        agent = _FakeAgent(db, order_id="o5")
        cost_budget.reserve_for_call(agent, prompt_tokens=1000, max_output_tokens=1000)
        # No usage reported: the cost is unknown, so the conservative hold stays.
        assert cost_budget.settle_pending_call(agent, None) == "unknown_cost"
        assert db.get_cost_order("o5")["reserved_usd"] == "0.020000"
        assert db.get_cost_order("o5")["spent_usd"] == "0.000000"
    finally:
        db.close()


def test_order_id_is_propagated_to_delegated_agents():
    # Children spend from the parent's order ceiling, never from a fresh budget.
    assert cost_budget.SUBAGENT_ATTRS == ("cost_order_id",)


def test_budget_exception_is_a_runtime_error():
    err = cost_budget.CostLimitExceeded("exhausted", order_id="o", detail="limit 1.00")
    assert isinstance(err, RuntimeError)
    assert "cost ceiling" in str(err).lower()


def test_configured_limit_turns_the_session_into_its_own_order(tmp_path, monkeypatch):
    _priced(monkeypatch)
    monkeypatch.setattr(cost_budget, "configured_limit_usd", lambda: Decimal("0.50"))
    db = SessionDB(tmp_path / "state.db")
    try:
        agent = _FakeAgent(db, order_id=None)
        assert cost_budget.order_id_for(agent) == "session:s1"
        assert db.get_cost_order("session:s1")["limit_usd"] == "0.500000"
        # Resolved once per run: re-entry never re-creates or re-arms the order.
        assert cost_budget.order_id_for(agent) == "session:s1"
    finally:
        db.close()


def test_two_parallel_children_share_one_ceiling(tmp_path, monkeypatch):
    _priced(monkeypatch)
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("order-shared", limit_usd="0.03")
        first = _FakeAgent(db, order_id="order-shared", session_id="child-1")
        second = _FakeAgent(db, order_id="order-shared", session_id="child-2")
        assert cost_budget.reserve_for_call(first, prompt_tokens=1000, max_output_tokens=1000) is not None
        # The sibling sees the sibling's hold: fan-out cannot multiply the ceiling.
        with pytest.raises(cost_budget.CostLimitExceeded) as exc:
            cost_budget.reserve_for_call(second, prompt_tokens=1000, max_output_tokens=1000)
        assert exc.value.reason == "exhausted"
        cost_budget.release_pending_call(first)
        assert cost_budget.reserve_for_call(second, prompt_tokens=1000, max_output_tokens=1000) is not None
    finally:
        db.close()


def test_shipped_default_config_has_no_ceiling(monkeypatch):
    # No invented dollar figure: without an operator setting, nothing is enforced.
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", dict, raising=False,
    )
    assert cost_budget.configured_limit_usd() is None


def test_a_disabled_or_unusable_limit_is_not_enforced(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"cost_limits": {"enabled": False, "order_limit_usd": "5.00"}}, raising=False,
    )
    assert cost_budget.configured_limit_usd() is None
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"cost_limits": {"order_limit_usd": "not-a-number"}}, raising=False,
    )
    assert cost_budget.configured_limit_usd() is None