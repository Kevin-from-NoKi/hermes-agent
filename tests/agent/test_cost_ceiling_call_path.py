"""Issue #13 / Arbeitspaket C, test 8: the ceiling stops the call BEFORE the provider.

Unit-testing `cost_budget` alone would not prove the wiring; this drives the
real `perform_api_call` chokepoint with an exhausted order and asserts the
provider client is never touched.
"""

from decimal import Decimal

import pytest

from agent import cost_budget
from agent.turn_api_call import perform_api_call
from hermes_state import SessionDB


class _ProviderSpy:
    def __init__(self):
        self.calls = 0

    def __call__(self, api_kwargs):
        self.calls += 1
        from types import SimpleNamespace

        return SimpleNamespace(id="resp-1", usage=None)


class _Agent:
    """Just enough of AIAgent for `perform_api_call` (non-streaming, no middleware)."""

    def __init__(self, db, order_id):
        self.provider = "openrouter"
        self.model = "vendor/model"
        self.base_url = ""
        self.api_key = ""
        self.api_mode = "chat_completions"
        self.session_id = "s-turn"
        self._session_db = db
        self._cost_order_id = order_id
        self._cost_order_disabled = False
        self._pending_call_budget = None
        self._disable_streaming = True
        self._model_request_active = None
        self._pending_redirect_lock = None
        self._last_prompt_size_tokens = 0
        self.thinking_callback = None
        self.log_prefix = ""
        self.platform = "cli"
        self.verbose_logging = False
        self.client = None
        self.spy = _ProviderSpy()
        self._interruptible_api_call = self.spy

    # --- collaborators the call path touches ---
    def _has_stream_consumers(self):
        return False

    def _has_pending_redirect(self):
        return False

    def _safe_print(self, *a, **k):
        pass

    def _vprint(self, *a, **k):
        pass

    def _buffer_vprint(self, *a, **k):
        pass

    def _buffer_diagnostic_status(self, *a, **k):
        pass


def _run(agent):
    return perform_api_call(
        agent, api_kwargs={"messages": [{"role": "user", "content": "hi" * 100}], "max_tokens": 500},
        _original_api_kwargs={}, _llm_middleware_trace=[], _moa_prepared_request=None, _retry=None,
        thinking_spinner=None, retry_count=0, api_call_count=1, api_request_id="req", effective_task_id=None,
        turn_id=None, interrupted=False,
    )


def _entry():
    return cost_budget.PricingEntry(
        input_cost_per_million=Decimal(10), output_cost_per_million=Decimal(10),
        cache_read_cost_per_million=Decimal(10), cache_write_cost_per_million=Decimal(10),
        source="official_docs_snapshot",
    )


def test_exhausted_order_never_reaches_the_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_budget, "_pricing_entry", lambda *a, **k: _entry())
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("turn-order", limit_usd="0.000001")
        agent = _Agent(db, order_id="turn-order")
        with pytest.raises(cost_budget.CostLimitExceeded):
            _run(agent)
        assert agent.spy.calls == 0, "the provider must not be called when the ceiling refused"
        assert db.get_cost_order("turn-order")["spent_usd"] == "0.000000"
    finally:
        db.close()


def test_admitted_call_is_held_then_settled(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_budget, "_pricing_entry", lambda *a, **k: _entry())
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("turn-order", limit_usd="1.00")
        agent = _Agent(db, order_id="turn-order")
        verdict = _run(agent)
        assert verdict.action == "fallthrough"
        assert agent.spy.calls == 1
        held = db.get_cost_order("turn-order")
        assert Decimal(held["reserved_usd"]) > 0, "money is held while the call is unaccounted"
        assert held["spent_usd"] == "0.000000"
        # The settle path belongs to usage accounting; here we prove the hold exists.
        assert cost_budget.settle_pending_call(agent, Decimal("0.5")) == "settled"
        assert db.get_cost_order("turn-order")["spent_usd"] == "0.500000"
    finally:
        db.close()


def test_unpriced_route_never_becomes_a_free_call(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_budget, "_pricing_entry", lambda *a, **k: None)
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("turn-order", limit_usd="1.00")
        agent = _Agent(db, order_id="turn-order")
        with pytest.raises(cost_budget.CostLimitExceeded) as exc:
            _run(agent)
        assert exc.value.reason == "unpriced"
        assert agent.spy.calls == 0
    finally:
        db.close()


def test_ambiguous_provider_failure_keeps_the_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_budget, "_pricing_entry", lambda *a, **k: _entry())
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("turn-order", limit_usd="1.00")
        agent = _Agent(db, order_id="turn-order")

        def _boom(api_kwargs):
            raise RuntimeError("provider exploded")

        agent._interruptible_api_call = _boom
        with pytest.raises(RuntimeError, match="provider exploded"):
            _run(agent)
        assert Decimal(db.get_cost_order("turn-order")["reserved_usd"]) > 0
        assert db.get_cost_order("turn-order")["spent_usd"] == "0.000000"
    finally:
        db.close()