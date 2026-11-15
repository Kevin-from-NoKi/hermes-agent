"""Issue #13: OpenRouter actual charge from a response or mocked metadata GET."""
from types import SimpleNamespace

from agent import openrouter_cost
from agent.turn_usage import _reconcile_and_settle_openrouter
from hermes_state import SessionDB


def test_inline_actual_charge_reconciles_once_in_state(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.update_token_counts("sess", model="vendor/model", billing_provider="openrouter",
                               input_tokens=10, output_tokens=2, api_call_count=1,
                               generation_id="gen-inline", estimated_cost_usd=0.02,
                               cost_status="estimated")
        response = SimpleNamespace(usage=SimpleNamespace(cost=0.03, is_byok=False))
        agent = SimpleNamespace(_session_db=db, api_key="")
        monkeypatch.setattr(openrouter_cost, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
        assert openrouter_cost.reconcile_response(agent, response, "gen-inline") == ("reconciled", 0.03)
        assert openrouter_cost.reconcile_response(agent, response, "gen-inline") == ("duplicate", 0.03)
        assert db.get_session("sess")["actual_cost_usd"] == 0.03
    finally:
        db.close()


def test_missing_inline_cost_fetches_generation_metadata_without_inference(monkeypatch):
    class Reply:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self, *_): return b'{"data":{"id":"gen-1","total_cost":0.05}}'

    calls = []
    def fake_open(request, timeout):
        calls.append((request.full_url, request.get_method(), timeout))
        return Reply()
    monkeypatch.setattr(openrouter_cost, "urlopen", fake_open)
    metadata = openrouter_cost.generation_metadata(SimpleNamespace(usage=SimpleNamespace()), "gen-1", "test-key")
    assert metadata["data"]["total_cost"] == 0.05
    assert calls == [("https://openrouter.ai/api/v1/generation?id=gen-1", "GET", 3)]


def test_no_key_and_no_cost_remain_pending(monkeypatch):
    monkeypatch.setattr(openrouter_cost, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert openrouter_cost.generation_metadata(SimpleNamespace(usage=None), "gen-1", "") is None


def test_turn_settles_order_from_actual_not_estimate(tmp_path):
    from agent.cost_budget import CallBudget
    from decimal import Decimal

    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("o", limit_usd="1")
        db.reserve_call_cost("o", "r", "0.40")
        db.update_token_counts("sess", model="vendor/model", billing_provider="openrouter",
                               input_tokens=10, output_tokens=2, api_call_count=1,
                               generation_id="gen-turn", estimated_cost_usd=0.10,
                               cost_status="estimated")
        agent = SimpleNamespace(_session_db=db, provider="openrouter", api_key="",
                                _pending_call_budget=CallBudget("o", "r", Decimal("0.40")))
        response = SimpleNamespace(usage=SimpleNamespace(cost=0.25))
        _reconcile_and_settle_openrouter(agent, response, "gen-turn", 0.10)
        assert db.get_cost_order("o")["spent_usd"] == "0.250000"
        assert db.get_session("sess")["actual_cost_usd"] == 0.25
        db.reserve_call_cost("o", "replay", "0.40")
        agent._pending_call_budget = CallBudget("o", "replay", Decimal("0.40"))
        _reconcile_and_settle_openrouter(agent, response, "gen-turn", 0.10)
        assert db.get_cost_order("o")["spent_usd"] == "0.250000"
        assert db.get_cost_order("o")["reserved_usd"] == "0.000000"
    finally:
        db.close()


def test_missing_charge_keeps_hold_and_estimate_distinct(tmp_path):
    from agent.cost_budget import CallBudget
    from decimal import Decimal

    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("o", limit_usd="1")
        db.reserve_call_cost("o", "r", "0.40")
        db.update_token_counts("sess", model="vendor/model", billing_provider="openrouter",
                               input_tokens=10, output_tokens=2, api_call_count=1,
                               generation_id="gen-pending", estimated_cost_usd=0.10,
                               cost_status="estimated")
        agent = SimpleNamespace(_session_db=db, provider="openrouter", api_key="",
                                _pending_call_budget=CallBudget("o", "r", Decimal("0.40")))
        _reconcile_and_settle_openrouter(agent, SimpleNamespace(usage=None), "gen-pending", 0.10)
        assert db.get_cost_order("o")["reserved_usd"] == "0.400000"
        assert db.get_cost_order("o")["spent_usd"] == "0.000000"
        assert db.get_session("sess")["actual_cost_usd"] is None
    finally:
        db.close()
