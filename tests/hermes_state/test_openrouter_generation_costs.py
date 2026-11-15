"""Issue Kevin-from-NoKi/hermes-kevin-core#13: one charge per OpenRouter generation."""

import pytest

from hermes_state import SessionDB


def _route(db, session, generation, model="vendor/model", estimate=None):
    db.update_token_counts(
        session, model=model, billing_provider="openrouter", input_tokens=100,
        output_tokens=10, api_call_count=1, estimated_cost_usd=estimate,
        cost_status="estimated" if estimate is not None else "unknown",
        generation_id=generation,
    )


def test_reconcile_only_once_after_route_change_and_reopen(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("mixed", source="cli", model="codex")
    db.update_token_counts("mixed", model="codex", billing_provider="openai-codex",
                           input_tokens=25, api_call_count=1, cost_status="included")
    _route(db, "mixed", "gen-a", estimate=0.02)
    _route(db, "mixed", "gen-b", estimate=0.01)
    _route(db, "mixed", "gen-a", estimate=0.02)
    assert db.get_session("mixed")["api_call_count"] == 3  # duplicate skipped
    assert db.reconcile_openrouter_generation_cost(
        "gen-a", {"data": {"id": "gen-a", "total_cost": 0.03, "model": "vendor/model"}}
    ) == "reconciled"
    db.close()

    db = SessionDB(path)
    try:
        assert db.reconcile_openrouter_generation_cost(
            "gen-a", {"data": {"id": "gen-a", "total_cost": 0.03}}
        ) == "duplicate"
        assert db.reconcile_openrouter_generation_cost(
            "gen-a", {"data": {"id": "gen-a", "total_cost": 0.04}}
        ) == "conflict"
        assert db.reconcile_openrouter_generation_cost(
            "gen-b", {"data": {"id": "gen-b"}}
        ) == "pending"
        assert db.reconcile_openrouter_generation_cost("gen-absent", None) == "missing"
        session = db.get_session("mixed")
        assert session["billing_provider"] == "openai-codex"
        assert session["estimated_cost_usd"] == pytest.approx(0.03)
        assert session["actual_cost_usd"] == pytest.approx(0.03)
        assert session["cost_status"] != "actual"  # second OR call still unverified
        rows = db._read_all(
            "SELECT billing_provider, model, actual_cost_usd, estimated_cost_usd, cost_status "
            "FROM session_model_usage WHERE session_id = 'mixed' ORDER BY rowid", (),
        )
        assert len(rows) == 2
        assert rows[0]["billing_provider"] == "openai-codex"
        assert rows[1]["billing_provider"] == "openrouter"
        assert rows[1]["actual_cost_usd"] == pytest.approx(0.03)
        assert rows[1]["estimated_cost_usd"] == pytest.approx(0.03)
        assert rows[1]["cost_status"] == "estimated"
    finally:
        db.close()


def test_generation_id_route_conflict_never_books_again(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("first", source="cli")
        db.create_session("second", source="cli")
        _route(db, "first", "gen-same", estimate=0.02)
        with pytest.raises(ValueError, match="generation.*different route"):
            _route(db, "second", "gen-same", estimate=0.05)
        assert db.get_session("second")["api_call_count"] == 0
        assert db.reconcile_openrouter_generation_cost(
            "gen-same", {"data": {"id": "wrong", "total_cost": 0.02}}
        ) == "conflict"
        assert db.reconcile_openrouter_generation_cost(
            "gen-same", {"data": {"id": "gen-same", "total_cost": -1}}
        ) == "pending"
        assert db.get_session("first")["actual_cost_usd"] is None
    finally:
        db.close()


def test_only_an_openrouter_response_stamps_a_generation_id():
    from types import SimpleNamespace

    from agent.turn_usage import _openrouter_generation_id

    response = SimpleNamespace(id="gen-xyz")
    assert _openrouter_generation_id(SimpleNamespace(provider="openrouter"), response) == "gen-xyz"
    # Anthropic/OpenAI ids are not OpenRouter generations and must not create
    # ledger rows nothing could ever settle.
    assert _openrouter_generation_id(SimpleNamespace(provider="anthropic"), response) is None
    assert _openrouter_generation_id(SimpleNamespace(provider="openrouter"), SimpleNamespace(id="")) is None


def test_interrupted_booking_leaves_no_half_charge(tmp_path, monkeypatch):
    """A crash between the ledger claim and the aggregates must roll both back."""
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("interrupted", source="cli")
        original = db._record_model_usage

        def _explode(*args, **kwargs):
            raise RuntimeError("simulated crash during booking")

        monkeypatch.setattr(db, "_record_model_usage", _explode)
        with pytest.raises(RuntimeError, match="simulated crash"):
            _route(db, "interrupted", "gen-crash", estimate=0.02)
        monkeypatch.setattr(db, "_record_model_usage", original)

        ledger = db._read_all(
            "SELECT generation_id FROM openrouter_generation_costs WHERE generation_id='gen-crash'", (),
        )
        assert ledger == [], "the generation id must not stay claimed without its token row"
        row = db._read_all(
            "SELECT api_call_count, estimated_cost_usd FROM sessions WHERE id='interrupted'", (),
        )[0]
        assert row["api_call_count"] == 0
        assert not row["estimated_cost_usd"]
        # Re-processing the same generation after the interruption books it once.
        _route(db, "interrupted", "gen-crash", estimate=0.02)
        _route(db, "interrupted", "gen-crash", estimate=0.02)
        assert db.get_session("interrupted")["api_call_count"] == 1
        assert db.reconcile_openrouter_generation_cost(
            "gen-crash", {"data": {"id": "gen-crash", "total_cost": 0.02}}
        ) == "reconciled"
    finally:
        db.close()
