"""Issue Kevin-from-NoKi/hermes-kevin-core#13: attribution across provider changes."""

from agent.insights import InsightsEngine
from hermes_state import SessionDB


def test_same_model_on_two_providers_stays_separate_in_insights(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("switch", source="cli", model="shared-model")
        for provider, tokens in (("openai-codex", 5), ("openrouter", 7), ("anthropic", 11)):
            db.update_token_counts(
                "switch", model="shared-model", billing_provider=provider,
                billing_mode="subscription_included" if provider == "openai-codex" else None,
                input_tokens=tokens, api_call_count=1,
            )
        rows = db._read_all(
            "SELECT billing_provider, input_tokens, billing_mode FROM session_model_usage "
            "WHERE session_id = 'switch' ORDER BY rowid", (),
        )
        assert [(r["billing_provider"], r["input_tokens"]) for r in rows] == [
            ("openai-codex", 5), ("openrouter", 7), ("anthropic", 11),
        ]
        assert rows[1]["billing_mode"] != "subscription_included"
        recent = db.get_recent_session_model_route("switch")
        session = db.get_session("switch")
        assert recent is not None and recent["billing_provider"] == "anthropic"
        assert session is not None and session["billing_provider"] == "openai-codex"
        models = InsightsEngine(db).generate(days=30)["models"]
        attributed = {(r["model"], r.get("provider")): r["input_tokens"] for r in models}
        assert attributed == {
            ("shared-model", "openai-codex"): 5,
            ("shared-model", "openrouter"): 7,
            ("shared-model", "anthropic"): 11,
        }
    finally:
        db.close()
