"""Issue #13: cost-limit refusal is terminal, not a retry/fallback trigger."""
from types import SimpleNamespace

from agent.cost_budget import CostLimitExceeded
from agent.turn_api_error import handle_api_error


def test_budget_refusal_stops_before_error_classifier_or_fallback():
    agent = SimpleNamespace(thinking_callback=None)
    verdict = handle_api_error(
        agent, api_error=CostLimitExceeded("exhausted", order_id="order"),
        _retry=None, thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
        system_message=None, active_system_prompt=None, conversation_history=[],
        approx_tokens=1, retry_count=0, max_retries=3, compression_attempts=0,
        max_compression_attempts=1, api_call_count=0, api_request_id="id",
        api_start_time=0, effective_task_id=None, turn_id=None,
    )
    assert verdict.action == "return"
    assert verdict.result["failed"] is True
    assert verdict.result["failure_retryable"] is False
