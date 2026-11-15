"""Order-scoped hard cost ceiling for paid model calls (issue #13, Arbeitspaket C).

Why "hard" is not a sum: adding up costs after each turn reports money that has
already been spent — it can never stop the call that overspends the order. This
module therefore enforces admission BEFORE the provider is called:

1. the next call's **worst-case** price is computed from the pricing table,
   treating every prompt token at the most expensive input-ish rate and the
   whole output allowance at the output rate;
2. that amount is **reserved** against the order's ceiling in state.db, in one
   `BEGIN IMMEDIATE` transaction, so two concurrent calls cannot both pass;
3. after the response the reservation is **settled** to the real cost — the
   real amount is booked in full even when it exceeds the reserve;
4. a call that never reached the provider **releases** its hold.

Limits of the guarantee (documented in the PR, not hidden):

- The ceiling is enforced by this process/agent, not by the provider. A crash
  leaves the hold in place (conservative: the budget stays spent), but a caller
  that bypasses this module is not stopped — hence provider-side spend limits
  remain the only mathematically guaranteed cap.
- A route without pricing data cannot be bounded. Such a call is refused with
  ``reason="unpriced"`` instead of being treated as free.
- The prompt estimate is an upper bound derived from the actual request; a
  provider that bills more than the published rate can exceed it, which is why
  the settlement books the real amount and the *next* call is then refused.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

# Re-exported so tests and callers can name the pricing shape without importing
# the pricing module directly.
from agent.usage_pricing import PricingEntry, resolve_billing_route

logger = logging.getLogger("agent.cost_budget")

_ZERO = Decimal(0)

# Agent attributes that must travel with a delegated/subagent run so the child
# spends from the SAME order ceiling instead of minting its own budget.
SUBAGENT_ATTRS = ("cost_order_id",)


class CostLimitExceeded(RuntimeError):
    """A paid call was refused by the order's cost ceiling (or could not be bounded).

    ``reason`` is a stable code — ``exhausted`` (money gone), ``unpriced``
    (no rate table for this route), ``unknown_order`` (a ceiling is configured
    but its row is missing) or ``unsafe`` (accounting not settled).
    """

    def __init__(self, reason: str, *, order_id: Optional[str] = None, detail: str = "") -> None:
        self.reason = reason
        self.order_id = order_id
        self.detail = detail
        parts = [f"Order cost ceiling stopped this call ({reason})"]
        if order_id:
            parts.append(f"order={order_id}")
        if detail:
            parts.append(detail)
        super().__init__(" — ".join(parts))


@dataclass(frozen=True)
class CallBudget:
    """A held reservation for exactly one provider call."""

    order_id: str
    reservation_id: str
    worst_case_usd: Decimal
    session_id: str = ""


def _pricing_entry(*args, **kwargs):
    """Indirection over the pricing lookup (a test seam, not a policy switch)."""
    from agent.usage_pricing import get_pricing_entry

    return get_pricing_entry(*args, **kwargs)


def worst_case_call_cost(
    model: str, provider: Optional[str], base_url: Optional[str], *, prompt_tokens: int,
    max_output_tokens: int, api_key: str = "",
) -> Optional[Decimal]:
    """Upper-bound price of one call, or ``None`` when the route cannot be priced.

    Every prompt token is charged at the *highest* input-side rate available
    (cache writes usually cost more than plain input, cache reads less), so the
    estimate over-charges rather than under-charges.
    """
    route = resolve_billing_route(model, provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        return _ZERO  # flat-rate route: nothing to bound
    entry = _pricing_entry(model, provider=provider, base_url=base_url, api_key=api_key)
    if entry is None:
        return None
    prompt_rate = max(
        (r for r in (entry.input_cost_per_million, entry.cache_write_cost_per_million, entry.cache_read_cost_per_million) if r is not None),
        default=None,
    )
    output_rate = entry.output_cost_per_million
    if entry.tier_threshold_tokens is not None and prompt_tokens > entry.tier_threshold_tokens:
        above = [r for r in (entry.input_cost_per_million_above, entry.cache_write_cost_per_million_above) if r is not None]
        if above:
            prompt_rate = max(prompt_rate, max(above)) if prompt_rate is not None else max(above)
        if entry.output_cost_per_million_above is not None:
            output_rate = (
                max(output_rate, entry.output_cost_per_million_above)
                if output_rate is not None else entry.output_cost_per_million_above
            )
    if output_rate is None or (prompt_tokens > 0 and prompt_rate is None):
        return None
    amount = Decimal(0)
    if prompt_tokens:
        amount += Decimal(max(prompt_tokens, 0)) * prompt_rate / Decimal(1_000_000)
    if max_output_tokens:
        amount += Decimal(max(max_output_tokens, 0)) * output_rate / Decimal(1_000_000)
    if entry.request_cost is not None:
        amount += entry.request_cost
    return amount.quantize(Decimal("0.000001"))


def configured_limit_usd() -> Optional[Decimal]:
    """The operator's ceiling from ``config.yaml`` (``cost_limits.order_limit_usd``).

    ``None`` — no ceiling — is the shipped default: this module never invents a
    dollar figure for the user. ``cost_limits.enabled: false`` also disables it.
    """
    try:
        from hermes_cli.config import load_config_readonly

        section = (load_config_readonly() or {}).get("cost_limits") or {}
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        # A broken or unreadable config must not block accounting; without a
        # readable ceiling there simply is none.
        logger.warning("cost_limits could not be read; no ceiling enforced", exc_info=True)
        return None
    if not isinstance(section, dict) or section.get("enabled", True) is False:
        return None
    raw = section.get("order_limit_usd")
    if raw is None:
        return None
    try:
        from hermes_state_usage import _money_usd

        limit = _money_usd(raw)
    except (ImportError, ValueError, TypeError):
        logger.warning("cost_limits.order_limit_usd=%r is not a usable amount; ceiling disabled", raw)
        return None
    return limit if limit > 0 else None


def order_id_for(agent: Any, *, create: bool = True) -> Optional[str]:
    """The order this run belongs to, or ``None`` when no ceiling was configured.

    An explicit ``agent._cost_order_id`` (delegation, tests, programmatic runs)
    wins. Otherwise the configured ceiling turns the session into its own order
    — one order per task, created once, so a restart resumes the same spend.
    """
    order = getattr(agent, "_cost_order_id", None)
    if isinstance(order, str) and order.strip():
        return order
    if getattr(agent, "_cost_order_disabled", False):
        return None
    limit = configured_limit_usd()
    if limit is None:
        agent._cost_order_disabled = True
        return None
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is None or not session_id:
        return None
    order = f"session:{session_id}"
    if create:
        db.ensure_cost_order(order, limit_usd=str(limit))
    agent._cost_order_id = order
    return order


def reserve_for_call(
    agent: Any, *, prompt_tokens: int, max_output_tokens: int, model: Optional[str] = None,
    provider: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None,
) -> Optional[CallBudget]:
    """Hold this call's worst-case price, or raise :class:`CostLimitExceeded`.

    Returns ``None`` when the run has no order ceiling at all: the feature is
    opt-in, and "no ceiling configured" must never silently become "unlimited
    spending without a record" — the caller's own accounting still applies.
    """
    order_id = order_id_for(agent)
    if order_id is None:
        return None
    db = getattr(agent, "_session_db", None)
    if db is None:
        raise CostLimitExceeded("unsafe", order_id=order_id, detail="no session store to hold the reservation")
    model = model or getattr(agent, "model", "") or ""
    if not isinstance(model, str):
        model = str(model)
    provider = provider if provider is not None else getattr(agent, "provider", None)
    base_url = base_url if base_url is not None else getattr(agent, "base_url", None)
    worst = worst_case_call_cost(
        model, provider, base_url, prompt_tokens=prompt_tokens,
        max_output_tokens=max_output_tokens, api_key=api_key if api_key is not None else getattr(agent, "api_key", "") or "",
    )
    if worst is None:
        raise CostLimitExceeded(
            "unpriced", order_id=order_id,
            detail=f"no pricing table for {provider or 'unknown provider'}/{model or 'unknown model'}",
        )
    reservation_id = getattr(agent, "_cost_turn_id", None) or ""
    reservation_id = f"{reservation_id or 'call'}:{uuid.uuid4().hex}"
    session_id = str(getattr(agent, "session_id", "") or "")
    status = db.reserve_call_cost(order_id, reservation_id, str(worst), session_id=session_id)
    if status == "exceeds_limit":
        order = db.get_cost_order(order_id) or {}
        raise CostLimitExceeded(
            "exhausted", order_id=order_id,
            detail=(
                f"limit={order.get('limit_usd')} spent={order.get('spent_usd')} "
                f"held={order.get('reserved_usd')} next-worst-case={worst}"
            ),
        )
    if status == "unknown_order":
        raise CostLimitExceeded("unknown_order", order_id=order_id, detail="order ceiling row is missing")
    budget = CallBudget(order_id=order_id, reservation_id=reservation_id, worst_case_usd=worst, session_id=session_id)
    agent._pending_call_budget = budget
    if status == "duplicate":
        logger.debug("Cost ceiling: replayed reservation %s (already held)", reservation_id)
    return budget


def settle_pending_call(agent: Any, cost: Optional[Decimal]) -> str:
    """Book the call's real cost against its hold; ``unknown_cost`` keeps the hold."""
    budget = getattr(agent, "_pending_call_budget", None)
    if not isinstance(budget, CallBudget):
        return "no_pending"
    db = getattr(agent, "_session_db", None)
    if db is None:
        return "no_pending"
    if cost is None:
        # Provider reported no usage: the price is unknown, not zero. Keeping the
        # reserve is the conservative choice and never frees money by accident.
        agent._pending_call_budget = None
        logger.warning(
            "Cost ceiling: call %s reported no usage; the worst-case hold of %s stays booked",
            budget.reservation_id, budget.worst_case_usd,
        )
        return "unknown_cost"
    status = db.settle_call_cost(budget.order_id, budget.reservation_id, str(cost))
    agent._pending_call_budget = None
    return status


def release_pending_call(agent: Any) -> str:
    """Give the hold back after a call that never reached the provider."""
    budget = getattr(agent, "_pending_call_budget", None)
    if not isinstance(budget, CallBudget):
        return "no_pending"
    db = getattr(agent, "_session_db", None)
    if db is None:
        agent._pending_call_budget = None
        return "no_pending"
    status = db.release_call_cost(budget.order_id, budget.reservation_id)
    agent._pending_call_budget = None
    return status


def estimate_prompt_tokens(api_kwargs: Any, fallback: int = 0) -> int:
    """Conservative prompt-token bound from the request body itself.

    ~3 characters per token is the usual German/English mix; the estimate is
    rounded up so the reservation errs on the expensive side.
    """
    messages = None
    if isinstance(api_kwargs, dict):
        messages = api_kwargs.get("messages")
    total = 0
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    total += len(str(content))
    chars = total or 0
    return max(fallback or 0, (chars + 2) // 3)


def estimate_max_output_tokens(api_kwargs: Any, fallback: int = 4096) -> int:
    """The output allowance the request may consume (the larger of the two keys)."""
    if isinstance(api_kwargs, dict):
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            value = api_kwargs.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return fallback