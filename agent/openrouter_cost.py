"""OpenRouter generation-cost lookup, never a second inference request.

The response's reported cost is preferred. When missing, the read-only generation
metadata endpoint can resolve an already persisted generation ID. A failed lookup
leaves the ledger pending; it never turns an unknown charge into zero.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def generation_metadata(response: Any, generation_id: str, api_key: str) -> dict | None:
    usage = _field(response, "usage")
    cost = _field(usage, "cost")
    if cost is not None:
        return {"data": {"id": generation_id, "total_cost": cost,
                         "is_byok": _field(usage, "is_byok")}}
    if not api_key:
        return None
    request = Request(
        "https://openrouter.ai/api/v1/generation?" + urlencode({"id": generation_id}),
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=3) as result:
            return json.load(result)
    except (OSError, ValueError, TypeError):
        logger.warning("OpenRouter generation metadata unavailable for %s; cost remains pending", generation_id)
        return None


def reconcile_response(agent: Any, response: Any, generation_id: str | None) -> tuple[str, float | None]:
    """Reconcile an already persisted generation; never claim success on a missed write."""
    if not generation_id:
        return "no_generation", None
    db = getattr(agent, "_session_db", None)
    if db is None or not db.flush_token_counts():
        return "pending", None
    metadata = generation_metadata(response, generation_id, getattr(agent, "api_key", "") or "")
    status = db.reconcile_openrouter_generation_cost(generation_id, metadata)
    if status not in ("reconciled", "duplicate"):
        return status, None
    data = metadata.get("data", metadata) if isinstance(metadata, dict) else {}
    try:
        amount = Decimal(str(data["total_cost"]))
        if amount.is_finite() and amount >= 0:
            return status, float(amount)
    except (KeyError, TypeError, ValueError, InvalidOperation):
        pass
    return "pending", None
