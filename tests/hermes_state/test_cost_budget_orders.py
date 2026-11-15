"""Issue #13 / Arbeitspaket C: an order-scoped, pre-call cost ceiling.

A sum computed after the fact is not a hard ceiling. These tests pin the
reserve-then-settle contract in state.db: worst-case money is taken out of the
order budget BEFORE the provider is called, settled to the real amount after,
and a reservation survives a restart instead of silently resetting the order.
"""

import pytest

from hermes_state import SessionDB


def test_reservation_blocks_second_call_and_settles_once(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        assert db.ensure_cost_order("order-a", limit_usd="1.00") == "created"
        assert db.ensure_cost_order("order-a", limit_usd="1.00") == "exists"
        # Two worst-case reservations fit; the third cannot, because held money
        # counts even though nothing has been charged yet.
        assert db.reserve_call_cost("order-a", "r1", "0.40") == "reserved"
        assert db.reserve_call_cost("order-a", "r2", "0.40") == "reserved"
        assert db.reserve_call_cost("order-a", "r3", "0.40") == "exceeds_limit"
        assert db.reserve_call_cost("order-a", "r1", "0.40") == "duplicate"
        assert db.get_cost_order("order-a")["reserved_usd"] == "0.800000"
        assert db.get_cost_order("order-a")["spent_usd"] == "0.000000"

        assert db.settle_call_cost("order-a", "r1", "0.10") == "settled"
        assert db.settle_call_cost("order-a", "r1", "0.10") == "duplicate"
        assert db.settle_call_cost("order-a", "r1", "9.00") == "conflict"
        order = db.get_cost_order("order-a")
        assert order["spent_usd"] == "0.100000"
        assert order["reserved_usd"] == "0.400000"
        assert db.settle_call_cost("order-a", "absent", "0.10") == "missing"
    finally:
        db.close()


def test_reservation_and_spend_survive_reopen_without_reset(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.ensure_cost_order("order-b", limit_usd="0.50")
    assert db.reserve_call_cost("order-b", "r1", "0.30") == "reserved"
    assert db.settle_call_cost("order-b", "r1", "0.30") == "settled"
    assert db.reserve_call_cost("order-b", "r2", "0.15") == "reserved"
    db.close()

    db = SessionDB(path)
    try:
        # The in-flight reservation is still held after a restart: a crash must
        # never hand the order a fresh budget.
        assert db.get_cost_order("order-b")["reserved_usd"] == "0.150000"
        assert db.reserve_call_cost("order-b", "r3", "0.10") == "exceeds_limit"
        assert db.reserve_call_cost("order-b", "r4", "0.05") == "reserved"
        assert db.get_cost_order("order-b")["reserved_usd"] == "0.200000"
    finally:
        db.close()


def test_overage_is_charged_in_full_and_stops_the_next_call(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("order-c", limit_usd="1.00")
        assert db.reserve_call_cost("order-c", "r1", "0.10") == "reserved"
        # The provider charged more than the reserve: the real amount is booked,
        # not clipped to what was reserved.
        assert db.settle_call_cost("order-c", "r1", "1.40") == "settled"
        assert db.get_cost_order("order-c")["spent_usd"] == "1.400000"
        assert db.get_cost_order("order-c")["overspent"] is True
        # Nothing left, not even a cent: the exceeded ceiling stays closed.
        assert db.reserve_call_cost("order-c", "r2", "0.01") == "exceeds_limit"
    finally:
        db.close()


def test_release_returns_money_but_never_below_spend(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("order-d", limit_usd="0.30")
        db.reserve_call_cost("order-d", "r1", "0.20")
        assert db.release_call_cost("order-d", "r1") == "released"
        assert db.release_call_cost("order-d", "r1") == "duplicate"
        order = db.get_cost_order("order-d")
        assert order["reserved_usd"] == "0.000000"
        assert order["spent_usd"] == "0.000000"
        assert db.reserve_call_cost("order-d", "r2", "0.30") == "reserved"
        assert db.settle_call_cost("order-d", "r2", "0.30") == "settled"
        assert db.reserve_call_cost("order-d", "r3", "0.01") == "exceeds_limit"
    finally:
        db.close()


@pytest.mark.parametrize("bad", ["", "-1", "nan", "inf", "abc"])
def test_invalid_limits_and_amounts_are_refused(tmp_path, bad):
    db = SessionDB(tmp_path / "state.db")
    try:
        with pytest.raises(ValueError):
            db.ensure_cost_order("order-e", limit_usd=bad)
        db.ensure_cost_order("order-e", limit_usd="1.00")
        with pytest.raises(ValueError):
            db.reserve_call_cost("order-e", "r1", bad)
        with pytest.raises(ValueError):
            db.settle_call_cost("order-e", "r1", bad)
    finally:
        db.close()


def test_limit_change_never_resets_the_spend(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.ensure_cost_order("order-f", limit_usd="1.00")
        db.reserve_call_cost("order-f", "r1", "0.50")
        db.settle_call_cost("order-f", "r1", "0.50")
        assert db.ensure_cost_order("order-f", limit_usd="0.40") == "exists"
        assert db.get_cost_order("order-f")["limit_usd"] == "1.000000"
        # Still the original ceiling: half of it is available, not four fifths.
        assert db.get_cost_order("order-f")["available_usd"] == "0.500000"
        assert db.reserve_call_cost("order-f", "r2", "0.50") == "reserved"
        assert db.reserve_call_cost("order-f", "r3", "0.01") == "exceeds_limit"
    finally:
        db.close()