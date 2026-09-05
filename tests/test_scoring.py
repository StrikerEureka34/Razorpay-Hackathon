from datetime import date
from decimal import Decimal

from src.controller.models import BankLine, Evidence, Invoice
from src.controller.scoring import score_pair


def _inv(**kw):
    base = dict(
        invoice_id="INV-000123", customer_id="CUST-0001", customer_name="Acme Pvt Ltd",
        issue_date=date(2025, 3, 1), due_date=date(2025, 4, 1), currency="INR",
        amount_gross=Decimal("1000.00"), tax_amount=Decimal("0.00"),
        terms="NET30", status="issued",
    )
    base.update(kw)
    return Invoice(**base)


def _line(**kw):
    base = dict(
        txn_id="TXN-000001", value_date=date(2025, 4, 1), posted_date=date(2025, 4, 1),
        amount_signed=Decimal("1000.00"), currency="INR", description="x",
        running_balance=Decimal("0"),
    )
    base.update(kw)
    return BankLine(**base)


def _ev(**kw):
    base = dict(
        txn_id="TXN-000001", full_refs=(), prefix_refs=(), suffix_refs=(),
        split_suffix=None,
        payout_refs=(), counterparty="", currency_hint=None, rail="NEFT",
    )
    base.update(kw)
    return Evidence(**base)


def test_exact_reference_scores_near_the_top():
    score, features = score_pair(_ev(full_refs=("INV-000123",)), _inv(), _line())
    assert score > 0.8
    assert features["ref_exact"] is True


def test_prefix_reference_scores_lower_than_exact():
    exact, _ = score_pair(_ev(full_refs=("INV-000123",)), _inv(), _line())
    prefix, _ = score_pair(_ev(prefix_refs=("INV-0001",)), _inv(), _line())
    assert 0 < prefix < exact


def test_reference_to_a_different_invoice_scores_zero_reference_credit():
    _, features = score_pair(_ev(full_refs=("INV-000999",)), _inv(), _line())
    assert features["ref_exact"] is False


def test_matching_name_contributes_when_no_reference_survives():
    score, features = score_pair(_ev(counterparty="ACME"), _inv(), _line())
    assert features["name_similarity"] > 0.9
    assert score > 0.3


def test_date_distance_reduces_the_score():
    near, _ = score_pair(_ev(full_refs=("INV-000123",)), _inv(), _line())
    far, _ = score_pair(
        _ev(full_refs=("INV-000123",)), _inv(), _line(value_date=date(2025, 4, 20))
    )
    assert far < near


def test_score_is_always_within_unit_interval():
    score, _ = score_pair(
        _ev(full_refs=("INV-000123",), counterparty="ACME"), _inv(), _line()
    )
    assert 0.0 <= score <= 1.0
