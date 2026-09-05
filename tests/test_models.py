from datetime import date
from decimal import Decimal

import pytest

from src.controller.models import BankLine, Candidate, Invoice, Payout


def test_invoice_is_frozen_and_uses_decimal():
    inv = Invoice(
        invoice_id="INV-000001", customer_id="CUST-0001", customer_name="Acme Pvt Ltd",
        issue_date=date(2025, 3, 22), due_date=date(2025, 4, 6), currency="INR",
        amount_gross=Decimal("25905.97"), tax_amount=Decimal("3951.76"),
        terms="NET15", status="issued",
    )
    assert isinstance(inv.amount_gross, Decimal)
    with pytest.raises(Exception):
        inv.amount_gross = Decimal("1")


def test_payout_charge_ids_is_a_tuple():
    p = Payout(
        payout_id="PAY-000005", payout_date=date(2025, 2, 10),
        gross=Decimal("1147.99"), fees=Decimal("35.65"), refunds=Decimal("0.00"),
        chargebacks=Decimal("0.00"), net=Decimal("1112.34"),
        charge_ids=("CHG-000005",),
    )
    assert p.charge_ids == ("CHG-000005",)
    assert len(p.charge_ids) == 1


def test_candidate_holds_frozen_id_sets():
    c = Candidate(
        invoice_ids=frozenset({"INV-000001"}), bank_txn_ids=frozenset({"TXN-000001"}),
        source="exact_amount", features={"date_delta": 0}, score=0.9,
    )
    assert c.invoice_ids == frozenset({"INV-000001"})


def test_bank_line_accepts_negative_amounts():
    bl = BankLine(
        txn_id="TXN-000001", value_date=date(2025, 1, 20), posted_date=date(2025, 1, 20),
        amount_signed=Decimal("-450.00"), currency="INR",
        description="BANK CHARGES JAN", running_balance=Decimal("100.00"),
    )
    assert bl.amount_signed < 0
