from decimal import Decimal

from src.controller.actions import disposition_for, journal_entries_for
from src.controller.ingest import load_dataset
from src.controller.models import Decision


def _decision(match_type, invoice_ids, txn_ids=frozenset({"TXN-000001"}), **kw):
    return Decision(
        match_id="MATCH-000001", invoice_ids=frozenset(invoice_ids),
        bank_txn_ids=txn_ids, match_type=match_type, confidence=0.9,
        route=kw.get("route", "match"), decided_by="rules", rationale="",
        reason_code=kw.get("reason_code", ""),
    )


def test_clean_match_debits_bank_and_credits_receivables():
    ds = load_dataset("data")
    inv = ds.invoices[0]
    lines = journal_entries_for(_decision("exact_amount", {inv.invoice_id}), ds)
    accounts = {l.account for l in lines}
    assert "Bank" in accounts and "Accounts Receivable" in accounts
    assert sum(l.debit for l in lines) == sum(l.credit for l in lines)


def test_every_entry_balances():
    ds = load_dataset("data")
    inv = ds.invoices[0]
    for match_type in ("exact_amount", "payout_chain", "fx_band", "split", "bundle"):
        lines = journal_entries_for(_decision(match_type, {inv.invoice_id}), ds)
        assert sum(l.debit for l in lines) == sum(l.credit for l in lines), match_type


def test_duplicate_proposes_no_entry():
    ds = load_dataset("data")
    assert journal_entries_for(_decision("duplicate", frozenset()), ds) == []


def test_processor_fee_entry_includes_fee_and_gst_accounts():
    ds = load_dataset("data")
    inv = ds.invoices[0]
    lines = journal_entries_for(_decision("payout_chain", {inv.invoice_id}), ds)
    accounts = {l.account for l in lines}
    assert "Processor Fees" in accounts
    assert "GST Input Credit" in accounts


def test_dispositions_are_specific_to_the_reason_code():
    assert "chase" in disposition_for(
        _decision("unmatchable", {"INV-000001"}, route="exception",
                  reason_code="unpaid_invoice")
    ).lower()
    assert disposition_for(
        _decision("unmatchable", frozenset(), route="exception",
                  reason_code="bank_charge")
    ) != ""


def test_every_real_match_in_a_full_run_balances():
    """Balance is an invariant of the whole run, not just of hand-built cases."""
    from src.controller.adjudicator.null import NullAdjudicator
    from src.controller.resolve import resolve
    from src.controller.routing import route

    ds = load_dataset("data")
    for d in route(ds, resolve(ds), NullAdjudicator()):
        lines = journal_entries_for(d, ds)
        if not lines:
            continue
        debit = sum((l.debit for l in lines), Decimal("0.00"))
        credit = sum((l.credit for l in lines), Decimal("0.00"))
        assert debit == credit, f"{d.match_id} ({d.match_type}) {debit} != {credit}"
        assert debit > 0, f"{d.match_id} posted a zero-value entry"
