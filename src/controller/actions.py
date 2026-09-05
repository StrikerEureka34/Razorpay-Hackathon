"""Proposed accounting actions.

Dispositions are rule-derived, never model-generated: the LLM writes the
human-readable rationale, but the accounting itself is a lookup so it can
never be hallucinated. Nothing here executes anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .ingest import Dataset
from .models import Decision

_GST_RATE = Decimal("0.18")
_ZERO = Decimal("0.00")


@dataclass(frozen=True)
class JournalLine:
    match_id: str
    account: str
    debit: Decimal
    credit: Decimal
    narrative: str


def journal_entries_for(decision: Decision, dataset: Dataset) -> list[JournalLine]:
    if decision.match_type == "duplicate":
        return []   # flagged for reversal, not posted
    if decision.route != "match" or not decision.invoice_ids:
        return []

    invoice_by_id = {i.invoice_id: i for i in dataset.invoices}
    line_by_id = {b.txn_id: b for b in dataset.bank_lines}

    gross = sum(
        (invoice_by_id[i].amount_gross for i in decision.invoice_ids if i in invoice_by_id),
        _ZERO,
    )
    received = sum(
        (abs(line_by_id[t].amount_signed) for t in decision.bank_txn_ids if t in line_by_id),
        _ZERO,
    )
    if gross == _ZERO:
        return []

    mid, out = decision.match_id, []

    def add(account: str, debit: Decimal, credit: Decimal, narrative: str) -> None:
        out.append(JournalLine(mid, account, debit, credit, narrative))

    if decision.match_type in ("payout_chain", "bundle"):
        fee_total = gross - received
        fee_base = (fee_total / (Decimal("1") + _GST_RATE)).quantize(Decimal("0.01"))
        gst = fee_total - fee_base
        add("Bank", received, _ZERO, "Processor settlement received")
        add("Processor Fees", fee_base, _ZERO, "Payment gateway fee")
        add("GST Input Credit", gst, _ZERO, "GST on gateway fee")
        add("Accounts Receivable", _ZERO, gross, "Clear invoice(s) in full")

    elif decision.match_type == "fx_band":
        add("Bank", received, _ZERO, "Foreign currency receipt")
        diff = gross - received
        if diff > _ZERO:
            add("FX Loss", diff, _ZERO, "Exchange rate variance")
        elif diff < _ZERO:
            add("FX Gain", _ZERO, -diff, "Exchange rate variance")
        add("Accounts Receivable", _ZERO, gross, "Clear invoice")

    elif decision.match_type == "split":
        add("Bank", received, _ZERO, "Partial settlements received")
        add("Accounts Receivable", _ZERO, received, "Clear invoice to the extent settled")

    else:
        add("Bank", received, _ZERO, "Customer payment received")
        add("Accounts Receivable", _ZERO, received, "Clear invoice")

    return out


_DISPOSITIONS = {
    "unpaid_invoice": "Chase the customer — invoice is past due with no bank credit found.",
    "bank_charge": "Post to Bank Charges expense; no receivable is involved.",
    "interest_credit": "Book as Interest Income; no receivable is involved.",
    "refund_no_invoice": "Request a processor trace — refund or chargeback with no matching invoice.",
    "tax_credit": "Route to the tax team; likely a GST/TDS credit to be offset.",
    "miscellaneous": "Hold for controller review — unidentified credit.",
    "ambiguous_candidates": "Controller to confirm which invoice this settles.",
}


def disposition_for(decision: Decision) -> str:
    if decision.match_type == "duplicate":
        return "Flag for reversal — duplicate bank line, do not post."
    return _DISPOSITIONS.get(
        decision.reason_code, "Hold for controller review."
    )
