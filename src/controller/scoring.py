"""Transparent weighted-sum scoring.

Deliberately not a learned model: every number in the audit trail has to be
explainable, and a fitted model would encode seed-42 idiosyncrasies.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from . import config
from .extract import name_similarity
from .models import BankLine, Evidence, Invoice

# Normalize by everything a surviving reference can earn on its own:
# reference + date + amount. Name similarity then adds on top, under the
# min(1.0, ...) cap below. Using the full weight sum as the denominator
# would cap a decisive reference-only match at 0.75 and needlessly narrow
# the ref-vs-no-ref margin that separates decoy pairs.
_MAX = config.W_REF_EXACT + config.W_DATE + config.W_AMOUNT


def invoice_amount_inr(invoice: Invoice) -> Decimal:
    """The invoice value in rupees.

    do_fx re-denominates an export invoice: amount_gross becomes the foreign
    figure and currency becomes USD/EUR/GBP. Everything downstream compares
    against bank lines, which are always INR, so convert at the reference rate.
    The dealt spread is absorbed by the caller's tolerance.
    """
    rate = config.FX_RATES.get(invoice.currency)
    return invoice.amount_gross * rate if rate else invoice.amount_gross


def score_pair(
    evidence: Evidence, invoice: Invoice, line: BankLine
) -> tuple[float, dict[str, Any]]:
    ref_exact = invoice.invoice_id in evidence.full_refs
    # A truncated reference constrains the invoice number without naming it:
    # "INV-00" from the head, "0101" from a split leg's ref_short tail.
    ref_prefix = any(
        invoice.invoice_id.startswith(p) for p in evidence.prefix_refs
    ) or any(
        invoice.invoice_id.endswith(s) for s in evidence.suffix_refs
    )

    name_sim = (
        name_similarity(evidence.counterparty, invoice.customer_name)
        if evidence.counterparty else 0.0
    )

    date_delta = abs((line.value_date - invoice.due_date).days)
    date_credit = max(0.0, 1.0 - date_delta / 30.0)

    # An export invoice is billed in foreign currency, so compare like with like:
    # convert it to the rupees the bank would actually receive.
    gross = invoice_amount_inr(invoice)
    amount_delta = (
        abs(abs(line.amount_signed) - gross) / gross if gross else Decimal("1")
    )
    amount_credit = max(0.0, 1.0 - float(amount_delta) * 10.0)

    total = 0.0
    if ref_exact:
        total += config.W_REF_EXACT
    elif ref_prefix:
        total += config.W_REF_PREFIX
    total += config.W_NAME * name_sim
    total += config.W_DATE * date_credit
    total += config.W_AMOUNT * amount_credit

    features = {
        "ref_exact": ref_exact,
        "ref_prefix": ref_prefix,
        "name_similarity": round(name_sim, 3),
        "date_delta": date_delta,
        "amount_delta_pct": float(amount_delta) * 100.0,
    }
    return min(1.0, total / _MAX), features
