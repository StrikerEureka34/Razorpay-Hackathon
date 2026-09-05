"""Candidate generation. Four indexes, each with its own date window."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from . import config
from .ingest import Dataset
from .extract import name_similarity
from .models import BankLine, Candidate, Evidence, Invoice, Payout
from .scoring import score_pair

# A recovered split partner is identified by name alone, so the bar is high.
_SPLIT_PARTNER_MIN_NAME = 0.9
# Largest drift one per-charge fee rounding can contribute to the payout net.
_ROUNDING_PER_CHARGE = Decimal("0.01")


def build_payout_by_net(payouts: tuple[Payout, ...]) -> dict[Decimal, list[Payout]]:
    index: dict[Decimal, list[Payout]] = defaultdict(list)
    for p in payouts:
        index[p.net].append(p)
    return index


def payouts_by_reference(payouts: tuple[Payout, ...]) -> dict[str, Payout]:
    """Index payouts by the id fragment that appears in bank narration.

    Processor narrations are never truncated (generate_dataset.py), so the
    payout id always survives intact and is a reliable join key.
    """
    index: dict[str, Payout] = {}
    for payout in payouts:
        digits = payout.payout_id.split("-")[-1]
        index[digits] = payout
        index[digits.lstrip("0") or "0"] = payout
    return index


def _within(line: BankLine, invoice: Invoice, days: int) -> bool:
    return abs((line.value_date - invoice.due_date).days) <= days


def candidates_for_line(
    line: BankLine,
    evidence: Evidence,
    dataset: Dataset,
    payout_by_net: dict[Decimal, list[Payout]],
) -> list[Candidate]:
    """One-to-one candidates only. Splits and bundles are found separately."""
    amount = abs(line.amount_signed)
    found: dict[str, Candidate] = {}

    def offer(invoice: Invoice, source: str) -> None:
        score, features = score_pair(evidence, invoice, line)
        existing = found.get(invoice.invoice_id)
        if existing is None or score > existing.score:
            found[invoice.invoice_id] = Candidate(
                invoice_ids=frozenset({invoice.invoice_id}),
                bank_txn_ids=frozenset({line.txn_id}),
                source=source, features=features, score=score,
            )

    for invoice in dataset.invoices:
        # 1. exact amount
        if invoice.amount_gross == amount and _within(
            line, invoice, config.WINDOW_DAYS_DEFAULT
        ):
            offer(invoice, "exact_amount")

        # 2. FX. An export invoice is billed in foreign currency, so its
        #    amount_gross is the foreign figure and the bank receives
        #    foreign * dealt_rate. Convert through the reference rate and allow
        #    for the dealt spread. The invoice's own currency field gates this,
        #    which is far tighter than looking for a currency in the narration:
        #    truncation drops that hint on half the FX rows.
        elif invoice.currency != "INR" and invoice.amount_gross:
            rate = config.FX_RATES.get(invoice.currency)
            if rate:
                expected = invoice.amount_gross * rate
                delta = abs(amount - expected) / expected
                if delta <= config.FX_TOLERANCE and _within(
                    line, invoice, config.WINDOW_DAYS_DEFAULT
                ):
                    offer(invoice, "fx_band")

    # 3. payout chain: bank amount == payout.net, invoice amount == payout.gross
    for payout in payout_by_net.get(amount, []):
        if len(payout.charge_ids) != 1:
            continue  # bundles are handled by bundles.py
        for invoice in dataset.invoices:
            if invoice.amount_gross == payout.gross:
                offer(invoice, "payout_chain")

    # A reference that survived truncation is decisive even outside the window.
    for invoice in dataset.invoices:
        if invoice.invoice_id in evidence.full_refs:
            offer(invoice, "reference")

    return sorted(found.values(), key=lambda c: c.score, reverse=True)


def find_settlements(
    dataset: Dataset, evidence_by_txn: dict[str, Evidence]
) -> list[Candidate]:
    """Resolve processor settlements through the payout file.

    bank narration -> payout id -> invoice_refs, confirmed against `net`. This
    is the documented join and it replaces guessing: a single-charge payout is a
    1:1 match on the fee-deducted amount, and a bundle is named outright rather
    than reconstructed by subset-sum.
    """
    by_ref = payouts_by_reference(dataset.payouts)
    known_invoices = {i.invoice_id for i in dataset.invoices}
    out: list[Candidate] = []

    for line in dataset.bank_lines:
        amount = abs(line.amount_signed)
        for ref in evidence_by_txn[line.txn_id].payout_refs:
            payout = by_ref.get(ref)
            if payout is None:
                continue
            # Confirm against net. Per-charge fee rounding lets the aggregate
            # drift up to a paisa per charge, so this is not exact equality.
            if abs(amount - payout.net) > _ROUNDING_PER_CHARGE * len(payout.charge_ids):
                continue
            invoices = frozenset(payout.invoice_refs) & known_invoices
            if not invoices or len(invoices) != len(payout.invoice_refs):
                continue
            out.append(Candidate(
                invoice_ids=invoices,
                bank_txn_ids=frozenset({line.txn_id}),
                source="payout_chain" if len(invoices) == 1 else "bundle",
                features={
                    "payout_id": payout.payout_id,
                    "k": len(invoices),
                    "gross": str(payout.gross),
                    "net": str(payout.net),
                    "fee": str(payout.fees),
                    "resolved_by": "payout invoice_refs",
                },
                score=0.99,
            ))
            break
    return out


def find_splits(
    dataset: Dataset, evidence_by_txn: dict[str, Evidence]
) -> list[Candidate]:
    """One invoice settled by two bank credits.

    A split leg's reference is "INV-000053-A", so ref_num is "A" and every
    short variant the generator might pick degrades to A / INVA / REFA. Most
    legs therefore carry no invoice number at all, and the recoverable signal
    is the pair itself: two credits from the same payer whose amounts sum to
    the invoice exactly, both inside the wider split window.

    The exact Decimal sum is the strong constraint. The name gate is what stops
    a coincidental pair of amounts from inventing a split, and the second leg
    can land up to 18 days after the due date, so the window has to be wide.
    """
    out: list[Candidate] = []
    claimed_by_reference: set[str] = set()

    credits = [b for b in dataset.bank_lines if b.amount_signed > 0]

    for invoice in dataset.invoices:
        gross = invoice.amount_gross

        legs = [
            line for line in credits
            if abs(line.amount_signed) < gross
            and _within(line, invoice, config.WINDOW_DAYS_SPLIT)
            and (
                invoice.invoice_id in evidence_by_txn[line.txn_id].full_refs
                or name_similarity(
                    evidence_by_txn[line.txn_id].counterparty, invoice.customer_name
                ) >= _SPLIT_PARTNER_MIN_NAME
            )
        ]
        if len(legs) < 2:
            continue

        pairs = [
            (a, b)
            for i, a in enumerate(legs)
            for b in legs[i + 1:]
            if abs(a.amount_signed) + abs(b.amount_signed) == gross
        ]
        if not pairs:
            continue

        # A reference on either leg promotes the pair above a name-only one.
        def names_it(line: BankLine) -> bool:
            return invoice.invoice_id in evidence_by_txn[line.txn_id].full_refs

        pairs.sort(key=lambda p: (names_it(p[0]) or names_it(p[1])), reverse=True)
        a, b = pairs[0]
        referenced = names_it(a) or names_it(b)
        if referenced:
            claimed_by_reference.update({a.txn_id, b.txn_id})

        out.append(Candidate(
            invoice_ids=frozenset({invoice.invoice_id}),
            bank_txn_ids=frozenset({a.txn_id, b.txn_id}),
            source="split",
            features={
                "legs": 2,
                "sums_exactly": True,
                "alternatives": len(pairs),
                "resolved_by": "reference" if referenced else "name+exact_sum",
            },
            # A referenced pair is near-certain. A name-only pair is strong but
            # not decisive, and a tie among several pairs weakens it further.
            score=0.98 if referenced else (0.93 if len(pairs) == 1 else 0.6),
        ))

    return sorted(out, key=lambda c: -c.score)
