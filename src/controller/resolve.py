"""Global resolution.

Order matters: duplicates, then splits, then bundles, then Hungarian
assignment over what remains. Bundles must precede assignment or Hungarian
will spuriously claim bundled invoices before the bundle search sees them.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import config
from .bundles import find_bundles
from .extract import extract_evidence, is_out_of_scope
from .indexes import (
    build_payout_by_net, candidates_for_line, find_settlements, find_splits,
)
from .ingest import Dataset
from .models import BankLine, Candidate, Evidence


@dataclass
class Resolution:
    accepted: list[Candidate] = field(default_factory=list)
    ambiguous: list[tuple[BankLine, list[Candidate]]] = field(default_factory=list)
    unmatched_lines: list[BankLine] = field(default_factory=list)
    # Operating outflows: never matched, never an exception, never scored.
    out_of_scope_lines: list[BankLine] = field(default_factory=list)
    unmatched_invoices: list[str] = field(default_factory=list)
    margins: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    # Candidates per open bank line, kept so routing can tell "no candidate"
    # (a genuine exception) from "lost the assignment" (a review item).
    candidates_by_line: dict[str, list[Candidate]] = field(default_factory=dict)


def find_duplicates(dataset: Dataset) -> list[Candidate]:
    """Identical (amount, description) within 2 days: the first posting is real.

    Transaction ids are shuffled after generation, so the old "lower txn_id is
    genuine" rule is now wrong more often than right. What still holds is the
    statement itself: a duplicate is a re-posting of a line that already
    appeared, so order by value date and fall back to position in the
    statement, which is the order a human reconciler would read it in.
    """
    order = {line.txn_id: n for n, line in enumerate(dataset.bank_lines)}
    groups: dict[tuple, list[BankLine]] = defaultdict(list)
    for line in dataset.bank_lines:
        groups[(line.amount_signed, line.description)].append(line)

    out: list[Candidate] = []
    for lines in groups.values():
        if len(lines) < 2:
            continue
        lines.sort(key=lambda b: (b.value_date, order[b.txn_id]))
        primary = lines[0]
        for extra in lines[1:]:
            delta = abs((extra.value_date - primary.value_date).days)
            if delta > config.DUPLICATE_WINDOW_DAYS:
                continue
            out.append(Candidate(
                invoice_ids=frozenset(),          # scored as a match, not an exception
                bank_txn_ids=frozenset({extra.txn_id}),
                source="duplicate",
                features={"primary_txn": primary.txn_id, "days_apart": delta},
                score=0.99,
            ))
    return out


def resolve(dataset: Dataset) -> Resolution:
    res = Resolution()
    res.evidence = {b.txn_id: extract_evidence(b) for b in dataset.bank_lines}
    payout_by_net = build_payout_by_net(dataset.payouts)

    # Take operating outflows off the table before anything else. Left in, they
    # would compete for invoices in the assignment (candidate generation works on
    # absolute amounts, so a debit can look like a credit) and then land on the
    # exception list, which is the single most expensive mistake here.
    res.out_of_scope_lines = [b for b in dataset.bank_lines if is_out_of_scope(b)]
    out_of_scope_ids = {b.txn_id for b in res.out_of_scope_lines}

    claimed_lines: set[str] = set()
    claimed_invoices: set[str] = set()

    def claim(candidate: Candidate) -> bool:
        if candidate.bank_txn_ids & claimed_lines:
            return False
        if candidate.invoice_ids & claimed_invoices:
            return False
        res.accepted.append(candidate)
        claimed_lines.update(candidate.bank_txn_ids)
        claimed_invoices.update(candidate.invoice_ids)
        return True

    # 1. duplicates
    for c in find_duplicates(dataset):
        claim(c)

    # 2. processor settlements, resolved through the payout file's invoice_refs.
    #    This runs before splits and bundles because it is the strongest signal
    #    available: the payout names its invoices outright.
    for c in sorted(find_settlements(dataset, res.evidence), key=lambda c: -c.score):
        claim(c)

    # 3. splits
    for c in sorted(find_splits(dataset, res.evidence), key=lambda c: -c.score):
        claim(c)

    # 3. bundles, over invoices that match no single bank line exactly
    line_amounts = {abs(b.amount_signed) for b in dataset.bank_lines}
    residual = {
        i.invoice_id for i in dataset.invoices
        if i.invoice_id not in claimed_invoices and i.amount_gross not in line_amounts
    }
    for c in sorted(find_bundles(dataset, residual, payout_by_net),
                    key=lambda c: -c.score):
        if c.score >= config.TAU_AUTO:
            claim(c)

    # 4. Hungarian assignment over the one-to-one remainder
    open_lines = [
        b for b in dataset.bank_lines
        if b.txn_id not in claimed_lines and b.txn_id not in out_of_scope_ids
    ]
    open_invoices = [i for i in dataset.invoices if i.invoice_id not in claimed_invoices]
    inv_index = {inv.invoice_id: n for n, inv in enumerate(open_invoices)}

    per_line: dict[str, list[Candidate]] = {}
    cost = np.zeros((len(open_lines), len(open_invoices)))
    for row, line in enumerate(open_lines):
        cands = candidates_for_line(
            line, res.evidence[line.txn_id], dataset, payout_by_net
        )
        cands = [c for c in cands if not (c.invoice_ids & claimed_invoices)]
        per_line[line.txn_id] = cands[: config.MAX_CANDIDATES_TO_LLM]
        for c in cands:
            inv_id = next(iter(c.invoice_ids))
            if inv_id in inv_index:
                cost[row, inv_index[inv_id]] = c.score

    if len(open_lines) and len(open_invoices):
        rows, cols = linear_sum_assignment(-cost)
    else:
        rows, cols = [], []

    for row, col in zip(rows, cols):
        line = open_lines[row]
        cands = per_line[line.txn_id]
        if not cands:
            continue
        assigned_id = open_invoices[col].invoice_id
        chosen = next((c for c in cands if assigned_id in c.invoice_ids), None)
        if chosen is None or chosen.score == 0.0:
            continue

        second = next((c.score for c in cands if c is not chosen), 0.0)
        margin = chosen.score - second
        res.margins[line.txn_id] = margin

        if margin >= config.TAU_AUTO:
            claim(chosen)
        else:
            res.ambiguous.append((line, cands))

    res.unmatched_lines = [
        b for b in dataset.bank_lines
        if b.txn_id not in claimed_lines
        and b.txn_id not in out_of_scope_ids
        and b.txn_id not in {l.txn_id for l, _ in res.ambiguous}
    ]
    res.unmatched_invoices = [
        i.invoice_id for i in dataset.invoices if i.invoice_id not in claimed_invoices
    ]
    # Only candidates whose invoice is still unclaimed are evidence that a
    # line is matchable; one pointing at an already-settled invoice is not.
    res.candidates_by_line = {
        txn_id: [c for c in cands if not (c.invoice_ids & claimed_invoices)]
        for txn_id, cands in per_line.items()
    }
    return res
