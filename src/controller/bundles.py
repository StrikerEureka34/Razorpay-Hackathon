"""Bundled payouts: several invoices settled as one bank credit.

The payout's charge_ids length gives the exact bundle size, which turns an
open subset-sum into a fixed-k search over a pool of roughly 40 invoices.
Ties are returned in full so the adjudicator can break them.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from itertools import combinations
from math import comb

from . import config
from .ingest import Dataset
from .models import Candidate, Payout

_ZERO = Decimal("0")
# Largest drift one per-charge fee rounding can contribute to the payout net.
_ROUNDING_PER_CHARGE = Decimal("0.01")
# Above this many combinations, brute force costs more than building the
# meet-in-the-middle index. C(56, 6) is 32M and takes ~22s exhaustively;
# splitting the pool in half brings it under a second.
_BRUTE_FORCE_LIMIT = 250_000


def _brute_force(
    amounts: dict[str, Decimal], keys: list[str], target: Decimal, k: int, limit: int
) -> list[set[str]]:
    out: list[set[str]] = []
    for combo in combinations(keys, k):
        if sum((amounts[c] for c in combo), _ZERO) == target:
            out.append(set(combo))
            if len(out) >= limit:
                break
    return out


def subset_sums(
    amounts: dict[str, Decimal], target: Decimal, k: int, limit: int = 50
) -> list[set[str]]:
    """All k-element subsets summing exactly to target. Exact Decimal arithmetic.

    Switches to meet-in-the-middle once an exhaustive scan gets expensive.
    The split is by subset *size*, not by halving the pool: index every
    k2-subset by its sum, then scan the k1-subsets looking up `target - sum`
    and keeping the pairs that are disjoint. Cost falls from C(n, k) to
    C(n, k1) + C(n, k2) — at n=70, k=6 that is 110k sums, not 131M.
    """
    keys = sorted(amounts)
    n = len(keys)
    if k <= 0 or k > n:
        return []

    if comb(n, k) <= _BRUTE_FORCE_LIMIT:
        return _brute_force(amounts, keys, target, k, limit)

    k1 = k // 2
    k2 = k - k1

    by_sum: dict[Decimal, list[frozenset[str]]] = defaultdict(list)
    for combo in combinations(keys, k2):
        by_sum[sum((amounts[c] for c in combo), _ZERO)].append(frozenset(combo))

    # Each solution is rediscovered once per way of splitting it into k1 + k2,
    # so dedupe rather than emitting C(k, k1) copies of the same subset.
    seen: set[frozenset[str]] = set()
    out: list[set[str]] = []
    for combo in combinations(keys, k1):
        head = frozenset(combo)
        residual = target - sum((amounts[c] for c in combo), _ZERO)
        for tail in by_sum.get(residual, ()):
            if head & tail:
                continue
            union = head | tail
            if union in seen:
                continue
            seen.add(union)
            out.append(set(union))
            if len(out) >= limit:
                return out
    return out


def find_bundles(
    dataset: Dataset,
    residual_invoice_ids: set[str],
    payout_by_net: dict[Decimal, list[Payout]],
) -> list[Candidate]:
    invoice_by_id = {i.invoice_id: i for i in dataset.invoices}
    out: list[Candidate] = []

    # do_bundled writes the bank line as d(total - fee) — a single rounding —
    # while processor_payouts.csv aggregates k individually-rounded per-charge
    # nets (generate_dataset.py:536-546). The two drift by up to a paisa per
    # charge, so the bank -> payout chain cannot use exact equality here.
    # This bound is structural, not fitted: it is k roundings of half a paisa.
    all_payouts = [p for group in payout_by_net.values() for p in group]

    for line in dataset.bank_lines:
        amount = abs(line.amount_signed)
        for payout in all_payouts:
            k = len(payout.charge_ids)
            if k < 2 or k > config.BUNDLE_MAX_K:
                continue
            if abs(amount - payout.net) > _ROUNDING_PER_CHARGE * k:
                continue

            pool = {
                inv_id: invoice_by_id[inv_id].amount_gross
                for inv_id in residual_invoice_ids
                if invoice_by_id[inv_id].due_date <= payout.payout_date
            }
            solutions = subset_sums(pool, target=payout.gross, k=k)
            if not solutions:
                continue

            # A unique solution is near-certain; ties escalate.
            score = 0.97 if len(solutions) == 1 else 0.45
            for solution in solutions:
                out.append(Candidate(
                    invoice_ids=frozenset(solution),
                    bank_txn_ids=frozenset({line.txn_id}),
                    source="bundle",
                    features={
                        "payout_id": payout.payout_id,
                        "k": k,
                        "solutions_found": len(solutions),
                        "fee": str(payout.fees),
                    },
                    score=score,
                ))
    return out
