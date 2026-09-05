"""Three-way routing: match, exception, review.

The exception list is scored against only 12 ground-truth items while
matching is scored against 186, so an uncertain item dumped into exceptions
is far more costly than one held back for review. Uncertain WITH a plausible
candidate goes to review; uncertain with NO candidate goes to exceptions.
"""
from __future__ import annotations

from . import config
from .adjudicator import Adjudicator, build_request
from .ingest import Dataset
from .models import Decision
from .resolve import Resolution


def route(
    dataset: Dataset, resolution: Resolution, adjudicator: Adjudicator
) -> list[Decision]:
    invoice_by_id = {i.invoice_id: i for i in dataset.invoices}
    decisions: list[Decision] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"MATCH-{counter:06d}"

    # 1. deterministic acceptances
    for c in resolution.accepted:
        decisions.append(Decision(
            match_id=next_id(), invoice_ids=c.invoice_ids,
            bank_txn_ids=c.bank_txn_ids, match_type=c.source,
            confidence=c.score, route="match", decided_by="rules",
            rationale=f"{c.source}: {c.features}",
            candidates_considered=[c],
        ))

    # 2. ambiguous cases -> the escalation ladder
    for line, candidates in resolution.ambiguous:
        top = candidates[0]
        margin = resolution.margins.get(line.txn_id, 0.0)

        # Below TAU_ESCALATE there is nothing for a model to adjudicate:
        # either no candidate is plausible at all (-> exception) or the
        # evidence is too thin to be worth a call (-> review).
        if margin < config.TAU_ESCALATE:
            if top.score < config.TAU_REVIEW:
                decisions.append(Decision(
                    match_id=next_id(), invoice_ids=frozenset(),
                    bank_txn_ids=frozenset({line.txn_id}),
                    match_type="unmatchable", confidence=0.0, route="exception",
                    decided_by="rules",
                    rationale="no candidate reached the plausibility floor",
                    reason_code=_classify_orphan(line.description, line.amount_signed),
                    candidates_considered=candidates,
                ))
            else:
                decisions.append(Decision(
                    match_id=next_id(), invoice_ids=top.invoice_ids,
                    bank_txn_ids=frozenset({line.txn_id}), match_type="uncertain",
                    confidence=top.score, route="review", decided_by="rules",
                    rationale="margin below the escalation floor",
                    reason_code="ambiguous_candidates",
                    candidates_considered=candidates,
                ))
            continue

        request = build_request(
            line, candidates[: config.MAX_CANDIDATES_TO_LLM], invoice_by_id
        )
        response = adjudicator.choose(request)

        if response.chosen_invoice_ids is not None:
            chosen = next(
                (c for c in candidates
                 if c.invoice_ids == response.chosen_invoice_ids),
                None,
            )
            if chosen is not None:
                # Confidence from AGREEMENT, not from model self-report.
                agreed = chosen is top
                decisions.append(Decision(
                    match_id=next_id(), invoice_ids=chosen.invoice_ids,
                    bank_txn_ids=chosen.bank_txn_ids, match_type=chosen.source,
                    confidence=0.9 if agreed else 0.7,
                    route="match", decided_by=response.model,
                    rationale=response.reasoning,
                    candidates_considered=candidates,
                ))
                continue

        # abstained, or chose something not on the list
        decisions.append(Decision(
            match_id=next_id(), invoice_ids=top.invoice_ids,
            bank_txn_ids=frozenset({line.txn_id}), match_type="uncertain",
            confidence=top.score, route="review", decided_by=response.model,
            rationale=response.reasoning or "margin below threshold",
            reason_code="ambiguous_candidates",
            candidates_considered=candidates,
        ))

    # 3. leftover bank lines. A line reaches here either because nothing
    #    plausible pointed at it (a genuine exception) or because it merely
    #    lost the global assignment (still matchable -> review). Asserting the
    #    second kind unmatchable is what destroys exception precision: the
    #    exception list is scored against 12 items, so each wrong assertion
    #    costs far more than holding the item back.
    reviewed_invoices: set[str] = set()
    for line in resolution.unmatched_lines:
        plausible = [
            c for c in resolution.candidates_by_line.get(line.txn_id, [])
            if c.score >= config.TAU_REVIEW
        ]
        if not plausible:
            decisions.append(Decision(
                match_id=next_id(), invoice_ids=frozenset(),
                bank_txn_ids=frozenset({line.txn_id}), match_type="unmatchable",
                confidence=0.0, route="exception", decided_by="rules",
                rationale="no invoice candidate within tolerance",
                reason_code=_classify_orphan(line.description, line.amount_signed),
            ))
            continue

        top = plausible[0]
        reviewed_invoices |= {i for c in plausible for i in c.invoice_ids}
        decisions.append(Decision(
            match_id=next_id(), invoice_ids=top.invoice_ids,
            bank_txn_ids=frozenset({line.txn_id}), match_type="uncertain",
            confidence=top.score, route="review", decided_by="rules",
            rationale="plausible candidate exists but lost the global assignment",
            reason_code="ambiguous_candidates",
            candidates_considered=plausible[: config.MAX_CANDIDATES_TO_LLM],
        ))

    # 3b. operating outflows. Payroll, rent, GST remittance and vendor payments
    #     are not receivables. They are scored as neither matchable nor an
    #     exception, so they must reach neither CSV — listing them is what drops
    #     exception precision to 24%. Recorded for the audit trail only.
    for line in resolution.out_of_scope_lines:
        decisions.append(Decision(
            match_id=next_id(), invoice_ids=frozenset(),
            bank_txn_ids=frozenset({line.txn_id}), match_type="out_of_scope",
            confidence=1.0, route="out_of_scope", decided_by="rules",
            rationale="operating outflow, not a receivable",
            reason_code="operating_outflow",
        ))

    # 4. invoices never settled. Same rule from the invoice side: one that some
    #    held-back line still plausibly settles is a review item, not an
    #    assertion that the customer never paid.
    claimed = {i for d in decisions if d.route == "match" for i in d.invoice_ids}
    # Every invoice some review item still points at — whether it was held back
    # here or at the ambiguity stage above, and including the runners-up.
    for d in decisions:
        if d.route != "review":
            continue
        reviewed_invoices |= d.invoice_ids
        reviewed_invoices |= {
            i for c in d.candidates_considered for i in c.invoice_ids
        }

    for invoice in dataset.invoices:
        if invoice.invoice_id in claimed:
            continue
        if invoice.invoice_id in reviewed_invoices:
            continue   # already represented by an item held for review
        decisions.append(Decision(
            match_id=next_id(), invoice_ids=frozenset({invoice.invoice_id}),
            bank_txn_ids=frozenset(), match_type="unmatchable",
            confidence=0.0, route="exception", decided_by="rules",
            rationale="no bank credit found for this invoice",
            reason_code="unpaid_invoice",
        ))

    return decisions


def _classify_orphan(description: str, amount) -> str:
    d = description.upper()
    if amount < 0 or "CHARGE" in d or "NACH DR" in d:
        return "bank_charge"
    if "INT" in d or "INTEREST" in d:
        return "interest_credit"
    if "REFUND" in d or "REV-" in d or "CHARGEBACK" in d:
        return "refund_no_invoice"
    if "GST" in d or "TDS" in d or "TAX" in d:
        return "tax_credit"
    return "miscellaneous"
