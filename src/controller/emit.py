"""Write all six artifacts."""
from __future__ import annotations

import csv
import json
import os
from decimal import Decimal
from typing import Any

from .actions import disposition_for, journal_entries_for
from .ingest import Dataset
from .models import Decision

_ZERO = Decimal("0.00")


def _amount_at_risk(decision: Decision, dataset: Dataset) -> Decimal:
    invoice_by_id = {i.invoice_id: i for i in dataset.invoices}
    line_by_id = {b.txn_id: b for b in dataset.bank_lines}
    total = sum(
        (invoice_by_id[i].amount_gross for i in decision.invoice_ids if i in invoice_by_id),
        _ZERO,
    )
    if total == _ZERO:
        total = sum(
            (abs(line_by_id[t].amount_signed)
             for t in decision.bank_txn_ids if t in line_by_id),
            _ZERO,
        )
    return total


def emit_all(
    decisions: list[Decision], dataset: Dataset, out_dir: str, stats: dict[str, Any]
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    matches = [d for d in decisions if d.route == "match"]
    exceptions = [d for d in decisions if d.route == "exception"]
    reviews = [d for d in decisions if d.route == "review"]

    with open(os.path.join(out_dir, "predicted_matches.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "invoice_ids", "bank_txn_ids",
                    "match_type", "confidence"])
        for d in matches:
            w.writerow([
                d.match_id,
                "|".join(sorted(d.invoice_ids)),   # blank for duplicates, by design
                "|".join(sorted(d.bank_txn_ids)),
                d.match_type, f"{d.confidence:.3f}",
            ])

    with open(os.path.join(out_dir, "predicted_exceptions.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_type", "item_id", "reason"])
        for d in exceptions:
            for inv_id in sorted(d.invoice_ids):
                w.writerow(["invoice", inv_id, d.reason_code or d.rationale])
            for txn_id in sorted(d.bank_txn_ids):
                w.writerow(["bank_txn", txn_id, d.reason_code or d.rationale])

    ranked = sorted(reviews, key=lambda d: _amount_at_risk(d, dataset), reverse=True)
    with open(os.path.join(out_dir, "review_queue.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "amount_at_risk", "best_candidate",
                    "why_uncertain", "suggested_action"])
        for d in ranked:
            w.writerow([
                "|".join(sorted(d.bank_txn_ids)) or "|".join(sorted(d.invoice_ids)),
                _amount_at_risk(d, dataset),
                "|".join(sorted(d.invoice_ids)),
                d.rationale, disposition_for(d),
            ])

    with open(os.path.join(out_dir, "journal_entries.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "account", "debit", "credit", "narrative"])
        for d in matches:
            for jl in journal_entries_for(d, dataset):
                w.writerow([jl.match_id, jl.account, jl.debit, jl.credit, jl.narrative])

    with open(os.path.join(out_dir, "audit_trail.jsonl"), "w", encoding="utf-8") as fh:
        for d in decisions:
            fh.write(json.dumps({
                "match_id": d.match_id,
                "route": d.route,
                "match_type": d.match_type,
                "invoice_ids": sorted(d.invoice_ids),
                "bank_txn_ids": sorted(d.bank_txn_ids),
                "confidence": round(d.confidence, 3),
                "decided_by": d.decided_by,
                "rationale": d.rationale,
                "reason_code": d.reason_code,
                "candidates_considered": [
                    {"invoice_ids": sorted(c.invoice_ids), "score": round(c.score, 3),
                     "source": c.source, "features": c.features}
                    for c in d.candidates_considered
                ],
            }, default=str) + "\n")

    with open(os.path.join(out_dir, "run_report.md"), "w", encoding="utf-8") as fh:
        fh.write("# Reconciliation Run Report\n\n")
        fh.write(f"- Matched: **{len(matches)}**\n")
        fh.write(f"- Exceptions: **{len(exceptions)}**\n")
        fh.write(f"- Held for review: **{len(reviews)}**\n\n")
        fh.write("## Who decided\n\n| Decided by | Count |\n|---|---|\n")
        by_who: dict[str, int] = {}
        for d in decisions:
            by_who[d.decided_by] = by_who.get(d.decided_by, 0) + 1
        for who, count in sorted(by_who.items(), key=lambda kv: -kv[1]):
            fh.write(f"| {who} | {count} |\n")
        fh.write("\n## Run statistics\n\n")
        for key, value in stats.items():
            fh.write(f"- {key}: {value}\n")
