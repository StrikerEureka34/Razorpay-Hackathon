#!/usr/bin/env python3
"""AI Finance Controller — reconciliation agent CLI."""
from __future__ import annotations

import argparse
import json
import time

from dotenv import load_dotenv

from src.controller.adjudicator.gemini import build_adjudicator
from src.controller.emit import emit_all
from src.controller.ingest import load_dataset
from src.controller.resolve import resolve
from src.controller.routing import route


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Reconcile invoices against bank activity")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-llm", action="store_true",
                    help="rules only; also the offline fallback")
    ap.add_argument("--no-cache", action="store_true",
                    help="force live model calls instead of the committed cache")
    ap.add_argument("--explain", metavar="TXN_ID",
                    help="print the full decision trace for one transaction")
    args = ap.parse_args()

    started = time.time()
    dataset = load_dataset(args.data)
    resolution = resolve(dataset)
    adjudicator = build_adjudicator(
        use_llm=not args.no_llm, use_cache=not args.no_cache
    )
    decisions = route(dataset, resolution, adjudicator)

    stats = {
        "elapsed_seconds": round(time.time() - started, 2),
        "adjudicator": type(adjudicator).__name__,
        "cache_hits": getattr(adjudicator, "hits", 0),
        "cache_misses": getattr(adjudicator, "misses", 0),
        "escalations_to_tier_two": getattr(
            getattr(adjudicator, "inner", None), "escalations", 0
        ),
    }
    emit_all(decisions, dataset, args.out, stats)

    if args.explain:
        for d in decisions:
            if args.explain in d.bank_txn_ids or args.explain in d.invoice_ids:
                print(json.dumps({
                    "match_id": d.match_id, "route": d.route,
                    "match_type": d.match_type, "confidence": round(d.confidence, 3),
                    "decided_by": d.decided_by, "rationale": d.rationale,
                    "candidates": [
                        {"invoice_ids": sorted(c.invoice_ids),
                         "score": round(c.score, 3), "source": c.source,
                         "features": c.features}
                        for c in d.candidates_considered
                    ],
                }, indent=2, default=str))

    matched = sum(1 for d in decisions if d.route == "match")
    print(f"\nmatched={matched}  "
          f"exceptions={sum(1 for d in decisions if d.route == 'exception')}  "
          f"review={sum(1 for d in decisions if d.route == 'review')}  "
          f"({stats['elapsed_seconds']}s)")
    print(f"Artifacts written to {args.out}/")


if __name__ == "__main__":
    main()
