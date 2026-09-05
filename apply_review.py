#!/usr/bin/env python3
"""Apply a controller's decisions from the review station, then re-score.

The dashboard exports human_decisions.csv. This folds those decisions back into
the agent's own output and reports what the human changed, so the value of the
review step is measured rather than asserted.

    python apply_review.py --decisions human_decisions.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from score import Scorer  # noqa: E402


def rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def headline(gt: str, out_dir: str) -> tuple[float, float, int]:
    """Matching F1, exception F1, and matched count, straight from the harness."""
    scorer = Scorer(
        gt_path=gt,
        pred_match_path=os.path.join(out_dir, "predicted_matches.csv"),
        pred_exc_path=os.path.join(out_dir, "predicted_exceptions.csv"),
    )
    scored = scorer.score()
    tp = fp = fn = 0
    for reason, r in scored.items():
        if reason.startswith("_"):
            continue
        tp += r["tp"]
        fp += r["fp"]
        fn += r["fn"]
    fp += scored.get("_false_positive", {}).get("fp", 0)

    def f1(t: int, p: int, n: int) -> float:
        prec = t / (t + p) if t + p else 0.0
        rec = t / (t + n) if t + n else 0.0
        return 2 * prec * rec / (prec + rec) * 100 if prec + rec else 0.0

    e = scored.get("_exceptions", {"tp": 0, "fp": 0, "fn": 0})
    return f1(tp, fp, fn), f1(e["tp"], e["fp"], e["fn"]), tp


def apply(results_dir: str, decisions_path: str, out_dir: str) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    for name in ("predicted_matches.csv", "predicted_exceptions.csv"):
        shutil.copy(os.path.join(results_dir, name), os.path.join(out_dir, name))

    matches = rows(os.path.join(out_dir, "predicted_matches.csv"))
    exceptions = rows(os.path.join(out_dir, "predicted_exceptions.csv"))
    settled = [d for d in rows(decisions_path) if d["action"] == "matched"]

    resolved = set()
    for n, d in enumerate(settled, start=1):
        targets = [t for t in d["matched_ids"].split("|") if t]
        if not targets:
            continue
        if d["item_type"] == "invoice":
            invoice_ids, txn_ids = [d["item_id"]], targets
        else:
            invoice_ids, txn_ids = targets, [d["item_id"]]
        matches.append({
            "match_id": f"HUMAN-{n:06d}",
            "invoice_ids": "|".join(sorted(invoice_ids)),
            "bank_txn_ids": "|".join(sorted(txn_ids)),
            "match_type": "human_review",
            "confidence": "1.000",
        })
        resolved.update(invoice_ids)
        resolved.update(txn_ids)

    kept = [e for e in exceptions if e["item_id"] not in resolved]

    with open(os.path.join(out_dir, "predicted_matches.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["match_id", "invoice_ids", "bank_txn_ids",
                                           "match_type", "confidence"])
        w.writeheader()
        w.writerows(matches)

    with open(os.path.join(out_dir, "predicted_exceptions.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_type", "item_id", "reason"])
        w.writeheader()
        w.writerows(kept)

    return settled


def main() -> None:
    ap = argparse.ArgumentParser(description="Fold review decisions back in and re-score")
    ap.add_argument("--decisions", default="human_decisions.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/reviewed")
    ap.add_argument("--ground-truth", default="data/ground_truth.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not os.path.exists(args.decisions):
        raise SystemExit(
            f"{args.decisions} not found. Work the review station in the dashboard, "
            "then use Export decisions as CSV."
        )

    before = headline(args.ground_truth, args.results)
    settled = apply(args.results, args.decisions, args.out)
    after = headline(args.ground_truth, args.out)

    print(f"\n  {len(settled)} items settled by a person\n")
    for d in settled:
        print(f"    {d['reviewer']:<32} {d['item_id']} -> {d['matched_ids'] or 'none'}"
              + (f"   {d['note']}" if d.get("note") else ""))

    print(f"\n  {'':<22}{'before':>10}{'after':>10}{'delta':>10}")
    for label, i in (("matching F1", 0), ("exception F1", 1), ("matched", 2)):
        b, a = before[i], after[i]
        suffix = "" if label == "matched" else "%"
        print(f"    {label:<20}{b:>9.1f}{suffix}{a:>9.1f}{suffix}{a - b:>+9.1f}{suffix}")
    print(f"\n  reviewed output in {args.out}/\n")


if __name__ == "__main__":
    main()
