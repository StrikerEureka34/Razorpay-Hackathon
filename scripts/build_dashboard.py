"""Build results/dashboard.html: one self-contained page, no server, no network.

Reads the artifacts reconcile.py already wrote plus the source CSVs, scores them
with score.py's own Scorer, and injects the whole run into frontend/dashboard.html
as JSON. Open the output by double-clicking it.

    python scripts/build_dashboard.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from score import Scorer  # noqa: E402  reuse the real scorer, never reimplement it


def rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def rates(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
    }


def collect(data_dir: str, results_dir: str, baseline: float) -> dict:
    scorer = Scorer(
        gt_path=os.path.join(data_dir, "ground_truth.csv"),
        pred_match_path=os.path.join(results_dir, "predicted_matches.csv"),
        pred_exc_path=os.path.join(results_dir, "predicted_exceptions.csv"),
    )
    scored = scorer.score()

    scenarios, tot_tp, tot_fp, tot_fn, tot_gt = [], 0, 0, 0, 0
    for reason, r in sorted(scored.items()):
        if reason.startswith("_"):
            continue
        scenarios.append({
            "reason": reason, "gt": r["gt_count"],
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            **rates(r["tp"], r["fp"], r["fn"]),
        })
        tot_tp += r["tp"]
        tot_fp += r["fp"]
        tot_fn += r["fn"]
        tot_gt += r["gt_count"]
    tot_fp += scored.get("_false_positive", {}).get("fp", 0)
    scenarios.sort(key=lambda s: (-s["gt"], s["reason"]))

    exc = scored.get("_exceptions", {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0})

    decisions = [
        json.loads(line)
        for line in open(os.path.join(results_dir, "audit_trail.jsonl"), encoding="utf-8")
        if line.strip()
    ]

    invoices = {
        r["invoice_id"]: [
            r["customer_name"], r["amount_gross"], r["currency"],
            r["due_date"], r["status"],
        ]
        for r in rows(os.path.join(data_dir, "invoices.csv"))
    }
    bank = rows(os.path.join(data_dir, "bank_statement.csv"))
    lines = {
        r["txn_id"]: [r["value_date"], r["amount_signed"], r["description"]]
        for r in bank
    }

    # Everything the deterministic core could not settle on its own. With the
    # adjudicator live these come back settled; with --no-llm the same items are
    # held for a human. Showing the residue rather than only the review queue
    # keeps the section meaningful in both configurations.
    actions = {
        r["item_id"]: r["suggested_action"]
        for r in rows(os.path.join(results_dir, "review_queue.csv"))
    }
    residue = []
    for d in decisions:
        if d["decided_by"] == "rules":
            continue
        at_risk = sum(float(invoices[i][1]) for i in d["invoice_ids"] if i in invoices)
        if not at_risk:
            at_risk = sum(abs(float(lines[t][1])) for t in d["bank_txn_ids"] if t in lines)
        item = d["bank_txn_ids"][0] if d["bank_txn_ids"] else (d["invoice_ids"] or [""])[0]
        residue.append({
            "item_id": item,
            "amount": at_risk,
            "route": d["route"],
            "decided_by": d["decided_by"],
            "settled": d["route"] == "match",
            "why": d["rationale"],
            "action": actions.get(item, ""),
            "chosen": d["invoice_ids"],
            "candidates": d["candidates_considered"],
        })
    residue.sort(key=lambda r: -r["amount"])

    # The controller's work queue: what the AGENT flagged, never what the scorer
    # knows. Surfacing the items it got wrong would just be showing the answer
    # key, and a human who only confirms the answer key proves nothing.
    queue = []
    for d in decisions:
        if d["route"] not in ("review", "exception"):
            continue
        for inv_id in d["invoice_ids"]:
            meta = invoices.get(inv_id, ["", "0", "INR", "", ""])
            queue.append({
                "kind": "invoice", "id": inv_id, "amount": float(meta[1]),
                "label": meta[0], "sub": meta[2] + ", due " + meta[3],
                "reason": d["reason_code"] or d["route"],
                "agent": d["rationale"], "candidates": d["candidates_considered"],
            })
        for txn_id in d["bank_txn_ids"]:
            meta = lines.get(txn_id, ["", "0", ""])
            queue.append({
                "kind": "bank_txn", "id": txn_id, "amount": float(meta[1]),
                "label": meta[2], "sub": "value " + meta[0],
                "reason": d["reason_code"] or d["route"],
                "agent": d["rationale"], "candidates": d["candidates_considered"],
            })
    queue.sort(key=lambda q: -abs(q["amount"]))

    stats: dict[str, str] = {}
    report = os.path.join(results_dir, "run_report.md")
    if os.path.exists(report):
        for line in open(report, encoding="utf-8"):
            if line.startswith("- ") and ": " in line:
                k, _, v = line[2:].strip().partition(": ")
                stats[k] = v

    by_route: dict[str, int] = {}
    by_decider: dict[str, int] = {}
    for d in decisions:
        by_route[d["route"]] = by_route.get(d["route"], 0) + 1
        by_decider[d["decided_by"]] = by_decider.get(d["decided_by"], 0) + 1

    return {
        "meta": {
            "data_dir": data_dir,
            "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "invoices": len(invoices),
            "bank_lines": len(bank),
            "payouts": len(rows(os.path.join(data_dir, "processor_payouts.csv"))),
            "ground_truth_rows": len(scorer.gt),
            "elapsed": stats.get("elapsed_seconds", "n/a"),
            "adjudicator": stats.get("adjudicator", "n/a"),
            "cache_hits": stats.get("cache_hits", "0"),
        },
        "headline": {
            "match_rate": tot_tp / tot_gt if tot_gt else 0.0,
            "matched": tot_tp, "matchable": tot_gt,
            "baseline_f1": baseline / 100.0,
            **rates(tot_tp, tot_fp, tot_fn),
        },
        "exceptions": {"gt": exc["gt_count"], **exc, **rates(exc["tp"], exc["fp"], exc["fn"])},
        "routes": by_route,
        "deciders": by_decider,
        "scenarios": scenarios,
        "residue": residue,
        "queue": queue,
        "decisions": decisions,
        "invoices": invoices,
        "lines": lines,
        "cash": [[r["value_date"], float(r["running_balance"])] for r in bank],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the standalone run dashboard")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/dashboard.html")
    # ponytail: the floor is a constant from adversarial_audit.py, which runs at
    # import and cannot be called. Re-measure and pass --baseline if it moves.
    ap.add_argument("--baseline", type=float, default=70.4,
                    help="greedy floor F1 percent, from adversarial_audit.py")
    args = ap.parse_args()

    run = collect(args.data, args.results, args.baseline)
    template = open(os.path.join(ROOT, "frontend", "dashboard.html"), encoding="utf-8").read()
    payload = json.dumps(run, separators=(",", ":"), default=str)
    if "__RUN_DATA__" not in template:
        raise SystemExit("frontend/dashboard.html is missing the __RUN_DATA__ placeholder")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(template.replace("__RUN_DATA__", payload))

    size = os.path.getsize(args.out) / 1024
    print(f"{args.out}  ({size:.0f} KB, {len(run['decisions'])} decisions embedded)")
    print(f"match rate {run['headline']['match_rate']:.1%}  "
          f"F1 {run['headline']['f1']:.1%}  floor {args.baseline}%")


if __name__ == "__main__":
    main()
