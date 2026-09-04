#!/usr/bin/env python3
"""
Scoring Harness — AI Finance Controller (Track 04)
====================================================

Compares your reconciliation agent's output against ground_truth.csv
and reports precision / recall / F1 broken down by scenario type.

Expected agent output format (CSV):
    predicted_matches.csv:
        match_id, invoice_ids, bank_txn_ids, match_type, confidence

    predicted_exceptions.csv:
        item_type (invoice | bank_txn), item_id, reason

Usage:
    python score.py --predictions results/ --ground-truth data/ground_truth.csv
    python score.py --predictions results/ --ground-truth data/ground_truth.csv --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def parse_id_set(cell: str) -> frozenset[str]:
    """Parse a pipe-delimited ID string into a frozenset."""
    if pd.isna(cell) or str(cell).strip() == "":
        return frozenset()
    return frozenset(s.strip() for s in str(cell).split("|") if s.strip())


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# SCORING LOGIC
# ═══════════════════════════════════════════════════════════════════

class Scorer:
    def __init__(self, gt_path: str, pred_match_path: str, pred_exc_path: str | None):
        self.gt = pd.read_csv(gt_path)
        self.pred_matches = pd.read_csv(pred_match_path) if os.path.exists(pred_match_path) else pd.DataFrame()
        self.pred_exceptions = (
            pd.read_csv(pred_exc_path) if pred_exc_path and os.path.exists(pred_exc_path)
            else pd.DataFrame()
        )

        # Parse ground truth into structured form
        self.gt_entries = []
        for _, row in self.gt.iterrows():
            self.gt_entries.append({
                "match_type": row["match_type"],
                "invoice_ids": parse_id_set(row["invoice_ids"]),
                "bank_txn_ids": parse_id_set(row["bank_txn_ids"]),
                "reason_code": row["reason_code"],
                "expected_resolution": row["expected_resolution"],
            })

        # Parse predictions
        self.pred_entries = []
        if not self.pred_matches.empty:
            for _, row in self.pred_matches.iterrows():
                self.pred_entries.append({
                    "invoice_ids": parse_id_set(row.get("invoice_ids", "")),
                    "bank_txn_ids": parse_id_set(row.get("bank_txn_ids", "")),
                    "match_type": row.get("match_type", "unknown"),
                    "confidence": row.get("confidence", 1.0),
                })

        # Parse predicted exceptions
        self.pred_exc_items: set[str] = set()
        if not self.pred_exceptions.empty:
            for _, row in self.pred_exceptions.iterrows():
                self.pred_exc_items.add(row["item_id"])

    def score(self) -> dict:
        """
        Score predictions against ground truth.

        A predicted match is "correct" if both its invoice_ids AND bank_txn_ids
        exactly match a ground-truth entry (set equality).

        For unmatchable items, we check whether the agent's exception list
        covers the true exceptions.
        """
        results_by_reason: dict[str, dict] = defaultdict(lambda: {
            "tp": 0, "fp": 0, "fn": 0,
            "gt_count": 0, "pred_count": 0,
        })

        # ── matchable entries (non-exception ground truth) ──────
        matchable_gt = [e for e in self.gt_entries if e["match_type"] != "unmatchable"]
        unmatchable_gt = [e for e in self.gt_entries if e["match_type"] == "unmatchable"]

        # Build a lookup for ground-truth matchable entries
        # Key: (frozenset of invoice_ids, frozenset of bank_txn_ids)
        gt_lookup: dict[tuple, dict] = {}
        for e in matchable_gt:
            key = (e["invoice_ids"], e["bank_txn_ids"])
            gt_lookup[key] = e
            results_by_reason[e["reason_code"]]["gt_count"] += 1

        matched_gt_keys: set[tuple] = set()

        for pred in self.pred_entries:
            key = (pred["invoice_ids"], pred["bank_txn_ids"])
            if key in gt_lookup:
                gt_entry = gt_lookup[key]
                results_by_reason[gt_entry["reason_code"]]["tp"] += 1
                results_by_reason[gt_entry["reason_code"]]["pred_count"] += 1
                matched_gt_keys.add(key)
            else:
                # False positive — no matching ground truth
                results_by_reason["_false_positive"]["fp"] += 1
                results_by_reason["_false_positive"]["pred_count"] += 1

        # False negatives — ground truth entries not matched by any prediction
        for key, gt_entry in gt_lookup.items():
            if key not in matched_gt_keys:
                results_by_reason[gt_entry["reason_code"]]["fn"] += 1

        # ── exception scoring ───────────────────────────────────
        true_exc_inv = set()
        true_exc_bank = set()
        for e in unmatchable_gt:
            true_exc_inv |= e["invoice_ids"]
            true_exc_bank |= e["bank_txn_ids"]
        true_exc_all = true_exc_inv | true_exc_bank

        exc_tp = len(self.pred_exc_items & true_exc_all)
        exc_fp = len(self.pred_exc_items - true_exc_all)
        exc_fn = len(true_exc_all - self.pred_exc_items)

        results_by_reason["_exceptions"] = {
            "tp": exc_tp, "fp": exc_fp, "fn": exc_fn,
            "gt_count": len(true_exc_all),
            "pred_count": len(self.pred_exc_items),
        }

        return dict(results_by_reason)

    def report(self, verbose: bool = False) -> str:
        """Generate a human-readable scoring report."""
        results = self.score()

        lines = []
        lines.append("=" * 72)
        lines.append("  RECONCILIATION SCORING REPORT")
        lines.append("=" * 72)
        lines.append("")

        # ── per-scenario table ───────────────────────────────
        lines.append(f"{'Reason Code':<30s} {'GT':>4s} {'TP':>4s} {'FP':>4s} {'FN':>4s}  {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
        lines.append("─" * 72)

        total_tp = total_fp = total_fn = total_gt = 0

        for reason in sorted(results.keys()):
            if reason.startswith("_"):
                continue
            r = results[reason]
            p = safe_div(r["tp"], r["tp"] + r["fp"])
            rec = safe_div(r["tp"], r["tp"] + r["fn"])
            f1 = safe_div(2 * p * rec, p + rec)
            lines.append(
                f"{reason:<30s} {r['gt_count']:>4d} {r['tp']:>4d} {r['fp']:>4d} {r['fn']:>4d}"
                f"  {p:>6.1%} {rec:>6.1%} {f1:>6.1%}"
            )
            total_tp += r["tp"]
            total_fp += r["fp"]
            total_fn += r["fn"]
            total_gt += r["gt_count"]

        # False positives (no ground truth match)
        if "_false_positive" in results:
            fp_r = results["_false_positive"]
            total_fp += fp_r["fp"]
            lines.append(
                f"{'[false positives]':<30s} {'':>4s} {'':>4s} {fp_r['fp']:>4d} {'':>4s}"
                f"  {'':>6s} {'':>6s} {'':>6s}"
            )

        lines.append("─" * 72)
        overall_p = safe_div(total_tp, total_tp + total_fp)
        overall_r = safe_div(total_tp, total_tp + total_fn)
        overall_f1 = safe_div(2 * overall_p * overall_r, overall_p + overall_r)
        lines.append(
            f"{'OVERALL MATCHING':<30s} {total_gt:>4d} {total_tp:>4d} {total_fp:>4d} {total_fn:>4d}"
            f"  {overall_p:>6.1%} {overall_r:>6.1%} {overall_f1:>6.1%}"
        )
        lines.append("")

        # ── exception scoring ────────────────────────────────
        if "_exceptions" in results:
            exc = results["_exceptions"]
            ep = safe_div(exc["tp"], exc["tp"] + exc["fp"])
            er = safe_div(exc["tp"], exc["tp"] + exc["fn"])
            ef1 = safe_div(2 * ep * er, ep + er)
            lines.append(f"{'EXCEPTION DETECTION':<30s} {exc['gt_count']:>4d} {exc['tp']:>4d} {exc['fp']:>4d} {exc['fn']:>4d}"
                         f"  {ep:>6.1%} {er:>6.1%} {ef1:>6.1%}")
            lines.append("")

        # ── headline metric ──────────────────────────────────
        lines.append("=" * 72)
        match_rate = safe_div(total_tp, total_gt) if total_gt else 0
        lines.append(f"  MATCH RATE:  {match_rate:.1%}  ({total_tp}/{total_gt} ground-truth entries matched)")
        lines.append(f"  PRECISION:   {overall_p:.1%}")
        lines.append(f"  RECALL:      {overall_r:.1%}")
        lines.append(f"  F1 SCORE:    {overall_f1:.1%}")
        lines.append("=" * 72)

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Score reconciliation agent output")
    ap.add_argument(
        "--predictions", required=True,
        help="Directory containing predicted_matches.csv and optionally predicted_exceptions.csv",
    )
    ap.add_argument(
        "--ground-truth", required=True,
        help="Path to ground_truth.csv",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pred_match = os.path.join(args.predictions, "predicted_matches.csv")
    pred_exc = os.path.join(args.predictions, "predicted_exceptions.csv")

    if not os.path.exists(pred_match):
        print(f"ERROR: {pred_match} not found. Your agent must output predicted_matches.csv.")
        sys.exit(1)

    if not os.path.exists(args.ground_truth):
        print(f"ERROR: {args.ground_truth} not found. Run generate_dataset.py first.")
        sys.exit(1)

    scorer = Scorer(
        gt_path=args.ground_truth,
        pred_match_path=pred_match,
        pred_exc_path=pred_exc if os.path.exists(pred_exc) else None,
    )
    print(scorer.report(verbose=args.verbose))


if __name__ == "__main__":
    main()
