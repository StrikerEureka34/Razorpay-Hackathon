"""Interim: resolve, dump the two scored CSVs, and call score.py.
Task 12 replaces this with the real emitter."""
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller.ingest import load_dataset       # noqa: E402
from src.controller.resolve import resolve           # noqa: E402

ds = load_dataset("data")
res = resolve(ds)
os.makedirs("results", exist_ok=True)

with open("results/predicted_matches.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["match_id", "invoice_ids", "bank_txn_ids", "match_type", "confidence"])
    for n, c in enumerate(res.accepted, start=1):
        w.writerow([
            f"MATCH-{n:06d}",
            "|".join(sorted(c.invoice_ids)),      # blank for duplicates, by design
            "|".join(sorted(c.bank_txn_ids)),
            c.source, f"{c.score:.3f}",
        ])

with open("results/predicted_exceptions.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["item_type", "item_id", "reason"])

print(f"accepted={len(res.accepted)} ambiguous={len(res.ambiguous)} "
      f"unmatched_lines={len(res.unmatched_lines)} "
      f"unmatched_invoices={len(res.unmatched_invoices)}\n")
subprocess.run([
    sys.executable, "score.py",
    "--predictions", "results/", "--ground-truth", "data/ground_truth.csv",
], check=False)
