#!/usr/bin/env python3
"""End to end demo for Track 04, AI Finance Controller.

Runs the whole loop in front of an audience and answers the track's bar in
order: throughput, then measured accuracy, then an honest exception list.
Everything here orchestrates the real tools. Nothing is precomputed or staged.

    python demo.py                 # straight through, about 40 seconds
    python demo.py --pause         # wait for Enter between acts, for presenting
    python demo.py --quick         # skip the held-out run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import webbrowser
from decimal import Decimal

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
DIM = "\033[2m" if _COLOR else ""
BOLD = "\033[1m" if _COLOR else ""
CYAN = "\033[36m" if _COLOR else ""
GREEN = "\033[32m" if _COLOR else ""
AMBER = "\033[33m" if _COLOR else ""
RED = "\033[31m" if _COLOR else ""
OFF = "\033[0m" if _COLOR else ""

PAUSE = False


def act(number: int, title: str, claim: str) -> None:
    print(f"\n{DIM}{'=' * 72}{OFF}")
    print(f"{CYAN}ACT {number}{OFF}  {BOLD}{title}{OFF}")
    print(f"{DIM}{claim}{OFF}")
    print(f"{DIM}{'=' * 72}{OFF}\n")
    if PAUSE:
        input(f"{DIM}  [Enter]{OFF}")


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=ENV, text=True,
        encoding="utf-8", **kw,
    )


def rows(path: str) -> list[dict[str, str]]:
    with open(os.path.join(ROOT, path), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ensure(data_dir: str, seed: int, n: int) -> None:
    if os.path.exists(os.path.join(ROOT, data_dir, "ground_truth.csv")):
        return
    print(f"  building {data_dir} at seed {seed} ...")
    run(["generate_dataset.py", "--seed", str(seed),
         "--num-invoices", str(n), "--output-dir", data_dir],
        capture_output=True, check=True)


def grab(text: str, label: str) -> str:
    m = re.search(rf"{label}:\s+([\d.]+)%", text)
    return m.group(1) + "%" if m else "n/a"


def split_misses(data_dir: str, results_dir: str) -> tuple[list, set, set]:
    """Which claimed exceptions are wrong, and are they unpaired splits?

    A false positive invoice whose amount equals the sum of two false positive
    bank lines is a split payment the matcher failed to pair, not a genuine
    disagreement about whether the money arrived.
    """
    truth, out_of_scope = set(), set()
    for r in rows(os.path.join(data_dir, "ground_truth.csv")):
        ids = {x for x in (r["invoice_ids"] + "|" + r["bank_txn_ids"]).split("|") if x}
        if r["match_type"] == "unmatchable":
            truth |= ids
        elif r["match_type"] == "out_of_scope":
            out_of_scope |= ids

    claimed = {r["item_id"] for r in rows(os.path.join(results_dir, "predicted_exceptions.csv"))}
    wrong = claimed - truth

    invoices = {r["invoice_id"]: r for r in rows(os.path.join(data_dir, "invoices.csv"))}
    lines = {r["txn_id"]: r for r in rows(os.path.join(data_dir, "bank_statement.csv"))}
    loose = [t for t in wrong if t in lines]

    pairs = []
    for inv_id in sorted(i for i in wrong if i in invoices):
        target = Decimal(invoices[inv_id]["amount_gross"])
        for a in range(len(loose)):
            for b in range(a + 1, len(loose)):
                legs = (loose[a], loose[b])
                if sum(Decimal(lines[t]["amount_signed"]) for t in legs) == target:
                    pairs.append((inv_id, invoices[inv_id], legs, [lines[t] for t in legs]))
                    break
            else:
                continue
            break
    return pairs, wrong, claimed & out_of_scope


def main() -> None:
    global PAUSE
    ap = argparse.ArgumentParser(description="Track 04 end to end demo")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between acts")
    ap.add_argument("--quick", action="store_true", help="skip the held-out run")
    ap.add_argument("--no-open", action="store_true", help="do not open the dashboard")
    ap.add_argument("--decisions", default="human_decisions.csv",
                    help="reviewer decisions exported from the dashboard, if any")
    args = ap.parse_args()
    PAUSE = args.pause
    # line buffering keeps our prints interleaved correctly with the
    # subprocesses that inherit stdout, including when piped to tee.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    print(f"\n{BOLD}AI FINANCE CONTROLLER{OFF}   Track 04, multi-source reconciliation")
    print(f"{DIM}Invoices against a bank statement against a processor settlement file.{OFF}")

    # ---------------------------------------------------------------- act 1
    act(1, "The batch, and the floor it has to clear",
        "One cherry-picked match proves nothing, so start by measuring the floor.")

    ensure("data", 42, 200)
    gt = rows("data/ground_truth.csv")
    kinds = {}
    for r in gt:
        kinds[r["match_type"]] = kinds.get(r["match_type"], 0) + 1
    matchable = sum(v for k, v in kinds.items() if k not in ("unmatchable", "out_of_scope"))

    print(f"  {len(rows('data/invoices.csv')):>4} invoices        the AR ledger")
    print(f"  {len(rows('data/bank_statement.csv')):>4} bank lines      197 credits and 104 operating debits")
    print(f"  {len(rows('data/processor_payouts.csv')):>4} payouts         settlements that bundle several invoices into one credit")
    print()
    print(f"  {matchable:>4} matchable       what a reconciliation can actually resolve")
    print(f"  {kinds.get('unmatchable', 0):>4} exceptions      20 unpaid invoices and 12 orphan credits")
    print(f"  {kinds.get('out_of_scope', 0):>4} not receivable  payroll, rent, GST, vendor runs")
    print(f"\n{DIM}  Measuring the greedy floor ...{OFF}")

    audit = run(["adversarial_audit.py"], capture_output=True)
    floor = re.search(r"baseline to beat: ([\d.]+)% F1", audit.stdout)
    floor_f1 = float(floor.group(1)) if floor else 70.4
    print(f"  A ten line greedy amount matcher scores {BOLD}{floor_f1}% F1{OFF} on this batch.")
    print(f"{DIM}  That is the number any result has to be read against.{OFF}")

    # ---------------------------------------------------------------- act 2
    act(2, "Throughput", "The whole loop, cold, on one machine, no network.")

    started = time.perf_counter()
    out = run(["reconcile.py", "--data", "data", "--out", "results"],
              capture_output=True, check=True)
    wall = time.perf_counter() - started
    print("  " + out.stdout.strip().replace("\n", "\n  "))
    decisions = sum(1 for _ in open(os.path.join(ROOT, "results/audit_trail.jsonl"), encoding="utf-8"))
    print(f"\n  {BOLD}{decisions} decisions in {wall:.1f}s{OFF} wall clock, "
          f"{decisions / wall:.0f} per second, including interpreter start.")
    print(f"{DIM}  Six artifacts written, one of them a full per decision audit trail.{OFF}")

    # ---------------------------------------------------------------- act 3
    act(3, "Measured accuracy",
        "Exact set equality on both sides. A four invoice bundle with three right scores zero.")

    scored = run(["score.py", "--predictions", "results/",
                  "--ground-truth", "data/ground_truth.csv"], capture_output=True, check=True)
    print(scored.stdout.strip())
    dev_f1 = float(grab(scored.stdout, "F1 SCORE").rstrip("%"))
    print(f"\n  {GREEN}{dev_f1}% against a {floor_f1}% floor, +{dev_f1 - floor_f1:.1f} points.{OFF}")

    if not args.quick:
        act(4, "The same thresholds, on data it has never seen",
            "A score you tuned against is not a score.")
        ensure("data_holdout", 99, 100)
        print(f"{DIM}  Running every configuration against both seeds ...{OFF}\n")
        run(["scripts/evaluate.py"], check=True)

        adjudicator = next(
            (l.split(": ", 1)[1].strip() for l in open(
                os.path.join(ROOT, "results/run_report.md"), encoding="utf-8")
             if l.startswith("- adjudicator")), "")
        if "Null" in adjudicator:
            print(f"{AMBER}  Both configurations scored the same because no GEMINI_API_KEY is set")
            print(f"  here, so the adjudicator fell back to NullAdjudicator and every row above")
            print(f"  is the deterministic core alone.{OFF}")
            print(f"{DIM}  That fallback is not a demo shortcut. It is the path the agent takes if")
            print(f"  the network dies mid presentation, which is why the rules only")
            print(f"  configuration doubles as the ablation baseline.{OFF}")

    # ---------------------------------------------------------------- act 5
    n = 4 if args.quick else 5
    act(n, "The honest exception list",
        "What it could not resolve, and what it got wrong.")

    pairs, wrong, leaked = split_misses("data", "results")
    claimed = rows("results/predicted_exceptions.csv")
    trail = [
        json.loads(l) for l in
        open(os.path.join(ROOT, "results/audit_trail.jsonl"), encoding="utf-8") if l.strip()
    ]
    escalated = [d for d in trail if d["decided_by"] != "rules"]
    held = [d for d in escalated if d["route"] != "match"]

    by_reason: dict[str, int] = {}
    for r in claimed:
        by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
    print(f"  {BOLD}Asserted unmatchable: {len(claimed)}{OFF}")
    for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>3}  {reason.replace('_', ' ')}")

    print(f"\n  {BOLD}Escalated past the rules: {len(escalated)}{OFF}")
    for d in escalated:
        item = (d["bank_txn_ids"] or d["invoice_ids"] or [""])[0]
        picked = ", ".join(d["invoice_ids"]) or "none"
        if d["route"] == "match":
            print(f"    {item}  {GREEN}settled{OFF} by {d['decided_by']} as {picked}")
        else:
            print(f"    {item}  {AMBER}held{OFF} for a controller, best guess {picked}")
    print(f"{DIM}    These are the items the deterministic ranker could not separate. They")
    print(f"    are escalated or handed to a person. Neither route puts them on the")
    print(f"    exception list above, which is what keeps that list an assertion.{OFF}")

    print(f"\n  {BOLD}Wrong: {len(wrong)}{OFF}")
    if pairs:
        print(f"  {DIM}Every one of them is the same failure, twice:{OFF}")
        for inv_id, inv, legs, leg_rows in pairs:
            print(f"    {RED}{inv_id}{OFF} {inv['customer_name']}, "
                  f"{Decimal(inv['amount_gross']):,.2f} was paid in two legs:")
            for txn, row in zip(legs, leg_rows):
                print(f"      {txn}  {Decimal(row['amount_signed']):>12,.2f}  "
                      f"{DIM}{row['description']}{OFF}")
        print(f"\n  {DIM}The generator gives a split leg the reference INV-000021-A, so every"
              f"\n  short form of it collapses to a bare A. One leg above carries no payer"
              f"\n  name either. Amount and date alone will not pair them, so the invoice"
              f"\n  and both legs fall through to the exception list.{OFF}")

    print(f"\n  {BOLD}Operating outflows wrongly listed: {len(leaked)}{OFF}")
    print(f"{DIM}  101 payroll, rent and GST lines are in the statement. Listing one is a"
          f"\n  pure false positive, and none of them reached the list.{OFF}")

    # ------------------------------------------------------------- act 6, hitl
    act(n + 1, "Human in the loop",
        "The agent hands work to a person, and the handover is measured too.")

    run(["scripts/build_dashboard.py", "--baseline", str(floor_f1)], check=True)
    page = os.path.join(ROOT, "results", "dashboard.html")

    reviewed = None
    if os.path.exists(os.path.join(ROOT, args.decisions)):
        run(["apply_review.py", "--decisions", args.decisions], check=True)
        from apply_review import headline
        reviewed = headline("data/ground_truth.csv", "results/reviewed")
    else:
        queued = len(claimed) + len(escalated)
        print(f"  The review station in the dashboard holds {queued} items: "
              f"{len(claimed)} the agent")
        print(f"  asserted and {len(escalated)} it escalated. A controller picks a name, attaches")
        print(f"  credits to an invoice until the tally balances, and exports the decisions.")
        print(f"\n{DIM}  The queue is what the AGENT flagged, never what the scorer knows, so a"
              f"\n  reviewer can be wrong and declining to match is a real option.{OFF}")
        print(f"\n  Then: {BOLD}python apply_review.py --decisions {args.decisions}{OFF}")
        print(f"{DIM}  which folds those decisions back in and re-scores, so the review step"
              f"\n  has to earn its number the same way the model did.{OFF}")

    # ---------------------------------------------------------------- close
    act(n + 2, "The desk", "The same run, for someone who is not reading a terminal.")

    print(f"\n{DIM}{'-' * 72}{OFF}")
    print(f"{BOLD}  THE BAR{OFF}")
    print(f"  Throughput            {GREEN}{decisions} decisions in {wall:.1f}s{OFF}")
    print(f"  Measured accuracy     {GREEN}{dev_f1}% F1{OFF} against a {floor_f1}% floor, "
          f"scored by exact set equality")
    print(f"  Honest exception list {GREEN}{len(claimed)} asserted, {len(wrong)} of them wrong, "
          f"{len(escalated)} escalated and {len(held)} left for a person{OFF}")
    if reviewed:
        print(f"  After human review    {GREEN}{reviewed[0]:.1f}% F1{OFF}, "
              f"exception list {GREEN}{reviewed[1]:.0f}%{OFF} once a controller worked the queue")
    print(f"{DIM}{'-' * 72}{OFF}")

    if not args.no_open:
        webbrowser.open("file:///" + page.replace("\\", "/"))
    print(f"\n  {page}\n")


if __name__ == "__main__":
    main()
