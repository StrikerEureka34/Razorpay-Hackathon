"""Adversarial audit: is this dataset a benchmark, or just self-consistent?

audit_data.py asserts the generator's own invariants (splits sum, decoys share an
amount). Those can never fail, so they carry no information. This asks the only
question that matters: how much headroom is left above a dumb baseline, and does
any column leak the label.

    python adversarial_audit.py
"""
import re
import pandas as pd
from collections import Counter

gt = pd.read_csv("data/ground_truth.csv").fillna("")
inv = pd.read_csv("data/invoices.csv")
bank = pd.read_csv("data/bank_statement.csv")
pay = pd.read_csv("data/processor_payouts.csv")

ids = lambda c: frozenset(s for s in str(c).split("|") if s)
gt["I"], gt["B"] = gt.invoice_ids.map(ids), gt.bank_txn_ids.map(ids)
truth = {(r.I, r.B) for r in gt.itertuples()
         if r.match_type not in ("unmatchable", "out_of_scope")}
oos_txn = {t for r in gt.itertuples() if r.match_type == "out_of_scope" for t in r.B}
exc_inv = {i for r in gt.itertuples() if r.match_type == "unmatchable" for i in r.I}
exc_txn = {t for r in gt.itertuples() if r.match_type == "unmatchable" for t in r.B}

norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
cust = dict(zip(inv.invoice_id, inv.customer_name))
amts = dict(zip(inv.invoice_id, inv.amount_gross))
issued = dict(zip(inv.invoice_id, pd.to_datetime(inv.issue_date)))


def f1(pred, gold):
    tp = len(pred & gold)
    p, r = tp / max(len(pred), 1), tp / max(len(gold), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def show(label, pred, gold):
    p, r, s = f1(pred, gold)
    print(f"  {label:<42} P={p:6.1%} R={r:6.1%} F1={s:6.1%}")
    return s


def greedy(band, name_gate=False, window=None):
    """Laziest possible solver: each credit takes the nearest unclaimed invoice
    whose amount is within `band` (one-sided, the bank is only ever short by a fee)."""
    free, pred = set(inv.invoice_id), set()
    for b in bank.sort_values("value_date").itertuples():
        if b.amount_signed <= 0:
            continue
        desc, vd, best = norm(b.description), pd.Timestamp(b.value_date), None
        # sorted(), not the set: string hashing is randomised per process, so
        # iterating `free` directly breaks amount ties differently every run and
        # the baseline swings by more than a point on identical data.
        for i in sorted(free):
            if not -0.0005 <= (amts[i] - b.amount_signed) / amts[i] <= band:
                continue
            nm = norm(cust[i])
            if name_gate and not (len(nm) >= 8 and nm[:8] in desc):
                continue
            gap = (vd - issued[i]).days
            if window and not 0 <= gap <= window:
                continue
            if best is None or abs(gap) < best[0]:
                best = (abs(gap), i)
        if best:
            free.discard(best[1])
            pred.add((frozenset({best[1]}), frozenset({b.txn_id})))
    return pred


print("=" * 74)
print("  ADVERSARIAL AUDIT: headroom above naive, and label leakage")
print("=" * 74)

print(f"\n--- 1. NAIVE BASELINES (vs {len(truth)} matchable ground-truth entries) ---")
scores = [
    show("exact amount, greedy nearest date", greedy(0.0005), truth),
    show("exact-or-fee band 3.5%", greedy(0.035), truth),
    show("fee band + payment within 75d of issue", greedy(0.035, window=75), truth),
    show("fee band + payer name in narration + 75d", greedy(0.035, True, 75), truth),
]
print(f"\n  >>> baseline to beat: {max(scores):.1%} F1. Report your agent as a delta on this,")
print(f"      not as a bare number. Headroom = {1 - max(scores):.1%}.")

print("\n--- 2. LABEL LEAKS (a leak = free score, no reasoning) ---")
show("invoices: status != 'issued' -> exception", set(inv[inv.status != "issued"].invoice_id), exc_inv)
kw = r"INT CREDIT|INTEREST|REFUND|REV-|CHARGEBACK|BANK CHARGES|TDS|MISCELLANEOUS|SERVICE TAX|NACH"
show("bank: keyword regex -> orphan line", set(bank[bank.description.str.contains(kw)].txn_id), exc_txn)
ordinal = lambda t: int(t.split("-")[1])
blocks = pd.DataFrame(
    [(r.reason_code, ordinal(t)) for r in gt.itertuples() for t in r.B]
, columns=["reason", "n"]).groupby("reason").n.agg(["min", "max", "count"])
overlap = sum(
    any(a["min"] <= b["min"] <= a["max"] for j, b in blocks.iterrows() if i != j)
    for i, a in blocks.iterrows()
)
print(f"  txn_id ordinal blocks per scenario: {len(blocks)} scenarios, {overlap} overlapping "
      f"-> ids {'ARE' if overlap < len(blocks) / 2 else 'are not'} a scenario label")

print("\n--- 3. IS THE THIRD SOURCE ACTUALLY A SOURCE? ---")
# The only test that matters: can a bank line be resolved to its invoices by
# going *through* the payout file? bank narration -> payout_id -> invoice_refs.
settled = gt[gt.reason_code.str.contains("processor_fee|bundled", regex=True)]
amt_of = dict(zip(bank.txn_id, bank.amount_signed))
desc_of = dict(zip(bank.txn_id, bank.description))
bridged = 0
for r in settled.itertuples():
    t = next(iter(r.B))
    hit = pay[pay.payout_id.map(lambda q: q.split("-")[1] in norm(desc_of[t]))]
    hit = hit[(hit.net - amt_of[t]).abs() < 0.02]
    if len(hit) == 1 and set(str(hit.iloc[0].invoice_refs).split("|")) == set(r.I):
        bridged += 1
print(f"  {len(pay)} payouts, {len(settled)} settled bank lines")
print(f"  resolvable via narration -> payout.net -> invoice_refs: {bridged}/{len(settled)}")
print(f"  invoice_refs that are real invoices: "
      f"{set(i for c in pay.invoice_refs for i in str(c).split('|')) <= set(inv.invoice_id)}")

print("\n--- 4. DOES THIS LOOK LIKE A BANK STATEMENT? ---")
print(f"  credits={(bank.amount_signed > 0).sum()}  debits={(bank.amount_signed < 0).sum()}"
      f"  (a real operating account is roughly half debits)")
print(f"  balance: open {bank.running_balance.iloc[0]:,.0f} -> close {bank.running_balance.iloc[-1]:,.0f}"
      f"  min {bank.running_balance.min():,.0f}  (it should move both ways)")
naive_exc = set(bank.txn_id) - {t for _, b in truth for t in b}
p_, r_, _ = f1(naive_exc, exc_txn)
print(f"  'every unmatched line is an exception' -> P={p_:.1%} ({len(naive_exc & oos_txn)} payroll/rent lines wrongly flagged)")
for k in ["BANK CHARGES", "NACH DR"]:
    s = bank[bank.description.str.contains(k, regex=False)]
    if len(s) and (s.amount_signed > 0).any():
        print(f"  SIGN ERROR: '{k}' narration carries a CREDIT: {list(s.amount_signed)}")
bad = bank[bank.description.str.match(r"DEPOSIT \d+\.\d\d$")]
for r in bad.itertuples():
    if not str(r.amount_signed).startswith(r.description.split()[1][0]):
        print(f"  TEMPLATE BUG: {r.description!r} for an amount of {r.amount_signed:,.2f}")
        break

print("\n--- 5. PAYMENT TIMING REALISM ---")
lags = [(pd.Timestamp(bank.loc[bank.txn_id == next(iter(r.B)), "value_date"].iloc[0]) - issued[next(iter(r.I))]).days
        for r in gt.itertuples() if r.match_type == "one_to_one"]
s = pd.Series(lags)
print(f"  issue->payment lag: min={s.min()}d  p50={s.median():.0f}d  max={s.max()}d")
due_of = dict(zip(inv.invoice_id, pd.to_datetime(inv.due_date)))
early = sum(1 for r in gt.itertuples() if r.match_type == "one_to_one"
            and pd.Timestamp(bank.loc[bank.txn_id == next(iter(r.B)), "value_date"].iloc[0]) < due_of[next(iter(r.I))])
print(f"  paid before due date: {early}  |  paid within 3d of issue: {(s <= 3).sum()}"
      f"  |  90+ days after issue: {(s >= 90).sum()}")

print("\n--- 6. REALIZED MIX (ground-truth rows, not invoices) ---")
for k, v in gt.reason_code.value_counts().items():
    print(f"  {k:<24}{v:>4}  {v / len(gt):6.1%}")
