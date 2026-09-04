"""Data quality audit: structural invariants of the generated dataset."""
import pandas as pd
from generate_dataset import SCENARIO_DISTRIBUTION, FX_RATES

gt = pd.read_csv("data/ground_truth.csv")
inv = pd.read_csv("data/invoices.csv")
bank = pd.read_csv("data/bank_statement.csv")
pay = pd.read_csv("data/processor_payouts.csv")

print("=" * 65)
print("       DATA QUALITY AUDIT & HYPOTHESIS RECHECK")
print("=" * 65)

checks_passed = 0
checks_total = 0

def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        checks_passed += 1
    print(f"[{status}] {name} {detail}")

# ── 1. Scenario weights & Exception Coverage ────
print("\n--- 1. Exception Coverage & Weights ---")
exc = gt[gt["match_type"] == "unmatchable"]
unpaid = exc[exc["reason_code"] == "unpaid_invoice"]
orphan = exc[exc["reason_code"] != "unpaid_invoice"]
weight_sum = sum(SCENARIO_DISTRIBUTION.values())
print(f"  Scenario weights sum: {weight_sum:.2f} (Leaves {1 - weight_sum:.2f} for unpaid)")
print(f"  Unpaid invoices in GT: {len(unpaid)} ({len(unpaid)/len(inv)*100:.1f}% of invoices)")
print(f"  Orphan bank lines in GT: {len(orphan)}")
print(f"  Total unmatchable: {len(exc)} / {len(gt)} = {len(exc)/len(gt)*100:.1f}%")

check("Scenario weights leave >= 10% for exceptions", abs(weight_sum - 0.90) < 0.001)
check("Unpaid invoices >= 10% of invoices", len(unpaid) >= int(len(inv) * 0.10))
check("Total exceptions >= 10% of GT entries", len(exc)/len(gt) >= 0.10)

# ── 2. Amount Distribution ──────────────────────
print("\n--- 2. Amount Distribution ---")
print(f"  Min:    {inv['amount_gross'].min():>12,.2f}")
print(f"  25th:   {inv['amount_gross'].quantile(0.25):>12,.2f}")
print(f"  Median: {inv['amount_gross'].quantile(0.50):>12,.2f}")
print(f"  75th:   {inv['amount_gross'].quantile(0.75):>12,.2f}")
print(f"  Max:    {inv['amount_gross'].max():>12,.2f}")
print(f"  Mean:   {inv['amount_gross'].mean():>12,.2f}")
check("Amount distribution has wide realistic spread", inv['amount_gross'].max() > inv['amount_gross'].min() * 50)

# ── 3. Bundled Payout Integrity ─────────────────
print("\n--- 3. Bundled Payout Integrity ---")
bundled = gt[gt["reason_code"].str.contains("bundled", na=False)]
bundle_ok = True
for _, row in bundled.iterrows():
    inv_ids = row["invoice_ids"].split("|")
    inv_sum = inv[inv["invoice_id"].isin(inv_ids)]["amount_gross"].sum()
    bank_amt = bank[bank["txn_id"] == row["bank_txn_ids"]]["amount_signed"].values[0]
    has_fee = "with_fee" in row["reason_code"]
    if has_fee:
        diff_ok = bank_amt < inv_sum
        if not diff_ok: bundle_ok = False
        print(f"  {row['match_id']} ({len(inv_ids)} invs): inv_sum={inv_sum:>10.2f}, bank={bank_amt:>10.2f} (fee: {inv_sum - bank_amt:.2f})")
    else:
        diff = abs(inv_sum - bank_amt)
        if diff >= 0.02: bundle_ok = False
        print(f"  {row['match_id']} ({len(inv_ids)} invs): inv_sum={inv_sum:>10.2f}, bank={bank_amt:>10.2f} (diff: {diff:.2f})")
check("All bundled payouts correctly aggregate invoice sets", bundle_ok and len(bundled) >= 4)

# ── 4. Split Payment Integrity ──────────────────
print("\n--- 4. Split Payment Integrity ---")
splits = gt[gt["reason_code"] == "split_payment"]
split_ok = 0
for _, row in splits.iterrows():
    inv_amt = inv[inv["invoice_id"] == row["invoice_ids"]]["amount_gross"].values[0]
    txn_ids = row["bank_txn_ids"].split("|")
    bank_sum = bank[bank["txn_id"].isin(txn_ids)]["amount_signed"].sum()
    if abs(inv_amt - bank_sum) < 0.02:
        split_ok += 1
print(f"  {split_ok}/{len(splits)} splits sum to exact invoice gross")
check("All split payments sum to exact invoice total", split_ok == len(splits) and len(splits) > 0)

# ── 5. Decoy Pairs (same amount, different entities) ─
print("\n--- 5. Decoy Pairs ---")
decoys = gt[gt["reason_code"] == "decoy"]
decoy_amts = []
for _, row in decoys.iterrows():
    inv_id = row["invoice_ids"]
    amt = inv[inv["invoice_id"] == inv_id]["amount_gross"].values[0]
    decoy_amts.append((inv_id, amt))
pairs_ok = True
for i in range(0, len(decoy_amts)-1, 2):
    same = abs(decoy_amts[i][1] - decoy_amts[i+1][1]) < 0.02
    if not same: pairs_ok = False
    print(f"  Pair ({decoy_amts[i][0]}, {decoy_amts[i+1][0]}): amount={decoy_amts[i][1]:.2f}, identical={same}")
check("All decoy pairs share identical amounts", pairs_ok and len(decoy_amts) >= 4)

# ── 6. FX Conversion Realism ────────────────────
print("\n--- 6. FX Cross-Border Remittances ---")
fx = gt[gt["reason_code"] == "fx_conversion"]
fx_ok = True
for _, row in fx.iterrows():
    inv_row = inv[inv["invoice_id"] == row["invoice_ids"]].iloc[0]
    bank_row = bank[bank["txn_id"] == row["bank_txn_ids"]].iloc[0]
    ccy = inv_row["currency"]
    f_amt = inv_row["amount_gross"]
    b_inr = bank_row["amount_signed"]
    implied_rate = b_inr / f_amt if f_amt > 0 else 0
    ref_rate = float(FX_RATES[ccy])
    rate_ok = abs(implied_rate - ref_rate) / ref_rate < 0.05
    if not rate_ok or ccy == "INR": fx_ok = False
    print(f"  {row['match_id']}: Inv={ccy} {f_amt:>8.2f} -> Bank=INR {b_inr:>10.2f} | Implied: {implied_rate:.2f} (Ref: {ref_rate:.2f})")
check("FX invoices billed in foreign currency with realistic INR settlement", fx_ok and len(fx) > 0)

# ── 7. Processor Fee Range ──────────────────────
print("\n--- 7. Processor Fee Range ---")
pf = gt[gt["reason_code"] == "processor_fee"]
fee_pcts = []
for _, row in pf.iterrows():
    inv_amt = inv[inv["invoice_id"] == row["invoice_ids"]]["amount_gross"].values[0]
    bank_amt = bank[bank["txn_id"] == row["bank_txn_ids"]]["amount_signed"].values[0]
    fee_pct = (inv_amt - bank_amt) / inv_amt * 100
    fee_pcts.append(fee_pct)
print(f"  Fee range: {min(fee_pcts):.2f}% to {max(fee_pcts):.2f}%")
check("Processor fee deductions match realistic gateway rates (2.0% - 3.5%)", 2.0 <= min(fee_pcts) <= max(fee_pcts) <= 3.5)

# ── 8. Invoice Status Alignment ─────────────────
print("\n--- 8. Invoice Status Alignment ---")
unpaid_ids = set(unpaid["invoice_ids"])
unpaid_statuses = inv[inv["invoice_id"].isin(unpaid_ids)]["status"].value_counts().to_dict()
matched_statuses = inv[~inv["invoice_id"].isin(unpaid_ids)]["status"].value_counts().to_dict()
print(f"  Unpaid invoice statuses in AR ledger: {unpaid_statuses}")
print(f"  Matched invoice statuses in AR ledger: {matched_statuses}")
# The AR status column must NOT predict the exception list. Real AR systems set
# status from aging, not from whether the cash arrived - if `status != issued`
# recovers the exceptions, the agent gets 2/3 of its exception list for free.
leak = set(inv[inv.status != "issued"].invoice_id)
tp = len(leak & set(unpaid_ids))
prec = tp / max(len(leak), 1); rec = tp / max(len(unpaid_ids), 1)
leak_f1 = 2 * prec * rec / max(prec + rec, 1e-9)
print(f"  status-as-oracle F1: {leak_f1:.1%} (want < 40% - near the base rate)")
check("Invoice status does not leak the exception label", leak_f1 < 0.40)

# ── 9. Narration Diversity ──────────────────────
print("\n--- 9. Sample Bank Narrations ---")
narr_samples = bank.sample(6, random_state=42)["description"].tolist()
for s in narr_samples:
    print(f"  {s}")
clean_giveaways = [s for s in bank["description"] if "INV-000" in s]
print(f"  Descriptions with literal 'INV-000' giveaway: {len(clean_giveaways)} / {len(bank)} ({len(clean_giveaways)/len(bank)*100:.1f}%)")
check("Bank narrations avoid trivial giveaways (< 25% with full prefix)", len(clean_giveaways) / len(bank) < 0.25)

# ── 10. Customer Volume Distribution ────────────
print("\n--- 10. Customer Volume Distribution ---")
vc = inv["customer_id"].value_counts()
print(f"  Min invoices/customer: {vc.min()}, Max: {vc.max()}, Mean: {vc.mean():.1f}")
check("Every customer has multiple transactions (min >= 2)", vc.min() >= 2)

# ── 11. Running Balance Consistency ─────────────
print("\n--- 11. Running Balance ---")
print(f"  Opening: 1,500,000.00 | Closing: {bank.iloc[-1]['running_balance']:,.2f} | Min: {bank['running_balance'].min():,.2f}")
check("Bank running balance stays positive throughout", (bank["running_balance"] > 0).all())
check("Statement has real operating outflows", (bank["amount_signed"] < 0).sum() >= 40)

# ── 12. Ground Truth Completeness ───────────────
print("\n--- 12. Ground Truth Completeness ---")
gt_inv_ids = set()
gt_bank_ids = set()
for _, row in gt.iterrows():
    if pd.notna(row["invoice_ids"]) and row["invoice_ids"]:
        gt_inv_ids.update(row["invoice_ids"].split("|"))
    if pd.notna(row["bank_txn_ids"]) and row["bank_txn_ids"]:
        gt_bank_ids.update(row["bank_txn_ids"].split("|"))

check("Every single invoice is in ground truth", gt_inv_ids == set(inv["invoice_id"]))
check("Every single bank line is in ground truth", gt_bank_ids == set(bank["txn_id"]))

print("\n" + "=" * 65)
print(f" SCORECARD: {checks_passed} / {checks_total} CHECKS PASSED ({checks_passed/checks_total*100:.0f}%)")
print("=" * 65)
