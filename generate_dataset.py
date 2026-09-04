#!/usr/bin/env python3
"""
Deterministic Synthetic Dataset Generator — AI Finance Controller (Track 04)
=============================================================================

Generates a multi-source reconciliation dataset with controlled corruption
scenarios and 100% accurate ground-truth labels.

Outputs:
    data/invoices.csv          — Accounts receivable ledger
    data/bank_statement.csv    — Bank transaction history
    data/processor_payouts.csv — Payment processor settlements
    data/ground_truth.csv      — Reconciliation answer key

Guarantees:
    - All amounts use Decimal arithmetic (no float drift)
    - Fully deterministic with fixed seed (re-run = identical output)
    - Zero LLM dependency — every value is computed programmatically
    - 100% accurate ground truth labels

Usage:
    python generate_dataset.py
    python generate_dataset.py --seed 42 --num-invoices 200 --output-dir data
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker


# ═══════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

DEFAULT_SEED = 42
DEFAULT_NUM_INVOICES = 200
DEFAULT_NUM_CUSTOMERS = 30
DEFAULT_OUTPUT_DIR = "data"

# Scenario weights (sum to exactly 0.90; remaining 0.10 = true exceptions)
SCENARIO_DISTRIBUTION = {
    "clean":           0.26,
    "processor_fee":   0.12,
    "timing_lag":      0.10,
    "bundled_payout":  0.10,
    "split_payment":   0.08,
    "messy_narration": 0.08,
    "name_variant":    0.05,
    "fx_conversion":   0.04,
    "duplicate_bank":  0.03,
    "decoy":           0.04,
}

# Fee structures
RAZORPAY_FEE_RATE = Decimal("0.02")     # 2%
RAZORPAY_GST_ON_FEE = Decimal("0.18")   # 18% GST on the fee
STRIPE_FEE_RATE = Decimal("0.029")      # 2.9%
STRIPE_FLAT_FEE = Decimal("2.36")       # ~₹2.36 flat

# Realistic IFSC codes
IFSC_CODES = [
    "HDFC0001234", "ICIC0002345", "SBIN0003456", "UTIB0004567",
    "KKBK0005678", "PUNB0006789", "BARB0007890", "IDFB0008901",
    "YESB0009012", "INDB0010123",
]

PAYMENT_TERMS = ["NET15", "NET30", "NET45", "NET60"]
PAYMENT_RAILS = ["NEFT", "IMPS", "RTGS", "UPI"]

FX_RATES = {
    "USD": Decimal("83.50"),
    "EUR": Decimal("91.20"),
    "GBP": Decimal("106.30"),
}
FX_CURRENCIES = list(FX_RATES.keys())


# ═══════════════════════════════════════════════════════════════════
# 2. NARRATION TEMPLATES (hardcoded — designed offline, not generated)
# ═══════════════════════════════════════════════════════════════════

NARRATION_TEMPLATES = {
    "NEFT": [
        "NEFT-{ifsc}-{company}-{ref}",
        "NEFT CR {ifsc} {company} {ref}",
        "NEFT/{ref}/{company}",
        "NEFT CR-{ref}-{company} PVT LTD",
        "NEFT-{ifsc}-{company}-INV{ref_short}",
        "NEFT CR {company} REF {ref_short}",
        "NEFT-{ifsc}-{company}",                # missing reference
        "NEFT CR {ref_short}",                   # no company name
    ],
    "IMPS": [
        "IMPS/{ref}/{company}/INR",
        "IMPS-{ref}-{company}",
        "IMPS P2M {ref} {company}",
        "IMPS/{ref}/{company}/PAYMENT",
        "IMPS CR {company} {ref_short}",
    ],
    "RTGS": [
        "RTGS-{ifsc}{ref}-{company} LTD",
        "RTGS CR {ifsc} {company} {ref}",
        "RTGS-{ifsc}-{company}-{ref}",
        "RTGS/{company}/{ref}",
    ],
    "UPI": [
        "UPI-{company}-{ref}-PAYMENT",
        "UPI/{ref}/{company}@okaxis",
        "UPI CR {company} {ref_short}",
        "UPI-{ref}-{company}",
        "UPI/{company}/{ref_short}",
    ],
}

PROCESSOR_NARRATION_TEMPLATES = [
    "RAZORPAY PAYOUT RPY-{payout_id}",
    "RAZORPAY*{company} RPY{payout_id}",
    "RAZORPAY SETTLEMENT-{payout_id}",
    "STRIPE TRANSFER ST-{payout_id}",
    "STRIPE PAYOUT po_{payout_id}",
    "CASHFREE PAY-{payout_id}",
]

EXCEPTION_NARRATION_TEMPLATES = [
    "INT CREDIT {ref}",
    "INTEREST PAYMENT Q{quarter}",
    "REFUND-{ref}",
    "REV-{ref} CHARGEBACK",
    "BANK CHARGES {month}",
    "GST TDS CREDIT {ref_short}",
    "DEPOSIT {amount_short}",
    "MISCELLANEOUS CREDIT",
    "SERVICE TAX REFUND {ref_short}",
    "NACH DR {ref}",
]


# ═══════════════════════════════════════════════════════════════════
# 3. HELPERS
# ═══════════════════════════════════════════════════════════════════

def d(value) -> Decimal:
    """Convert to Decimal, quantized to 2 decimal places."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def gen_id(prefix: str, index: int, width: int = 6) -> str:
    """Generate a formatted ID like INV-000001."""
    return f"{prefix}-{index:0{width}d}"


def rand_date(start: datetime, end: datetime, rng: random.Random) -> datetime:
    """Random date in [start, end]."""
    delta = max(0, (end - start).days)
    return start + timedelta(days=rng.randint(0, delta))


def build_narration(
    rail: str,
    company: str,
    ref: str,
    ifsc: str,
    rng: random.Random,
    payout_id: str = "",
    amount: Decimal = Decimal("0"),
) -> str:
    """Fill a narration template with deterministic values, then truncate."""
    ref_num = ref.split("-")[-1] if "-" in ref else ref
    ref_short_int = str(int(ref_num)) if ref_num.isdigit() else ref_num[-4:]
    ref_variants = [
        ref,                         # e.g. INV-000068
        f"INV{ref_short_int}",       # e.g. INV68
        ref_num,                     # e.g. 000068
        ref_short_int,               # e.g. 68
        f"REF{ref_num[-4:]}",        # e.g. REF0068
        "",                          # reference omitted
    ]
    chosen_ref = rng.choice(ref_variants)
    ref_short = ref_num[-4:] if len(ref_num) >= 4 else ref_num
    amount_short = str(amount)[-4:] if amount else "0000"

    if rail == "PROCESSOR":
        tpl = rng.choice(PROCESSOR_NARRATION_TEMPLATES)
        text = tpl.format(
            company=company.upper()[:20],
            payout_id=payout_id,
            ref=ref,
        )
    else:
        tpl = rng.choice(NARRATION_TEMPLATES.get(rail, NARRATION_TEMPLATES["NEFT"]))
        text = tpl.format(
            company=company.upper()[:20],
            ref=chosen_ref,
            ref_short=ref_short,
            ifsc=ifsc,
            amount=str(amount),
            amount_short=amount_short,
        )

    # Realistic truncation at 32-50 chars
    return text[: rng.randint(32, 50)]


def build_messy_narration(
    company: str, ref: str, ifsc: str, rng: random.Random, amount: Decimal
) -> str:
    """Deliberately garbled narration — missing ref, truncated, wrong format."""
    choices = [
        f"NEFT-{ifsc}-{company.upper()[:12]}",
        f"NEFT CR {company.upper()[:8]}... {ref[:3]}",
        f"IMPS/{ref[::-1][:6]}/{company.upper()[:10]}",
        f"DEPOSIT {str(amount)[-4:]}",
        f"NEFT {company.split()[0][:4].upper()} {ref[-4:]}",
        f"CR-{ref.replace('-', '')}#{company.upper()[:6]}",
        f"TRF FROM {company.upper()[:15]}",
        f"CREDIT {company.split()[0].upper()[:6]}",
    ]
    return rng.choice(choices)


def name_variant(name: str, rng: random.Random) -> str:
    """Realistic variant of a company name."""
    options = [
        name.upper(),
        name.lower(),
        name.title(),
        name.replace(" Pvt Ltd", " Private Limited") if " Pvt Ltd" in name else name + " PVT LTD",
        name.replace(" Pvt Ltd", " PVT LTD") if " Pvt Ltd" in name else name.upper(),
        name.replace(" Pvt Ltd", "") if " Pvt Ltd" in name else name,
        name.split()[0] + " CORP" if len(name.split()) > 1 else name + " LTD",
        name[:15] if len(name) > 15 else name,
        "M/S " + name,
        name.replace(" ", ""),
        name + " India",
    ]
    return rng.choice(options)


# ═══════════════════════════════════════════════════════════════════
# 4. DATA GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_customers(n: int, fake: Faker, rng: random.Random) -> list[dict]:
    """Create a fixed pool of Indian companies."""
    customers = []
    for i in range(n):
        customers.append({
            "customer_id": gen_id("CUST", i + 1, 4),
            "customer_name": fake.company(),
        })
    return customers


def generate_invoices(
    n: int,
    customers: list[dict],
    fake: Faker,
    rng: random.Random,
    np_rng: np.random.RandomState,
) -> list[dict]:
    """Generate invoices with log-normal amounts and realistic dates."""
    # Fixed reference window for reproducibility
    end_date = datetime(2025, 3, 31)
    start_date = end_date - timedelta(days=90)

    # Tiered customer weights so invoice volume reflects realistic business patterns:
    # 5 anchor accounts (wt 4), 15 mid-market (wt 2), 10 small (wt 1)
    n_cust = len(customers)
    cust_weights = [4] * min(5, n_cust) + [2] * min(15, max(0, n_cust - 5)) + [1] * max(0, n_cust - 20)
    if len(cust_weights) < n_cust:
        cust_weights += [1] * (n_cust - len(cust_weights))
    elif len(cust_weights) > n_cust:
        cust_weights = cust_weights[:n_cust]
    total_w = sum(cust_weights)
    cust_probs = [w / total_w for w in cust_weights]

    # Pre-allocate at least 2 invoices per customer, then fill the remainder by weight
    cust_pool = []
    min_per_cust = 2 if n >= 2 * n_cust else 1
    for c in customers:
        cust_pool.extend([c] * min_per_cust)
    rem_count = n - len(cust_pool)
    if rem_count > 0:
        cust_pool.extend(rng.choices(customers, weights=cust_probs, k=rem_count))
    rng.shuffle(cust_pool)

    invoices = []
    for i in range(n):
        # Log-normal → mostly ₹5K-₹50K, occasional ₹5L+
        raw = np_rng.lognormal(mean=9.5, sigma=1.0)
        base_amount = d(max(500, min(1_000_000, raw)))
        tax = d(base_amount * Decimal("0.18"))
        total = d(base_amount + tax)

        issue = rand_date(start_date, end_date, rng)
        terms = rng.choice(PAYMENT_TERMS)
        term_days = int(terms.replace("NET", ""))
        due = issue + timedelta(days=term_days)

        cust = cust_pool[i]

        invoices.append({
            "invoice_id": gen_id("INV", i + 1),
            "customer_id": cust["customer_id"],
            "customer_name": cust["customer_name"],
            "issue_date": issue.strftime("%Y-%m-%d"),
            "due_date": due.strftime("%Y-%m-%d"),
            "currency": "INR",
            "amount_gross": total,
            "tax_amount": tax,
            "terms": terms,
            "status": "issued",
            # ── internal (not exported) ──
            "_scenario": None,
        })
    return invoices


# ═══════════════════════════════════════════════════════════════════
# 5. SCENARIO ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════

def assign_scenarios(invoices: list[dict], rng: random.Random):
    """
    Tag each invoice with a scenario and separate into processing groups.

    Returns:
        regular   — invoices with 1-to-1 scenarios
        unpaid    — invoices with no bank match (exceptions)
        bundles   — list of invoice groups for bundled payout
        decoys    — list of (inv_a, inv_b) pairs with identical amounts
    """
    n = len(invoices)
    idx = list(range(n))
    rng.shuffle(idx)

    regular: list[dict] = []
    unpaid: list[dict] = []
    bundles: list[list[dict]] = []
    decoys: list[tuple[dict, dict]] = []

    pos = 0
    for scenario, weight in SCENARIO_DISTRIBUTION.items():
        count = int(n * weight)

        if scenario == "bundled_payout":
            # carve into groups of 3-5
            pool = [invoices[idx[j]] for j in range(pos, min(pos + count, n))]
            for inv in pool:
                inv["_scenario"] = "bundled_payout"
            gi = 0
            while gi < len(pool):
                rem = len(pool) - gi
                if rem <= 5:
                    sz = rem
                else:
                    sz = rng.randint(3, 4)
                grp = pool[gi : gi + sz]
                if len(grp) >= 2:
                    bundles.append(grp)
                else:
                    for inv in grp:
                        inv["_scenario"] = "clean"
                    regular.extend(grp)
                gi += sz
            pos += count

        elif scenario == "decoy":
            pool = [invoices[idx[j]] for j in range(pos, min(pos + count, n))]
            for inv in pool:
                inv["_scenario"] = "decoy"
            pi = 0
            while pi + 1 < len(pool):
                a, b = pool[pi], pool[pi + 1]
                # force identical amount & close dates
                b["amount_gross"] = a["amount_gross"]
                b["tax_amount"] = a["tax_amount"]
                b["issue_date"] = a["issue_date"]
                b["due_date"] = a["due_date"]
                decoys.append((a, b))
                pi += 2
            if pi < len(pool):
                pool[pi]["_scenario"] = "clean"
                regular.append(pool[pi])
            pos += count

        else:
            for j in range(pos, min(pos + count, n)):
                invoices[idx[j]]["_scenario"] = scenario
                regular.append(invoices[idx[j]])
            pos += count

    # ~10 % true exceptions (unpaid invoices)
    exc_count = int(n * 0.10)
    for j in range(pos, min(pos + exc_count, n)):
        inv = invoices[idx[j]]
        inv["_scenario"] = "exception_unpaid"
        inv["status"] = rng.choice(["overdue", "disputed", "pending"])
        unpaid.append(inv)
    pos += exc_count

    # leftover → clean
    for j in range(pos, n):
        invoices[idx[j]]["_scenario"] = "clean"
        regular.append(invoices[idx[j]])

    return regular, unpaid, bundles, decoys



# ═══════════════════════════════════════════════════════════════════
# 6. BANK-LINE GENERATOR (core engine)
# ═══════════════════════════════════════════════════════════════════

class Reconciler:
    """Produces bank lines + ground truth for every assigned scenario."""

    def __init__(self, rng: random.Random, fake: Faker):
        self.rng = rng
        self.fake = fake
        self._txn = 0
        self._pay = 0
        self._match = 0
        self._chg = 0
        self.bank_lines: list[dict] = []
        self.ground_truth: list[dict] = []
        self.charges: list[dict] = []   # raw charges → aggregated into payouts

    # ── id generators ────────────────────────────
    def _tid(self):
        self._txn += 1; return gen_id("TXN", self._txn)

    def _pid(self):
        self._pay += 1; return gen_id("PAY", self._pay)

    def _mid(self):
        self._match += 1; return gen_id("MATCH", self._match)

    def _cid(self):
        self._chg += 1; return gen_id("CHG", self._chg)

    def _ifsc(self):
        return self.rng.choice(IFSC_CODES)

    def _rail(self):
        return self.rng.choice(PAYMENT_RAILS)

    # ── low-level helpers ────────────────────────
    def _bl(self, amount, value_date, desc, currency="INR", posted_date=None):
        tid = self._tid()
        if posted_date is None:
            vd = datetime.strptime(value_date, "%Y-%m-%d")
            posted_date = (vd + timedelta(days=self.rng.randint(0, 1))).strftime("%Y-%m-%d")
        return {
            "txn_id": tid,
            "value_date": value_date,
            "posted_date": posted_date,
            "amount_signed": amount,
            "currency": currency,
            "description": desc,
            "running_balance": Decimal("0"),
        }

    def _gt(self, mtype, inv_ids, txn_ids, reason, resolution):
        self.ground_truth.append({
            "match_id": self._mid(),
            "match_type": mtype,
            "invoice_ids": "|".join(inv_ids),
            "bank_txn_ids": "|".join(txn_ids),
            "reason_code": reason,
            "expected_resolution": resolution,
        })

    def _pay_date(self, inv, extra_lag=0):
        due = datetime.strptime(inv["due_date"], "%Y-%m-%d")
        return (due + timedelta(days=self.rng.randint(-2, 5) + extra_lag)).strftime("%Y-%m-%d")

    # ── scenario handlers ────────────────────────

    def do_clean(self, inv):
        amt = inv["amount_gross"]
        narr = build_narration(
            self._rail(), inv["customer_name"], inv["invoice_id"],
            self._ifsc(), self.rng, amount=amt,
        )
        bl = self._bl(amt, self._pay_date(inv), narr)
        self.bank_lines.append(bl)
        self._gt("one_to_one", [inv["invoice_id"]], [bl["txn_id"]],
                 "clean", "auto_match")

    def do_processor_fee(self, inv):
        amt = inv["amount_gross"]
        if self.rng.random() < 0.6:
            base_fee = d(amt * RAZORPAY_FEE_RATE)
            gst = d(base_fee * RAZORPAY_GST_ON_FEE)
            fee = d(base_fee + gst)
            proc = "RAZORPAY"
        else:
            fee = d(amt * STRIPE_FEE_RATE + STRIPE_FLAT_FEE)
            proc = "STRIPE"
        net = d(amt - fee)

        pid = self._pid()
        narr = build_narration(
            "PROCESSOR", inv["customer_name"], inv["invoice_id"],
            "", self.rng, payout_id=pid[-6:], amount=net,
        )
        pd_ = self._pay_date(inv)
        bl = self._bl(net, pd_, narr)
        self.bank_lines.append(bl)

        self.charges.append({
            "payout_id": pid, "payout_date": pd_,
            "charge_id": self._cid(),
            "gross": amt, "fees": fee,
            "refunds": Decimal("0"), "chargebacks": Decimal("0"),
            "net": net, "invoice_id": inv["invoice_id"],
            "processor": proc,
        })
        self._gt("one_to_one", [inv["invoice_id"]], [bl["txn_id"]],
                 "processor_fee", "match_with_fee_adjustment")

    def do_timing_lag(self, inv):
        amt = inv["amount_gross"]
        narr = build_narration(
            self._rail(), inv["customer_name"], inv["invoice_id"],
            self._ifsc(), self.rng, amount=amt,
        )
        lag = self.rng.randint(1, 5)
        due = datetime.strptime(inv["due_date"], "%Y-%m-%d")
        vd = (due + timedelta(days=lag)).strftime("%Y-%m-%d")

        posted = None
        if self.rng.random() < 0.3:
            posted = (due + timedelta(days=lag + self.rng.randint(1, 2))).strftime("%Y-%m-%d")

        bl = self._bl(amt, vd, narr, posted_date=posted)
        self.bank_lines.append(bl)
        self._gt("one_to_one", [inv["invoice_id"]], [bl["txn_id"]],
                 "timing_lag", "match_with_date_tolerance")

    def do_bundled(self, group: list[dict]):
        total = d(sum(inv["amount_gross"] for inv in group))
        pid = self._pid()

        # half the bundles also have a fee deducted
        if self.rng.random() < 0.5:
            base_fee = d(total * RAZORPAY_FEE_RATE)
            gst = d(base_fee * RAZORPAY_GST_ON_FEE)
            fee = d(base_fee + gst)
        else:
            fee = Decimal("0")
        net = d(total - fee)

        narr = build_narration(
            "PROCESSOR", group[0]["customer_name"], "",
            "", self.rng, payout_id=pid[-6:], amount=net,
        )
        latest_due = max(datetime.strptime(i["due_date"], "%Y-%m-%d") for i in group)
        pd_ = (latest_due + timedelta(days=self.rng.randint(1, 3))).strftime("%Y-%m-%d")

        bl = self._bl(net, pd_, narr)
        self.bank_lines.append(bl)

        inv_ids = [i["invoice_id"] for i in group]
        for inv in group:
            per_fee = d(inv["amount_gross"] / total * fee) if fee else Decimal("0")
            self.charges.append({
                "payout_id": pid, "payout_date": pd_,
                "charge_id": self._cid(),
                "gross": inv["amount_gross"],
                "fees": per_fee,
                "refunds": Decimal("0"), "chargebacks": Decimal("0"),
                "net": d(inv["amount_gross"] - per_fee),
                "invoice_id": inv["invoice_id"],
                "processor": "RAZORPAY",
            })

        reason = "bundled_payout_with_fee" if fee else "bundled_payout"
        self._gt("many_to_one", inv_ids, [bl["txn_id"]],
                 reason, "match_bundled")

    def do_split(self, inv):
        amt = inv["amount_gross"]
        pct = d(str(self.rng.randint(30, 60) / 100))
        a1 = d(amt * pct)
        a2 = d(amt - a1)

        rail, ifsc = self._rail(), self._ifsc()
        n1 = build_narration(rail, inv["customer_name"],
                             inv["invoice_id"] + "-A", ifsc, self.rng, amount=a1)
        n2 = build_narration(rail, inv["customer_name"],
                             inv["invoice_id"] + "-B", ifsc, self.rng, amount=a2)

        due = datetime.strptime(inv["due_date"], "%Y-%m-%d")
        d1 = (due + timedelta(days=self.rng.randint(-2, 3))).strftime("%Y-%m-%d")
        d2 = (due + timedelta(days=self.rng.randint(4, 18))).strftime("%Y-%m-%d")

        bl1 = self._bl(a1, d1, n1)
        bl2 = self._bl(a2, d2, n2)
        self.bank_lines.extend([bl1, bl2])
        self._gt("one_to_many", [inv["invoice_id"]],
                 [bl1["txn_id"], bl2["txn_id"]],
                 "split_payment", "match_split")

    def do_messy(self, inv):
        amt = inv["amount_gross"]
        narr = build_messy_narration(
            inv["customer_name"], inv["invoice_id"],
            self._ifsc(), self.rng, amt,
        )
        bl = self._bl(amt, self._pay_date(inv), narr)
        self.bank_lines.append(bl)
        self._gt("one_to_one", [inv["invoice_id"]], [bl["txn_id"]],
                 "messy_narration", "match_fuzzy")

    def do_name_variant(self, inv):
        amt = inv["amount_gross"]
        variant = name_variant(inv["customer_name"], self.rng)
        narr = build_narration(
            self._rail(), variant, inv["invoice_id"],
            self._ifsc(), self.rng, amount=amt,
        )
        bl = self._bl(amt, self._pay_date(inv), narr)
        self.bank_lines.append(bl)
        self._gt("one_to_one", [inv["invoice_id"]], [bl["txn_id"]],
                 "name_variant", "match_entity_resolution")

    def do_fx(self, inv):
        inr = inv["amount_gross"]
        ccy = self.rng.choice(FX_CURRENCIES)
        rate = FX_RATES[ccy]
        spread = Decimal(str(round(self.rng.uniform(0.985, 1.015), 4)))
        eff_rate = d(rate * spread)
        foreign = d(inr / eff_rate)
        bank_inr = d(foreign * rate)       # bank settlement in domestic INR

        # Update invoice to reflect billing in foreign currency
        inv["currency"] = ccy
        inv["amount_gross"] = foreign
        inv["tax_amount"] = Decimal("0.00")  # export invoice: zero-rated

        narr_choices = [
            f"INWARD TT {ccy} {foreign} REF {inv['invoice_id'][-6:]}",
            f"FOREX CR {ccy}/{inv['customer_name'].upper()[:12]}-{inv['invoice_id'][-4:]}",
            f"NEFT-{self._ifsc()}-{inv['customer_name'].upper()[:10]}-{ccy}",
            f"SWIFT INW {inv['customer_name'].upper()[:14]} {foreign}",
        ]
        narr = self.rng.choice(narr_choices)[: self.rng.randint(32, 50)]
        bl = self._bl(bank_inr, self._pay_date(inv), narr)
        self.bank_lines.append(bl)
        self._gt("one_to_one", [inv["invoice_id"]], [bl["txn_id"]],
                 "fx_conversion", "match_with_fx_tolerance")


    def do_duplicate(self, inv):
        amt = inv["amount_gross"]
        rail, ifsc = self._rail(), self._ifsc()
        narr = build_narration(
            rail, inv["customer_name"], inv["invoice_id"],
            ifsc, self.rng, amount=amt,
        )
        pd_ = self._pay_date(inv)
        bl1 = self._bl(amt, pd_, narr)

        # duplicate: same narration, 0-1 day later
        dup_d = datetime.strptime(pd_, "%Y-%m-%d") + timedelta(days=self.rng.randint(0, 1))
        bl2 = self._bl(amt, dup_d.strftime("%Y-%m-%d"), narr)

        self.bank_lines.extend([bl1, bl2])
        self._gt("one_to_one", [inv["invoice_id"]], [bl1["txn_id"]],
                 "clean", "auto_match")
        self._gt("duplicate", [], [bl2["txn_id"]],
                 "duplicate_bank_line", "flag_duplicate")

    def do_decoy(self, a, b):
        amt = a["amount_gross"]
        rail, ifsc = self._rail(), self._ifsc()

        na = build_narration(rail, a["customer_name"], a["invoice_id"],
                             ifsc, self.rng, amount=amt)
        nb = build_narration(rail, b["customer_name"], b["invoice_id"],
                             ifsc, self.rng, amount=amt)

        bla = self._bl(amt, self._pay_date(a), na)
        blb = self._bl(amt, self._pay_date(b), nb)
        self.bank_lines.extend([bla, blb])

        self._gt("one_to_one", [a["invoice_id"]], [bla["txn_id"]],
                 "decoy", "match_by_reference")
        self._gt("one_to_one", [b["invoice_id"]], [blb["txn_id"]],
                 "decoy", "match_by_reference")

    # ── exceptions ───────────────────────────────

    def add_unpaid_invoices(self, unpaid: list[dict]):
        """Invoices with no bank match."""
        for inv in unpaid:
            self._gt("unmatchable", [inv["invoice_id"]], [],
                     "unpaid_invoice", "add_to_exception_list")

    def add_orphan_bank_lines(self, count: int):
        """Bank credits/debits with no matching invoice."""
        base = datetime(2025, 1, 15)
        for _ in range(count):
            amt = d(str(self.rng.uniform(50, 15000)))
            dt = (base + timedelta(days=self.rng.randint(0, 80))).strftime("%Y-%m-%d")

            tpl = self.rng.choice(EXCEPTION_NARRATION_TEMPLATES)
            narr = tpl.format(
                ref=gen_id("REF", self.rng.randint(1000, 9999)),
                ref_short=str(self.rng.randint(100000, 999999)),
                quarter=str(self.rng.randint(1, 4)),
                month=datetime.strptime(dt, "%Y-%m-%d").strftime("%b").upper(),
                amount_short=str(amt)[-4:],
            )
            if self.rng.random() < 0.3:
                amt = -amt                   # debit (bank charge)

            bl = self._bl(amt, dt, narr)
            self.bank_lines.append(bl)
            exc_type = self.rng.choice([
                "interest_credit", "bank_charge", "refund_no_invoice",
                "tax_credit", "miscellaneous",
            ])
            self._gt("unmatchable", [], [bl["txn_id"]],
                     exc_type, "add_to_exception_list")


# ═══════════════════════════════════════════════════════════════════
# 7. PAYOUT AGGREGATION & RUNNING BALANCE
# ═══════════════════════════════════════════════════════════════════

def aggregate_payouts(charges: list[dict]) -> list[dict]:
    """Roll individual charges up to payout-level records."""
    buckets: dict[str, dict] = {}
    for ch in charges:
        pid = ch["payout_id"]
        if pid not in buckets:
            buckets[pid] = {
                "payout_id": pid,
                "payout_date": ch["payout_date"],
                "gross": Decimal("0"), "fees": Decimal("0"),
                "refunds": Decimal("0"), "chargebacks": Decimal("0"),
                "net": Decimal("0"), "charge_ids": [],
            }
        b = buckets[pid]
        b["gross"] += ch["gross"]
        b["fees"] += ch["fees"]
        b["refunds"] += ch["refunds"]
        b["chargebacks"] += ch["chargebacks"]
        b["net"] += ch["net"]
        b["charge_ids"].append(ch["charge_id"])

    out = []
    for b in buckets.values():
        out.append({
            "payout_id": b["payout_id"],
            "payout_date": b["payout_date"],
            "gross": d(b["gross"]),
            "fees": d(b["fees"]),
            "refunds": d(b["refunds"]),
            "chargebacks": d(b["chargebacks"]),
            "net": d(b["net"]),
            "charge_ids": "|".join(b["charge_ids"]),
        })
    return sorted(out, key=lambda x: x["payout_date"])


def compute_running_balance(bank_lines: list[dict], opening=Decimal("500000.00")):
    """Sort by date and accumulate running balance."""
    bank_lines.sort(key=lambda x: (x["value_date"], x["txn_id"]))
    bal = opening
    for bl in bank_lines:
        bal = d(bal + bl["amount_signed"])
        bl["running_balance"] = bal
    return bank_lines


# ═══════════════════════════════════════════════════════════════════
# 8. EXPORT
# ═══════════════════════════════════════════════════════════════════

def export(invoices, bank_lines, payouts, ground_truth, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    inv_cols = [
        "invoice_id", "customer_id", "customer_name", "issue_date",
        "due_date", "currency", "amount_gross", "tax_amount", "terms", "status",
    ]
    pd.DataFrame(invoices)[inv_cols].to_csv(
        os.path.join(out_dir, "invoices.csv"), index=False)

    bl_cols = [
        "txn_id", "value_date", "posted_date", "amount_signed",
        "currency", "description", "running_balance",
    ]
    pd.DataFrame(bank_lines)[bl_cols].to_csv(
        os.path.join(out_dir, "bank_statement.csv"), index=False)

    if payouts:
        pay_cols = [
            "payout_id", "payout_date", "gross", "fees",
            "refunds", "chargebacks", "net", "charge_ids",
        ]
        pd.DataFrame(payouts)[pay_cols].to_csv(
            os.path.join(out_dir, "processor_payouts.csv"), index=False)

    gt_cols = [
        "match_id", "match_type", "invoice_ids", "bank_txn_ids",
        "reason_code", "expected_resolution",
    ]
    pd.DataFrame(ground_truth)[gt_cols].to_csv(
        os.path.join(out_dir, "ground_truth.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic reconciliation dataset")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--num-invoices", type=int, default=DEFAULT_NUM_INVOICES)
    ap.add_argument("--num-customers", type=int, default=DEFAULT_NUM_CUSTOMERS)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    # ── deterministic setup ──────────────────────
    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)
    fake = Faker("en_IN")
    Faker.seed(args.seed)

    print(f"{'='*60}")
    print(f"  Synthetic Dataset Generator — Track 04")
    print(f"  seed={args.seed}  invoices={args.num_invoices}")
    print(f"{'='*60}\n")

    # ── generate ─────────────────────────────────
    customers = generate_customers(args.num_customers, fake, rng)
    print(f"[1/7] Generated {len(customers)} customers")

    invoices = generate_invoices(args.num_invoices, customers, fake, rng, np_rng)
    print(f"[2/7] Generated {len(invoices)} invoices")

    regular, unpaid, bundles, decoys = assign_scenarios(invoices, rng)
    counts = Counter(i["_scenario"] for i in invoices)
    print(f"[3/7] Assigned scenarios:")
    for sc, cnt in sorted(counts.items()):
        print(f"       {sc:28s} {cnt:4d}  ({cnt/len(invoices)*100:5.1f}%)")

    # ── build bank lines & ground truth ──────────
    rec = Reconciler(rng, fake)

    dispatch = {
        "clean":           rec.do_clean,
        "processor_fee":   rec.do_processor_fee,
        "timing_lag":      rec.do_timing_lag,
        "split_payment":   rec.do_split,
        "messy_narration": rec.do_messy,
        "name_variant":    rec.do_name_variant,
        "fx_conversion":   rec.do_fx,
        "duplicate_bank":  rec.do_duplicate,
    }
    for inv in regular:
        handler = dispatch.get(inv["_scenario"])
        if handler:
            handler(inv)

    for grp in bundles:
        rec.do_bundled(grp)

    for a, b in decoys:
        rec.do_decoy(a, b)

    rec.add_unpaid_invoices(unpaid)
    orphan_count = max(8, int(len(invoices) * 0.05))
    rec.add_orphan_bank_lines(orphan_count)

    print(f"[4/7] Generated {len(rec.bank_lines)} bank lines")
    print(f"[5/7] Generated {len(rec.ground_truth)} ground-truth entries")

    # ── post-processing ──────────────────────────
    rec.bank_lines = compute_running_balance(rec.bank_lines)
    print(f"[6/7] Computed running balance")

    payouts = aggregate_payouts(rec.charges)
    print(f"       Generated {len(payouts)} processor payouts")

    # ── export ───────────────────────────────────
    export(invoices, rec.bank_lines, payouts, rec.ground_truth, args.output_dir)
    print(f"[7/7] Exported to {args.output_dir}/\n")

    # ── summary ──────────────────────────────────
    gt_types = Counter(g["match_type"] for g in rec.ground_truth)
    reason_codes = Counter(g["reason_code"] for g in rec.ground_truth)

    print(f"{'-'*60}")
    print(f"  SUMMARY")
    print(f"{'-'*60}")
    print(f"  Invoices:           {len(invoices):>6}")
    print(f"  Bank lines:         {len(rec.bank_lines):>6}")
    print(f"  Processor payouts:  {len(payouts):>6}")
    print(f"  Ground-truth rows:  {len(rec.ground_truth):>6}")
    print()
    print(f"  Match types:")
    for mt, c in sorted(gt_types.items()):
        print(f"    {mt:20s} {c:4d}")
    print()
    print(f"  Reason codes:")
    for rc, c in sorted(reason_codes.items()):
        print(f"    {rc:28s} {c:4d}")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
