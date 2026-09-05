"""Frozen record types. Money is always Decimal."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    customer_id: str
    customer_name: str
    issue_date: date
    due_date: date
    currency: str
    amount_gross: Decimal
    tax_amount: Decimal
    terms: str
    status: str


@dataclass(frozen=True)
class BankLine:
    txn_id: str
    value_date: date
    posted_date: date
    amount_signed: Decimal
    currency: str
    description: str
    running_balance: Decimal


@dataclass(frozen=True)
class Payout:
    payout_id: str
    payout_date: date
    gross: Decimal
    fees: Decimal
    refunds: Decimal
    chargebacks: Decimal
    net: Decimal
    charge_ids: tuple[str, ...]
    # The payout file names its invoices directly. This is the intended join for
    # settlements: bank -> payout id -> invoice_refs, confirmed against net.
    invoice_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evidence:
    """What we could read out of one bank line's narration."""
    txn_id: str
    full_refs: tuple[str, ...]        # e.g. ("INV-000123",)
    prefix_refs: tuple[str, ...]      # e.g. ("INV-00",) — a startswith constraint
    suffix_refs: tuple[str, ...]      # e.g. ("0101",) — an endswith constraint
    split_suffix: str | None          # "A" or "B"
    payout_refs: tuple[str, ...]      # last-6 fragments of a payout id
    counterparty: str                 # normalized, may be ""
    currency_hint: str | None         # "USD" | "EUR" | "GBP"
    rail: str | None                  # NEFT | IMPS | RTGS | UPI | PROCESSOR


@dataclass(frozen=True)
class Candidate:
    invoice_ids: frozenset[str]
    bank_txn_ids: frozenset[str]
    source: str                       # exact_amount|payout_chain|fx_band|split|bundle
    features: dict[str, Any]
    score: float


@dataclass
class Decision:
    match_id: str
    invoice_ids: frozenset[str]
    bank_txn_ids: frozenset[str]
    match_type: str
    confidence: float
    route: str                        # "match" | "exception" | "review"
    decided_by: str                   # "rules" | "flash" | "pro" | "fallback"
    rationale: str
    reason_code: str = ""
    candidates_considered: list[Candidate] = field(default_factory=list)
