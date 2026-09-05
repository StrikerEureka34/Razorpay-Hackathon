"""Pull structured evidence out of free-text bank narration.

Narration is truncated to 32-50 chars by the generator, so references
frequently survive only as fragments. A fragment is a prefix constraint,
not a match.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .models import BankLine, Evidence

# build_narration (generate_dataset.py:184) picks one of six reference forms:
#   INV-000068 | INV68 | 000068 | 68 | REF0068 | (omitted)
# and templates additionally interpolate {ref_short}, the last four digits.
# Every one of those is the same number with different padding and prefixing,
# so int() normalization collapses them onto a single invoice id.
_REF_MARKED = re.compile(r"(?:INV|REF)-?(\d{1,6})(?!\d)")
# A bare run is the {ref} or {ref_short} field standing alone. Four digits is the
# shortest safe width: the lookbehind rejects IFSC interiors like HDFC0001234,
# and anything under four digits collides with ordinary numbers in narration.
_REF_BARE = re.compile(r"(?<![A-Za-z0-9_-])(\d{4,6})(?!\d)")
# "DEPOSIT 41224" (build_messy_narration) is an amount, not a reference.
_AMOUNT_WORDS = re.compile(r"\b(?:DEPOSIT|AMT|AMOUNT|BAL|BALANCE)\s+\d+")


# Only the full "INV-000068-A" form survives for a split leg; every short
# variant collapses to A / INVA / REFA and carries no invoice number at all.
_SPLIT_SUFFIX = re.compile(r"INV-?\d{1,6}-([AB])(?![A-Z0-9])")
# All six PROCESSOR_NARRATION_TEMPLATES (generate_dataset.py:125). Matched
# against the UPPERCASED narration, so "po_" has to be written "PO_".
_PAYOUT_REF = re.compile(r"(?:RPY-?|PO_|ST-|SETTLEMENT-|PAY-|PAY)(\d{4,6})")
_CURRENCY = re.compile(r"\b(USD|EUR|GBP)\b")
_RAILS = ("NEFT", "IMPS", "RTGS", "UPI")
_PROCESSORS = ("RAZORPAY", "STRIPE", "CASHFREE")

_SUFFIXES = (
    "PRIVATE LIMITED", "PVT LTD", "PVT", "LIMITED", "LTD",
    "CORP", "INDIA", "LLC", "PLC", "GROUP", "AND SONS",
)


def _to_invoice_id(digits: str) -> str | None:
    """Normalize any reference variant to a canonical invoice id."""
    number = int(digits)
    return f"INV-{number:06d}" if number > 0 else None


# Operating outflows (generate_dataset.py:144). These are neither matchable nor
# exceptions: score.py drops them from both sets, so one written to
# predicted_exceptions.csv is a pure false positive. All of them are debits.
#
# "ACCT MAINT FEE" is deliberately included even though EXCEPTION_KINDS can also
# emit it as a bank_charge. The generator produces one a month as an operating
# cost, so treating it as out of scope risks a single exception false negative
# while excluding it guarantees several false positives.
_OPERATING = re.compile(
    r"PAYROLL|SALARY PAYOUT|PREMISES LEASE|\bRENT-|GST PAYMENT|GSTR3B"
    r"|VENDOR PAYMENT|UTILITIES|ACCT MAINT FEE"
)


def is_out_of_scope(line: BankLine) -> bool:
    """True for operating outflows: payroll, rent, tax remittance, vendors.

    Debit-gated on purpose. A customer credit must never be dropped, and the
    narrations above only ever appear on outgoing money.
    """
    if line.amount_signed >= 0:
        return False
    return bool(_OPERATING.search(line.description.upper()))


def normalize_name(name: str) -> str:
    """Uppercase, strip punctuation and corporate suffixes, collapse whitespace."""
    s = name.upper()
    s = re.sub(r"\bM/S\b", " ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if s.endswith(" " + suf) or s == suf:
                s = s[: -len(suf)].strip()
                changed = True
    return s


def name_similarity(a: str, b: str) -> float:
    """Similarity robust to space removal and truncation.

    name_variant() emits name.replace(" ", ""), and narration truncates the
    company field, so we compare space-stripped forms and also score a
    prefix relationship generously.
    """
    na, nb = normalize_name(a).replace(" ", ""), normalize_name(b).replace(" ", "")
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= 6 and longer.startswith(shorter):
        return 0.97
    # Narration truncates the company field at both ends, so the surviving
    # fragment is often an infix rather than a prefix ("GADE AND SHER" out of
    # "KONDA GADE AND SHERE"). Weaker evidence than a prefix, but decisive
    # enough to separate a decoy pair, which nothing else distinguishes.
    if len(shorter) >= 6 and shorter in longer:
        return 0.93
    return SequenceMatcher(None, na, nb).ratio()


def _counterparty(description: str) -> str:
    """Best-effort company fragment: the longest alphabetic run that is not a
    rail, a processor, an IFSC code, or a reference."""
    cleaned = re.sub(r"INV-?\d+", " ", description.upper())
    # Truncation can cut a reference down to a bare "INV" / "INV-" tail. Left
    # in place it corrupts the counterparty, which is the only thing that
    # separates a decoy pair once the reference itself is gone.
    # A split leg also degrades its reference to a lone "A"/"B" or "INVA"/"REFB"
    # ("NEFT CR CHAUDRY, CHAHAL AND  REF B"), and that stray letter is enough to
    # drag a real payer name below the split-pairing threshold.
    cleaned = re.sub(r"\b(?:INV|REF)-?[AB]?\b", " ", cleaned)
    cleaned = re.sub(r"(?<![A-Z])[AB](?![A-Z])", " ", cleaned)
    cleaned = re.sub(r"[A-Z]{4}0\d{6}", " ", cleaned)  # IFSC
    # Punctuation stays a delimiter. Folding commas or hyphens into the name
    # instead was measured and is worse: the NEFT/RTGS templates use them as
    # field separators, and merging those fields costs decoy accuracy. The
    # leading fragment that survives ("CHOUDHURY") is a prefix of the full
    # name, which name_similarity already scores at 0.97.
    parts = re.split(r"[^A-Z ]+", cleaned)
    best = ""
    for part in parts:
        tokens = [
            t for t in part.split()
            if t not in _RAILS and t not in _PROCESSORS
            and t not in ("CR", "DR", "TRF", "FROM", "PAYOUT", "PAYMENT",
                          "SETTLEMENT", "TRANSFER", "REF", "P2M", "INR", "DEPOSIT",
                          "CREDIT", "USD", "EUR", "GBP")
        ]
        candidate = " ".join(tokens).strip()
        if len(candidate) > len(best):
            best = candidate
    return normalize_name(best)


def extract_evidence(line: BankLine) -> Evidence:
    desc = line.description.upper()

    # Blank out amounts before reading references, so "DEPOSIT 41224" cannot
    # contribute its own money as an invoice number.
    scannable = _AMOUNT_WORDS.sub(" ", desc)

    full: set[str] = set()
    for digits in _REF_MARKED.findall(scannable):
        resolved = _to_invoice_id(digits)
        if resolved:
            full.add(resolved)
    for digits in _REF_BARE.findall(scannable):
        resolved = _to_invoice_id(digits)
        if resolved:
            full.add(resolved)

    # A split leg's ref is "INV-000068-A", so ref_num is "A" and every short
    # variant degrades to A / INVA / REFA — no digits at all. Only the full form
    # identifies the invoice; the rest are recovered by amount and name.
    suffixes: set[str] = set()

    # An all-zero fragment is a reference truncated before its digits arrived
    # ("INV-00"). It carries no number, so it stays a startswith constraint.
    prefixes: set[str] = set()
    for frag in _REF_MARKED.findall(scannable):
        if int(frag) == 0 and frag not in suffixes:
            prefixes.add(f"INV-{frag}")

    rail = next((r for r in _RAILS if desc.startswith(r)), None)
    if rail is None and any(p in desc for p in _PROCESSORS):
        rail = "PROCESSOR"

    suffix_match = _SPLIT_SUFFIX.search(desc)
    currency_match = _CURRENCY.search(desc)

    return Evidence(
        txn_id=line.txn_id,
        full_refs=tuple(sorted(full)),
        prefix_refs=tuple(sorted(prefixes)),
        # A complete reference already names the invoice; no suffix needed.
        suffix_refs=() if full else tuple(sorted(suffixes)),
        split_suffix=suffix_match.group(1) if suffix_match else None,
        payout_refs=tuple(sorted(set(_PAYOUT_REF.findall(desc)))),
        counterparty=_counterparty(line.description),
        currency_hint=currency_match.group(1) if currency_match else None,
        rail=rail,
    )
