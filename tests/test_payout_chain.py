"""processor_payouts.csv now names its invoices directly.

The README is explicit that this is the intended route: a settlement credit
carries only the payout id, so you join to the payout file, confirm against
`net`, then read `invoice_refs`. Bundled settlements cannot be solved any
other way. Processor narrations are never truncated, so the id always survives.
"""
import pandas as pd

from src.controller.extract import extract_evidence
from src.controller.indexes import build_payout_by_net, payouts_by_reference
from src.controller.ingest import load_dataset


def test_payouts_expose_their_invoice_refs():
    ds = load_dataset("data")
    assert ds.payouts
    for payout in ds.payouts:
        assert payout.invoice_refs, f"{payout.payout_id} has no invoice_refs"
        assert len(payout.invoice_refs) == len(payout.charge_ids)
        assert all(r.startswith("INV-") for r in payout.invoice_refs)


def test_every_settlement_line_resolves_to_its_payout():
    """The payout id in the narration is the join key and is never truncated."""
    ds = load_dataset("data")
    by_ref = payouts_by_reference(ds.payouts)
    gt = pd.read_csv("data/ground_truth.csv")
    settlement_reasons = {"processor_fee", "bundled_payout", "bundled_payout_with_fee"}

    settlement_txns = {
        t
        for r in gt.itertuples()
        if r.reason_code in settlement_reasons
        for t in (r.bank_txn_ids.split("|") if isinstance(r.bank_txn_ids, str) else [])
    }
    assert settlement_txns

    resolved = 0
    for line in ds.bank_lines:
        if line.txn_id not in settlement_txns:
            continue
        evidence = extract_evidence(line)
        if any(ref in by_ref for ref in evidence.payout_refs):
            resolved += 1
    assert resolved == len(settlement_txns), (
        f"only {resolved} of {len(settlement_txns)} settlement lines found their payout"
    )


def test_payout_net_confirms_the_bank_amount():
    """Confirm against net, as the README says, rather than trusting the id alone."""
    ds = load_dataset("data")
    by_ref = payouts_by_reference(ds.payouts)
    by_net = build_payout_by_net(ds.payouts)
    assert by_net

    for line in ds.bank_lines:
        for ref in extract_evidence(line).payout_refs:
            payout = by_ref.get(ref)
            if payout is None:
                continue
            # Per-charge fee rounding lets the aggregate drift a paisa per charge.
            drift = abs(abs(line.amount_signed) - payout.net)
            assert drift <= len(payout.charge_ids) * 1, (
                f"{line.txn_id} claims {payout.payout_id} but amounts differ by {drift}"
            )
