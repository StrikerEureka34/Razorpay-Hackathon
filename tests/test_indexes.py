from decimal import Decimal

import pandas as pd

from src.controller.extract import extract_evidence
from src.controller.indexes import (
    build_payout_by_net, candidates_for_line, find_splits,
)
from src.controller.ingest import load_dataset
from src.controller.scoring import invoice_amount_inr


def test_exact_amount_index_finds_the_clean_match():
    ds = load_dataset("data")
    by_net = build_payout_by_net(ds.payouts)
    hits = 0
    for line in ds.bank_lines:
        ev = extract_evidence(line)
        if not ev.full_refs:
            continue
        cands = candidates_for_line(line, ev, ds, by_net)
        target = ev.full_refs[0]
        if any(target in c.invoice_ids for c in cands):
            hits += 1
    assert hits > 70, f"reference-bearing lines producing the right candidate: {hits}"


def test_payout_chain_recovers_a_fee_deducted_line():
    """A processor_fee bank line equals payout.net; the invoice equals payout.gross."""
    ds = load_dataset("data")
    by_net = build_payout_by_net(ds.payouts)
    singles = [p for p in ds.payouts if len(p.charge_ids) == 1]
    assert singles
    payout = singles[0]
    line = next(
        b for b in ds.bank_lines if abs(b.amount_signed) == payout.net
    )
    cands = candidates_for_line(line, extract_evidence(line), ds, by_net)
    assert any(c.source == "payout_chain" for c in cands)


def test_fx_candidates_stay_inside_the_two_percent_band():
    ds = load_dataset("data")
    by_net = build_payout_by_net(ds.payouts)
    inv_by_id = {i.invoice_id: i for i in ds.invoices}
    for line in ds.bank_lines:
        for c in candidates_for_line(line, extract_evidence(line), ds, by_net):
            if c.source != "fx_band":
                continue
            inv = inv_by_id[next(iter(c.invoice_ids))]
            # The invoice is billed in foreign currency; compare in rupees.
            expected = invoice_amount_inr(inv)
            delta = abs(abs(line.amount_signed) - expected) / expected
            assert delta <= Decimal("0.02")
            assert inv.currency != "INR", "fx_band fired on a rupee invoice"


def test_find_splits_recovers_legs_that_kept_no_reference():
    """A split leg's ref is "INV-000053-A", so every short variant degrades to
    A / INVA / REFA and carries no invoice number. What survives is the payer
    name and the fact that the two legs sum to the invoice exactly."""
    ds = load_dataset("data")
    ev_by_txn = {b.txn_id: extract_evidence(b) for b in ds.bank_lines}
    splits = find_splits(ds, ev_by_txn)
    found = {next(iter(c.invoice_ids)) for c in splits}
    for invoice_id in ("INV-000053", "INV-000001", "INV-000189", "INV-000172"):
        assert invoice_id in found, f"{invoice_id} not recovered"


def test_find_splits_recovers_most_of_the_sixteen():
    ds = load_dataset("data")
    ev_by_txn = {b.txn_id: extract_evidence(b) for b in ds.bank_lines}
    gt = pd.read_csv("data/ground_truth.csv")
    truth = {
        (r.invoice_ids, frozenset(r.bank_txn_ids.split("|")))
        for r in gt[gt.reason_code == "split_payment"].itertuples()
    }
    got = {(next(iter(c.invoice_ids)), c.bank_txn_ids) for c in find_splits(ds, ev_by_txn)}
    correct = len(got & truth)
    assert correct >= 13, f"only {correct} of {len(truth)} splits recovered"
    assert len(got - truth) <= 2, f"{len(got - truth)} spurious split pairs"



def test_find_splits_never_pairs_legs_of_different_customers():
    """The partner search is constrained by name as well as by the exact sum;
    without that it would invent splits out of coincidental amounts."""
    from src.controller.extract import name_similarity

    ds = load_dataset("data")
    ev_by_txn = {b.txn_id: extract_evidence(b) for b in ds.bank_lines}
    inv_by_id = {i.invoice_id: i for i in ds.invoices}
    line_by_id = {b.txn_id: b for b in ds.bank_lines}

    for c in find_splits(ds, ev_by_txn):
        invoice = inv_by_id[next(iter(c.invoice_ids))]
        for txn_id in c.bank_txn_ids:
            ev = ev_by_txn[txn_id]
            names_match = name_similarity(ev.counterparty, invoice.customer_name) >= 0.9
            refs_invoice = invoice.invoice_id in ev.full_refs or any(
                invoice.invoice_id.endswith(s) for s in ev.suffix_refs
            )
            assert names_match or refs_invoice, (
                f"{txn_id} ({line_by_id[txn_id].description!r}) paired to "
                f"{invoice.invoice_id} on neither name nor reference"
            )


def test_find_splits_pairs_two_legs_summing_to_one_invoice():
    ds = load_dataset("data")
    ev_by_txn = {b.txn_id: extract_evidence(b) for b in ds.bank_lines}
    splits = find_splits(ds, ev_by_txn)
    assert len(splits) >= 10, f"expected most of the 16 splits, found {len(splits)}"
    inv_by_id = {i.invoice_id: i for i in ds.invoices}
    line_by_id = {b.txn_id: b for b in ds.bank_lines}
    for c in splits:
        assert len(c.bank_txn_ids) == 2
        inv = inv_by_id[next(iter(c.invoice_ids))]
        total = sum(abs(line_by_id[t].amount_signed) for t in c.bank_txn_ids)
        assert total == inv.amount_gross
