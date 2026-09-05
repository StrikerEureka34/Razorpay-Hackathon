"""Operating outflows are neither matchable nor exceptions.

generate_dataset.py:144 emits 101 of them (payroll, rent, GST remittance,
vendor payments, utilities, account fees). score.py excludes them from the
matchable set AND from the exception set, so every one written to
predicted_exceptions.csv is a straight false positive. The README puts the
cost plainly: dumping them drops exception precision to 24%.
"""
import pandas as pd

from src.controller.extract import is_out_of_scope
from src.controller.ingest import load_dataset


def _labels(data_dir):
    gt = pd.read_csv(f"{data_dir}/ground_truth.csv")
    out = {}
    for r in gt.itertuples():
        cell = r.bank_txn_ids if isinstance(r.bank_txn_ids, str) else ""
        for txn_id in filter(None, cell.split("|")):
            out[txn_id] = r.match_type
    return out


def test_operating_outflows_are_recognised_on_the_dev_seed():
    ds = load_dataset("data")
    labels = _labels("data")
    tp = fp = fn = 0
    for line in ds.bank_lines:
        predicted = is_out_of_scope(line)
        actual = labels.get(line.txn_id) == "out_of_scope"
        tp += predicted and actual
        fp += predicted and not actual
        fn += (not predicted) and actual
    assert fp == 0, f"{fp} in-scope lines wrongly dropped as operating outflows"
    assert fn == 0, f"{fn} operating outflows missed"
    assert tp == 101


def test_a_genuine_bank_charge_is_not_an_operating_outflow():
    """BANK CHARGES and NACH DR are real exceptions and must survive."""
    ds = load_dataset("data")
    labels = _labels("data")
    for line in ds.bank_lines:
        if labels.get(line.txn_id) == "unmatchable":
            assert not is_out_of_scope(line), (
                f"{line.txn_id} {line.description!r} dropped, but it is an exception"
            )


def test_no_customer_credit_is_ever_dropped():
    """Operating outflows are debits. Dropping a credit would lose a match."""
    ds = load_dataset("data")
    for line in ds.bank_lines:
        if line.amount_signed > 0:
            assert not is_out_of_scope(line), f"{line.txn_id} is a credit"
