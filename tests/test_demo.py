from decimal import Decimal

from demo import split_misses


def test_wrong_exceptions_are_explained_as_unpaired_splits():
    """Every claimed exception that is wrong should be accounted for, not just counted.

    The demo asserts on stage that each false positive is one leg of a split
    payment. If the matcher improves or regresses, this is what catches the
    claim going stale.
    """
    pairs, wrong, _leaked = split_misses("data", "results")
    explained = {inv_id for inv_id, _inv, _legs, _rows in pairs}
    explained |= {txn for _i, _inv, legs, _rows in pairs for txn in legs}
    assert wrong, "expected the run to have some exception false positives to explain"
    assert explained == wrong, f"unexplained false positives: {sorted(wrong - explained)}"


def test_each_pair_sums_to_the_invoice_exactly():
    pairs, _wrong, _leaked = split_misses("data", "results")
    assert pairs
    for _inv_id, inv, _legs, leg_rows in pairs:
        total = sum(Decimal(r["amount_signed"]) for r in leg_rows)
        assert total == Decimal(inv["amount_gross"])


def test_no_operating_outflow_reaches_the_exception_list():
    """The single most expensive mistake available on this dataset."""
    _pairs, _wrong, leaked = split_misses("data", "results")
    assert leaked == set(), f"payroll or rent listed as exceptions: {sorted(leaked)}"
