from decimal import Decimal

from src.controller.ingest import load_dataset


def test_loads_expected_volumes():
    ds = load_dataset("data")
    assert len(ds.invoices) == 200
    assert len(ds.bank_lines) == 301
    assert len(ds.payouts) == 30


def test_all_money_is_decimal_never_float():
    ds = load_dataset("data")
    for inv in ds.invoices:
        assert isinstance(inv.amount_gross, Decimal)
        assert not isinstance(inv.amount_gross, float)
    for bl in ds.bank_lines:
        assert isinstance(bl.amount_signed, Decimal)
    for p in ds.payouts:
        assert isinstance(p.gross, Decimal) and isinstance(p.net, Decimal)


def test_charge_ids_are_split_on_pipe():
    ds = load_dataset("data")
    multi = [p for p in ds.payouts if len(p.charge_ids) > 1]
    assert multi, "expected at least one bundled payout with several charges"
    for p in ds.payouts:
        assert all(c.startswith("CHG-") for c in p.charge_ids)


def test_dates_are_parsed_to_date_objects():
    ds = load_dataset("data")
    assert ds.invoices[0].due_date.year == 2025
    assert ds.bank_lines[0].value_date.year == 2025
