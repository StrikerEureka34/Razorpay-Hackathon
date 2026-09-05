from decimal import Decimal

from src.controller.bundles import find_bundles, subset_sums
from src.controller.indexes import build_payout_by_net
from src.controller.ingest import load_dataset


def _pool(n: int) -> dict[str, Decimal]:
    """Irregular invoice-like amounts. Arithmetic progressions would produce
    thousands of colliding subset sums, which real invoice values do not."""
    import random

    rng = random.Random(4242)
    return {
        f"INV-{i:06d}": Decimal(f"{rng.randint(10_000, 999_999)}.{rng.randint(0, 99):02d}")
        for i in range(n)
    }


def test_subset_sums_finds_the_only_exact_combination():
    amounts = {
        "A": Decimal("100.00"), "B": Decimal("250.00"),
        "C": Decimal("400.00"), "D": Decimal("50.00"),
    }
    result = subset_sums(amounts, target=Decimal("750.00"), k=3)
    assert result == [{"A", "B", "C"}]


def test_subset_sums_returns_every_tie():
    amounts = {
        "A": Decimal("100.00"), "B": Decimal("200.00"),
        "C": Decimal("100.00"), "D": Decimal("200.00"),
    }
    result = subset_sums(amounts, target=Decimal("300.00"), k=2)
    assert len(result) == 4  # every A/C x B/D pairing


def test_subset_sums_respects_the_size_constraint():
    amounts = {"A": Decimal("100.00"), "B": Decimal("100.00"), "C": Decimal("200.00")}
    assert subset_sums(amounts, target=Decimal("200.00"), k=2) == [{"A", "B"}]


def test_subset_sums_agrees_with_brute_force_on_a_large_pool():
    """The meet-in-the-middle path must return exactly what an exhaustive
    scan would. C(40, 5) is 658k combinations, well past the brute-force
    threshold, so this exercises the optimized branch."""
    from itertools import combinations

    amounts = _pool(40)
    target = sum(
        (amounts[k] for k in ("INV-000003", "INV-000011", "INV-000019",
                              "INV-000027", "INV-000035")),
        Decimal("0"),
    )
    expected = [
        set(c) for c in combinations(sorted(amounts), 5)
        if sum((amounts[x] for x in c), Decimal("0")) == target
    ]
    got = subset_sums(amounts, target=target, k=5)
    assert sorted(map(sorted, got)) == sorted(map(sorted, expected))


def test_subset_sums_is_fast_enough_for_a_six_way_bundle():
    """C(56, 6) is 32M combinations; the full run budget is under 10s."""
    import time

    amounts = _pool(56)
    target = sum(
        (amounts[f"INV-{n:06d}"] for n in (2, 9, 16, 23, 30, 44)), Decimal("0")
    )
    started = time.time()
    got = subset_sums(amounts, target=target, k=6)
    elapsed = time.time() - started
    assert any(
        s == {f"INV-{n:06d}" for n in (2, 9, 16, 23, 30, 44)} for s in got
    ), "the planted bundle was not recovered"
    assert elapsed < 5.0, f"six-way subset-sum took {elapsed:.1f}s"


def test_bundle_survives_per_charge_fee_rounding_drift():
    """do_bundled sets the bank line to d(total - fee) — one rounding — while
    processor_payouts.csv aggregates k individually-rounded per-charge nets
    (generate_dataset.py:536-546). PAY-000025 lands a paisa apart from its own
    bank line, so an exact-equality net lookup drops the whole bundle."""
    ds = load_dataset("data")
    by_net = build_payout_by_net(ds.payouts)
    line_amounts = {abs(b.amount_signed) for b in ds.bank_lines}
    residual = {i.invoice_id for i in ds.invoices if i.amount_gross not in line_amounts}

    drifted = [p for p in ds.payouts if len(p.charge_ids) >= 2 and p.net not in line_amounts]
    assert drifted, "expected a payout whose net misses every bank line exactly"

    covered = {t for c in find_bundles(ds, residual, by_net) for t in c.bank_txn_ids}
    for payout in drifted:
        near = [
            b for b in ds.bank_lines
            if abs(abs(b.amount_signed) - payout.net) <= Decimal("0.01") * len(payout.charge_ids)
        ]
        assert near, f"no bank line near {payout.payout_id}"
        assert any(b.txn_id in covered for b in near), (
            f"{payout.payout_id} produced no bundle candidate"
        )


def test_find_bundles_recovers_the_multi_charge_payouts():
    ds = load_dataset("data")
    by_net = build_payout_by_net(ds.payouts)
    multi = [p for p in ds.payouts if len(p.charge_ids) >= 2]
    assert multi, "dataset should contain bundled payouts"

    # residual pool: invoices whose amount matches no single bank line exactly
    line_amounts = {abs(b.amount_signed) for b in ds.bank_lines}
    residual = {i.invoice_id for i in ds.invoices if i.amount_gross not in line_amounts}

    bundles = find_bundles(ds, residual, by_net)
    assert len(bundles) >= 2, f"expected to recover bundles, found {len(bundles)}"
    for c in bundles:
        assert len(c.invoice_ids) >= 2
        assert len(c.bank_txn_ids) == 1
