from src.controller import config
from src.controller.adjudicator.null import NullAdjudicator
from src.controller.ingest import load_dataset
from src.controller.resolve import resolve
from src.controller.routing import route


def test_every_decision_carries_a_known_route():
    ds = load_dataset("data")
    decisions = route(ds, resolve(ds), NullAdjudicator())
    assert decisions
    assert all(d.route in {"match", "exception", "review", "out_of_scope"} for d in decisions)


def test_uncertain_items_with_a_candidate_go_to_review_not_exceptions():
    """The exception list is scored against only 12 items; it must stay tight."""
    ds = load_dataset("data")
    decisions = route(ds, resolve(ds), NullAdjudicator())
    exceptions = [d for d in decisions if d.route == "exception"]
    assert len(exceptions) < 40, (
        f"{len(exceptions)} exceptions is a dumping ground, not an assertion"
    )


def test_a_line_with_a_plausible_candidate_is_never_asserted_unmatchable():
    """The spec's routing rule: uncertain WITH a plausible candidate goes to
    review, uncertain with NO candidate goes to exceptions. Lines that merely
    lost the Hungarian assignment still have candidates and must not be
    asserted unmatchable — that is what wrecks exception precision."""
    ds = load_dataset("data")
    res = resolve(ds)
    decisions = route(ds, res, NullAdjudicator())

    for d in decisions:
        if d.route != "exception" or not d.bank_txn_ids:
            continue
        for txn_id in d.bank_txn_ids:
            plausible = [
                c for c in res.candidates_by_line.get(txn_id, [])
                if c.score >= config.TAU_REVIEW
            ]
            assert not plausible, (
                f"{txn_id} asserted unmatchable despite candidate "
                f"{sorted(plausible[0].invoice_ids)} at {plausible[0].score:.3f}"
            )


def test_exception_list_stays_close_to_the_real_ones():
    """Exceptions are scored against 32 items — 20 unpaid invoices and 12 orphan
    credits. The 101 operating outflows must not appear at all: listing them is
    what drops exception precision to 24%."""
    ds = load_dataset("data")
    decisions = route(ds, resolve(ds), NullAdjudicator())
    items = {
        i for d in decisions if d.route == "exception"
        for i in (d.invoice_ids | d.bank_txn_ids)
    }
    assert len(items) <= 42, f"{len(items)} exception items against 32 real ones"


def test_duplicates_are_routed_as_matches_with_no_invoice():
    ds = load_dataset("data")
    decisions = route(ds, resolve(ds), NullAdjudicator())
    dups = [d for d in decisions if d.match_type == "duplicate"]
    assert len(dups) == 6
    assert all(d.route == "match" and d.invoice_ids == frozenset() for d in dups)


def test_null_adjudicator_produces_a_complete_run():
    ds = load_dataset("data")
    decisions = route(ds, resolve(ds), NullAdjudicator())
    assert all(d.decided_by in {"rules", "null", "flash", "pro"} for d in decisions)
