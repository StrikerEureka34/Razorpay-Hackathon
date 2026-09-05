from src.controller.ingest import load_dataset
from src.controller.resolve import find_duplicates, resolve


def test_duplicate_pairs_keep_the_lower_txn_id_as_the_real_match():
    ds = load_dataset("data")
    dups = find_duplicates(ds)
    assert len(dups) == 6, f"expected 6 duplicate bank lines, found {len(dups)}"
    for c in dups:
        assert c.invoice_ids == frozenset()   # scored as a match with no invoice
        assert len(c.bank_txn_ids) == 1


def test_no_invoice_is_claimed_by_two_accepted_matches():
    ds = load_dataset("data")
    res = resolve(ds)
    seen: set[str] = set()
    for c in res.accepted:
        overlap = seen & c.invoice_ids
        assert not overlap, f"invoice claimed twice: {overlap}"
        seen |= c.invoice_ids


def test_no_bank_line_is_claimed_by_two_accepted_matches():
    ds = load_dataset("data")
    res = resolve(ds)
    seen: set[str] = set()
    for c in res.accepted:
        overlap = seen & c.bank_txn_ids
        assert not overlap, f"bank line claimed twice: {overlap}"
        seen |= c.bank_txn_ids


def test_resolution_accepts_the_large_majority_of_lines():
    ds = load_dataset("data")
    res = resolve(ds)
    assert len(res.accepted) > 140, f"only {len(res.accepted)} accepted"


def test_ambiguous_cases_are_surfaced_rather_than_guessed():
    ds = load_dataset("data")
    res = resolve(ds)
    for _line, cands in res.ambiguous:
        assert len(cands) >= 1
        assert len(cands) <= 5
