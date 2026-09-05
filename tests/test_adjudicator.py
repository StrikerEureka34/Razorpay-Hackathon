from src.controller.adjudicator import AdjudicationRequest, AdjudicationResponse
from src.controller.adjudicator.cache import DiskCache
from src.controller.adjudicator.null import NullAdjudicator


def _request() -> AdjudicationRequest:
    return AdjudicationRequest(
        txn_id="TXN-000001",
        bank_line={"amount": "1000.00", "date": "2025-04-01", "description": "NEFT ACME"},
        candidates=(
            {"invoice_id": "INV-000001", "amount": "1000.00", "name_similarity": 0.9},
            {"invoice_id": "INV-000002", "amount": "1000.00", "name_similarity": 0.4},
        ),
    )


def test_null_adjudicator_always_abstains():
    resp = NullAdjudicator().choose(_request())
    assert resp.chosen_invoice_ids is None
    assert resp.model == "null"


def test_cache_key_is_stable_across_dict_ordering(tmp_path):
    cache = DiskCache(str(tmp_path))
    a = _request()
    b = AdjudicationRequest(
        txn_id="TXN-000001",
        bank_line={"description": "NEFT ACME", "date": "2025-04-01", "amount": "1000.00"},
        candidates=a.candidates,
    )
    assert cache.key(a) == cache.key(b)


def test_cache_round_trips_a_response(tmp_path):
    cache = DiskCache(str(tmp_path))
    req = _request()
    assert cache.get(req) is None
    resp = AdjudicationResponse(
        chosen_invoice_ids=frozenset({"INV-000001"}),
        reasoning="reference matches", model="test",
    )
    cache.put(req, resp)
    again = cache.get(req)
    assert again is not None
    assert again.chosen_invoice_ids == frozenset({"INV-000001"})
    assert again.model == "test"


def test_cache_miss_for_a_different_request(tmp_path):
    cache = DiskCache(str(tmp_path))
    cache.put(_request(), AdjudicationResponse(None, "abstain", "test"))
    other = AdjudicationRequest(
        txn_id="TXN-000999", bank_line={"amount": "1.00"}, candidates=(),
    )
    assert cache.get(other) is None
