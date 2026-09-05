from src.controller.adjudicator import AdjudicationRequest, AdjudicationResponse
from src.controller.adjudicator.cache import DiskCache
from src.controller.adjudicator.gemini import TieredAdjudicator


class FakeAdjudicator:
    def __init__(self, answer, name):
        self.answer, self.name, self.calls = answer, name, 0

    def choose(self, request):
        self.calls += 1
        return AdjudicationResponse(self.answer, "fake", self.name)


def _request():
    return AdjudicationRequest(
        txn_id="TXN-000001", bank_line={"amount": "1.00"},
        candidates=({"invoice_id": "INV-000001"},),
    )


def test_tier_two_is_not_called_when_tier_one_decides():
    flash = FakeAdjudicator(frozenset({"INV-000001"}), "flash")
    pro = FakeAdjudicator(frozenset({"INV-000002"}), "pro")
    resp = TieredAdjudicator(flash, pro).choose(_request())
    assert resp.model == "flash"
    assert pro.calls == 0


def test_abstention_escalates_to_tier_two():
    flash = FakeAdjudicator(None, "flash")
    pro = FakeAdjudicator(frozenset({"INV-000002"}), "pro")
    resp = TieredAdjudicator(flash, pro).choose(_request())
    assert resp.model == "pro"
    assert pro.calls == 1


def test_both_abstaining_returns_an_abstention():
    flash, pro = FakeAdjudicator(None, "flash"), FakeAdjudicator(None, "pro")
    assert TieredAdjudicator(flash, pro).choose(_request()).chosen_invoice_ids is None


def test_cache_prevents_a_second_call(tmp_path):
    from src.controller.adjudicator.gemini import CachedAdjudicator

    inner = FakeAdjudicator(frozenset({"INV-000001"}), "flash")
    cached = CachedAdjudicator(inner, DiskCache(str(tmp_path)))
    cached.choose(_request())
    cached.choose(_request())
    assert inner.calls == 1


def test_build_adjudicator_falls_back_to_null_without_a_key(monkeypatch):
    """The demo must survive a missing key: no exception, no network."""
    from src.controller.adjudicator.gemini import build_adjudicator

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    adj = build_adjudicator(use_llm=True)
    assert type(adj).__name__ == "NullAdjudicator"
    assert adj.choose(_request()).chosen_invoice_ids is None


def test_build_adjudicator_honours_no_llm(monkeypatch):
    from src.controller.adjudicator.gemini import build_adjudicator

    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    assert type(build_adjudicator(use_llm=False)).__name__ == "NullAdjudicator"
