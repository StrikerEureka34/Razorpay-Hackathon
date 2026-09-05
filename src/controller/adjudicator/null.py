"""Always abstains.

Does double duty: the rules-only ablation baseline, and the fallback when
the API key or network is unavailable at demo time. Because it is the
ablation baseline it is exercised on every test run, so the fallback path
is never untested code.
"""
from __future__ import annotations

from . import AdjudicationRequest, AdjudicationResponse


class NullAdjudicator:
    def choose(self, request: AdjudicationRequest) -> AdjudicationResponse:
        return AdjudicationResponse(
            chosen_invoice_ids=None,
            reasoning="no adjudicator configured",
            model="null",
        )
