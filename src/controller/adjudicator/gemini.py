"""Gemini adjudicator plus the caching and escalation wrappers.

Layering, outermost first:  Cached( Tiered( Flash, Pro ) )
so a cache hit costs nothing regardless of which tier originally answered.
"""
from __future__ import annotations

import json
import os

from .. import config
from . import AdjudicationRequest, AdjudicationResponse, Adjudicator
from .cache import DiskCache

_SYSTEM = """You are a reconciliation adjudicator for an accounts-receivable team.

You are given one bank transaction and a short list of candidate invoices.
Every number has already been computed for you. Do NOT perform arithmetic and
do NOT invent candidates outside the list.

Choose the single candidate the bank transaction settles, or return
insufficient_evidence if the evidence genuinely does not distinguish them.
Abstaining is the correct answer when two candidates are equally supported;
a wrong match is more costly than an abstention.

Weigh the evidence in this order:
1. An exact invoice reference in the narration is close to decisive.
2. A counterparty name match is strong when no reference survived truncation.
3. Date proximity is weak evidence and never decisive on its own.
4. An exact amount match is expected for most candidates and so rarely
   discriminates between them.
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_invoice_id": {"type": "string"},
        "insufficient_evidence": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["insufficient_evidence", "reasoning"],
}


class GeminiAdjudicator:
    """One tier. Wrap with CachedAdjudicator and TieredAdjudicator."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        from google import genai  # imported lazily so --no-llm needs no SDK

        self.model = model
        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    def choose(self, request: AdjudicationRequest) -> AdjudicationResponse:
        from google.genai import types

        prompt = (
            f"{_SYSTEM}\n\nBank transaction:\n"
            f"{json.dumps(request.bank_line, indent=2)}\n\n"
            f"Candidate invoices:\n{json.dumps(list(request.candidates), indent=2)}\n"
        )
        try:
            result = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=_SCHEMA,
                ),
            )
            parsed = json.loads(result.text)
        except Exception as exc:  # network, quota, schema drift
            return AdjudicationResponse(None, f"adjudicator error: {exc}", self.model)

        if parsed.get("insufficient_evidence") or not parsed.get("chosen_invoice_id"):
            return AdjudicationResponse(
                None, parsed.get("reasoning", ""), self.model
            )
        return AdjudicationResponse(
            frozenset({parsed["chosen_invoice_id"]}),
            parsed.get("reasoning", ""),
            self.model,
        )


class CachedAdjudicator:
    def __init__(self, inner: Adjudicator, cache: DiskCache) -> None:
        self.inner, self.cache = inner, cache
        self.hits = self.misses = 0

    def choose(self, request: AdjudicationRequest) -> AdjudicationResponse:
        cached = self.cache.get(request)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        response = self.inner.choose(request)
        self.cache.put(request, response)
        return response


class TieredAdjudicator:
    """Escalate to a stronger model only where the first tier abstains."""

    def __init__(self, primary: Adjudicator, escalation: Adjudicator) -> None:
        self.primary, self.escalation = primary, escalation
        self.escalations = 0

    def choose(self, request: AdjudicationRequest) -> AdjudicationResponse:
        first = self.primary.choose(request)
        if first.chosen_invoice_ids is not None:
            return first
        self.escalations += 1
        return self.escalation.choose(request)


def build_adjudicator(use_llm: bool, use_cache: bool = True) -> Adjudicator:
    """Assemble the ladder, degrading to NullAdjudicator if anything is missing."""
    from .null import NullAdjudicator

    if not use_llm or not os.environ.get("GEMINI_API_KEY"):
        return NullAdjudicator()
    try:
        tiered = TieredAdjudicator(
            GeminiAdjudicator(config.FLASH_MODEL),
            GeminiAdjudicator(config.PRO_MODEL),
        )
    except Exception:
        return NullAdjudicator()
    if not use_cache:
        return tiered
    return CachedAdjudicator(tiered, DiskCache(config.CACHE_DIR))
