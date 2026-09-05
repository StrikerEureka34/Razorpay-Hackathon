"""SHA-256 disk cache. Committed to the repo so the demo runs offline."""
from __future__ import annotations

import hashlib
import json
import os

from . import AdjudicationRequest, AdjudicationResponse


class DiskCache:
    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def key(self, request: AdjudicationRequest) -> str:
        canonical = json.dumps(
            {
                "txn_id": request.txn_id,
                "bank_line": request.bank_line,
                "candidates": list(request.candidates),
            },
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, request: AdjudicationRequest) -> str:
        return os.path.join(self.directory, f"{self.key(request)}.json")

    def get(self, request: AdjudicationRequest) -> AdjudicationResponse | None:
        path = self._path(request)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        chosen = raw.get("chosen_invoice_ids")
        return AdjudicationResponse(
            chosen_invoice_ids=frozenset(chosen) if chosen else None,
            reasoning=raw.get("reasoning", ""),
            model=raw.get("model", "cached"),
        )

    def put(self, request: AdjudicationRequest, response: AdjudicationResponse) -> None:
        with open(self._path(request), "w", encoding="utf-8") as fh:
            json.dump({
                "chosen_invoice_ids": (
                    sorted(response.chosen_invoice_ids)
                    if response.chosen_invoice_ids else None
                ),
                "reasoning": response.reasoning,
                "model": response.model,
            }, fh, indent=2, sort_keys=True)
