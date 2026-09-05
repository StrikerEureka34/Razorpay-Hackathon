"""Provider-agnostic adjudication.

The model never does arithmetic and never enumerates. It receives at most
five pre-computed candidates and returns a choice or abstains.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..models import BankLine, Candidate, Invoice


@dataclass(frozen=True)
class AdjudicationRequest:
    txn_id: str
    bank_line: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AdjudicationResponse:
    chosen_invoice_ids: frozenset[str] | None   # None == insufficient evidence
    reasoning: str
    model: str


class Adjudicator(Protocol):
    def choose(self, request: AdjudicationRequest) -> AdjudicationResponse: ...


def build_request(
    line: BankLine,
    candidates: list[Candidate],
    invoice_by_id: dict[str, Invoice],
) -> AdjudicationRequest:
    """Flatten a bank line and its candidates into a model-ready payload.

    Every number here is already computed. The model compares, it does not
    calculate.
    """
    payload = []
    for c in candidates:
        inv_id = next(iter(c.invoice_ids), "")
        invoice = invoice_by_id.get(inv_id)
        payload.append({
            "invoice_id": inv_id,
            "customer_name": invoice.customer_name if invoice else "",
            "invoice_amount": str(invoice.amount_gross) if invoice else "",
            "due_date": invoice.due_date.isoformat() if invoice else "",
            "deterministic_score": round(c.score, 3),
            "source": c.source,
            **{k: v for k, v in c.features.items()},
        })
    return AdjudicationRequest(
        txn_id=line.txn_id,
        bank_line={
            "amount": str(abs(line.amount_signed)),
            "value_date": line.value_date.isoformat(),
            "description": line.description,
        },
        candidates=tuple(payload),
    )
