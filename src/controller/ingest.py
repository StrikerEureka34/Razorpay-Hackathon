"""CSV -> models. Decimal conversion happens here and nowhere else."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from .models import BankLine, Invoice, Payout


@dataclass(frozen=True)
class Dataset:
    invoices: tuple[Invoice, ...]
    bank_lines: tuple[BankLine, ...]
    payouts: tuple[Payout, ...]


def _dec(value: str) -> Decimal:
    """Parse money. Goes through str so no float ever exists, even briefly."""
    text = (value or "0").strip()
    return Decimal(text if text else "0")


def _day(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_dataset(data_dir: str) -> Dataset:
    invoices = tuple(
        Invoice(
            invoice_id=r["invoice_id"], customer_id=r["customer_id"],
            customer_name=r["customer_name"], issue_date=_day(r["issue_date"]),
            due_date=_day(r["due_date"]), currency=r["currency"],
            amount_gross=_dec(r["amount_gross"]), tax_amount=_dec(r["tax_amount"]),
            terms=r["terms"], status=r["status"],
        )
        for r in _rows(os.path.join(data_dir, "invoices.csv"))
    )

    bank_lines = tuple(
        BankLine(
            txn_id=r["txn_id"], value_date=_day(r["value_date"]),
            posted_date=_day(r["posted_date"]), amount_signed=_dec(r["amount_signed"]),
            currency=r["currency"], description=r["description"],
            running_balance=_dec(r["running_balance"]),
        )
        for r in _rows(os.path.join(data_dir, "bank_statement.csv"))
    )

    payout_path = os.path.join(data_dir, "processor_payouts.csv")
    payouts: tuple[Payout, ...] = ()
    if os.path.exists(payout_path):
        payouts = tuple(
            Payout(
                payout_id=r["payout_id"], payout_date=_day(r["payout_date"]),
                gross=_dec(r["gross"]), fees=_dec(r["fees"]),
                refunds=_dec(r["refunds"]), chargebacks=_dec(r["chargebacks"]),
                net=_dec(r["net"]),
                charge_ids=tuple(c for c in r["charge_ids"].split("|") if c),
                invoice_refs=tuple(
                    c for c in (r.get("invoice_refs") or "").split("|") if c
                ),
            )
            for r in _rows(payout_path)
        )

    return Dataset(invoices=invoices, bank_lines=bank_lines, payouts=payouts)
