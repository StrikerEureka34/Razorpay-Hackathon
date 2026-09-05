import csv
import json
import os

from src.controller.adjudicator.null import NullAdjudicator
from src.controller.emit import emit_all
from src.controller.ingest import load_dataset
from src.controller.resolve import resolve
from src.controller.routing import route


def _run(tmp_path):
    ds = load_dataset("data")
    decisions = route(ds, resolve(ds), NullAdjudicator())
    emit_all(decisions, ds, str(tmp_path), stats={"llm_calls": 0})
    return ds, decisions


def test_writes_every_expected_artifact(tmp_path):
    _run(tmp_path)
    for name in ("predicted_matches.csv", "predicted_exceptions.csv",
                 "review_queue.csv", "journal_entries.csv",
                 "audit_trail.jsonl", "run_report.md"):
        assert os.path.exists(tmp_path / name), name


def test_duplicate_rows_have_a_blank_invoice_ids_cell(tmp_path):
    """score.py parses "" to an empty frozenset; duplicates must match that."""
    _run(tmp_path)
    with open(tmp_path / "predicted_matches.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    dups = [r for r in rows if r["match_type"] == "duplicate"]
    assert dups
    assert all(r["invoice_ids"] == "" for r in dups)


def test_review_items_are_absent_from_the_scored_exception_file(tmp_path):
    _run(tmp_path)
    with open(tmp_path / "predicted_exceptions.csv", encoding="utf-8") as fh:
        exception_ids = {r["item_id"] for r in csv.DictReader(fh)}
    with open(tmp_path / "review_queue.csv", encoding="utf-8") as fh:
        review_ids = {r["item_id"] for r in csv.DictReader(fh)}
    assert not (exception_ids & review_ids)


def test_audit_trail_has_one_json_object_per_decision(tmp_path):
    _, decisions = _run(tmp_path)
    with open(tmp_path / "audit_trail.jsonl", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert len(records) == len(decisions)
    assert all("decided_by" in r and "rationale" in r for r in records)


def test_review_queue_is_ranked_by_amount_at_risk(tmp_path):
    _run(tmp_path)
    with open(tmp_path / "review_queue.csv", encoding="utf-8") as fh:
        amounts = [float(r["amount_at_risk"]) for r in csv.DictReader(fh)]
    assert amounts == sorted(amounts, reverse=True)


def test_no_bank_line_is_emitted_as_both_a_match_and_an_exception(tmp_path):
    """A line routed to matches must never also be asserted unmatchable."""
    _run(tmp_path)
    with open(tmp_path / "predicted_matches.csv", encoding="utf-8") as fh:
        matched = {
            t for r in csv.DictReader(fh)
            for t in r["bank_txn_ids"].split("|") if t
        }
    with open(tmp_path / "predicted_exceptions.csv", encoding="utf-8") as fh:
        excepted = {r["item_id"] for r in csv.DictReader(fh)}
    assert not (matched & excepted), sorted(matched & excepted)
