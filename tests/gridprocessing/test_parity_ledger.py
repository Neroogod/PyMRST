"""Integrity tests for the file-level gridprocessing parity ledger."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "gridprocessing" / "parity.csv"
VALID_PORT_STATUSES = {
    "ported",
    "ported-consolidated",
    "excluded-catalog",
    "excluded-documentation",
    "excluded-deprecated",
    "excluded-matlab-compatibility",
    "test-fixture",
    "pending-review",
}
VALID_VALIDATION_STATUSES = {"validated-reference", "unvalidated"}


def _rows():
    with LEDGER.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_ledger_covers_the_reference_tree_once():
    rows = _rows()
    assert len(rows) == 69
    paths = [row["mrst_path"] for row in rows]
    assert len(paths) == len(set(paths))


def test_ledger_uses_only_reviewed_statuses():
    rows = _rows()
    assert {row["port_status"] for row in rows} <= VALID_PORT_STATUSES
    assert {row["validation_status"] for row in rows} <= VALID_VALIDATION_STATUSES


def test_every_counterpart_exists():
    for row in _rows():
        for counterpart in filter(None, row["python_counterpart"].split("; ")):
            assert (ROOT / counterpart).is_file(), (
                row["mrst_path"], counterpart
            )


def test_exclusions_and_fixtures_have_explanations():
    for row in _rows():
        if row["port_status"].startswith("excluded-") or (
            row["port_status"] == "test-fixture"
        ):
            assert row["port_evidence"], row["mrst_path"]
