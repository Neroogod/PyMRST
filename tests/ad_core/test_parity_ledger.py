"""Integrity tests for the file-level MRST ad-core parity ledger."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "ad_core" / "parity.csv"
VALID_PORT_STATUSES = {
    "ported",
    "ported-consolidated",
    "excluded-catalog",
    "pending-review",
}
VALID_VALIDATION_STATUSES = {"validated-reference", "unvalidated"}


def _rows():
    with LEDGER.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_ad_core_ledger_covers_reference_tree():
    rows = _rows()
    assert len(rows) == 345
    paths = [row["mrst_path"] for row in rows]
    assert len(paths) == len(set(paths))


def test_ad_core_ledger_uses_known_statuses():
    rows = _rows()
    assert {row["port_status"] for row in rows} <= VALID_PORT_STATUSES
    assert {row["validation_status"] for row in rows} <= VALID_VALIDATION_STATUSES


def test_candidate_counterparts_exist():
    for row in _rows():
        for counterpart in filter(None, row["python_counterpart"].split("; ")):
            assert (ROOT / counterpart).is_file(), (row["mrst_path"], counterpart)
