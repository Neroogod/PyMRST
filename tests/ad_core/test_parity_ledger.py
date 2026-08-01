"""Integrity tests for the file-level MRST ad-core parity ledger."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "ad_core" / "parity.csv"
MRST = ROOT.parent / "mrst-2024a-Geomech" / "modules" / "ad-core"
VALID_PORT_STATUSES = {
    "ported",
    "ported-consolidated",
    "excluded-catalog",
    "excluded-platform",
    "excluded-external",
    "excluded-deprecated",
    "excluded-external-solver",
    "excluded-visualization",
    "excluded-benchmark-driver",
    "pending-review",
}
VALID_VALIDATION_STATUSES = {"validated-reference", "unvalidated"}


def _rows():
    with LEDGER.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_ad_core_ledger_covers_reference_tree():
    """The ledger must list every MRST file, and each one once.

    This used to assert `len(rows) == 345`, which is not what the name
    promises and breaks whenever the tree or the scan changes - it had been
    failing at 346 against a hardcoded 345, telling nobody anything. Comparing
    against the reference tree checks the property actually wanted, and cannot
    drift.
    """
    rows = _rows()
    paths = [row["mrst_path"] for row in rows]
    assert len(paths) == len(set(paths)), "a file is listed twice"

    if not MRST.is_dir():
        pytest.skip(f"the MRST reference tree is not at {MRST}")
    on_disk = {str(p.relative_to(MRST)) for p in MRST.rglob("*.m")}
    assert set(paths) == on_disk, (
        f"{len(on_disk - set(paths))} MRST files are unlisted, "
        f"{len(set(paths) - on_disk)} listed files are gone; "
        f"regenerate the ledger")


def test_ad_core_ledger_uses_known_statuses():
    rows = _rows()
    assert {row["port_status"] for row in rows} <= VALID_PORT_STATUSES
    assert {row["validation_status"] for row in rows} <= VALID_VALIDATION_STATUSES


def test_candidate_counterparts_exist():
    for row in _rows():
        for counterpart in filter(None, row["python_counterpart"].split("; ")):
            assert (ROOT / counterpart).is_file(), (row["mrst_path"], counterpart)
