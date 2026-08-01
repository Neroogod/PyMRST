from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "compositional" / "parity.csv"


def _rows():
    with LEDGER.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_audit_regenerates_complete_structural_inventory():
    result = subprocess.run(
        [sys.executable, "scripts/audit_compositional_parity.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = _rows()
    assert len(rows) == 96
    assert sum(row["kind"] == "executable" for row in rows) == 82
    assert sum(row["kind"] == "catalog" for row in rows) == 14
    assert not [row for row in rows if row["status"] == "missing"]


def test_known_partial_rows_are_not_overclaimed():
    rows = {row["mrst_path"]: row for row in _rows()}
    expected = set()
    assert {path for path, row in rows.items() if row["status"] == "partial"} == expected
    assert rows["examples/compositionalValidationSimple.m"]["status"] == "broad"


def test_every_executable_owner_and_evidence_path_exists():
    for row in _rows():
        assert (ROOT / row["python_owner"]).is_file(), row["mrst_path"]
        if row["kind"] == "executable":
            assert row["validation_evidence"], row["mrst_path"]
            for evidence in row["validation_evidence"].split("; "):
                assert (ROOT / evidence).is_file(), evidence
