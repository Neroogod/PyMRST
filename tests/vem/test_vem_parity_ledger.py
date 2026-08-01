"""The `vem` parity ledger says what is true.

A generated ledger is only useful if it is regenerated. This checks that the
committed `VEM_PARITY.md` and `docs/vem/parity.csv` still match what the
auditor produces from the current tree, so a file that moves without the
ledger being refreshed is caught here rather than believed.

It also pins the classifications that carry meaning: nothing may be
`misplaced` (the whole point of Phases 1 and 2), and the one entry that is
still pending must be the one that is genuinely blocked.

The basename is deliberately distinct from `tests/ad_core/test_parity_ledger.py`
and `tests/ad_blackoil/test_parity_ledger.py`, which already collide with each
other and block whole-suite collection.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "VEM_PARITY.md"
CSV = REPO / "docs" / "vem" / "parity.csv"
AUDITOR = REPO / "scripts" / "audit_vem_parity.py"

# The only MRST file with no Python counterpart, and why.
BLOCKED = {"producerInjectorGrid"}


@pytest.fixture(scope="module")
def rows():
    if not CSV.exists():
        pytest.skip(f"{CSV.name} absent; run scripts/audit_vem_parity.py")
    with CSV.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def production(rows):
    return [r for r in rows if r["port_status"] != "excluded-catalog"]


def test_the_ledger_is_up_to_date(tmp_path):
    """Regenerating must not change the committed files."""
    before = (LEDGER.read_text(encoding="utf-8"), CSV.read_text(encoding="utf-8"))
    result = subprocess.run([sys.executable, str(AUDITOR)],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr
    after = (LEDGER.read_text(encoding="utf-8"), CSV.read_text(encoding="utf-8"))
    assert before[1] == after[1], "docs/vem/parity.csv is stale; regenerate it"
    assert before[0] == after[0], "VEM_PARITY.md is stale; regenerate it"


def test_every_mrst_file_is_accounted_for(rows):
    mrst = REPO.parent / "mrst-2024a-Geomech" / "modules" / "vem"
    if not mrst.is_dir():
        pytest.skip("MRST reference tree not available")
    assert len(rows) == len(list(mrst.rglob("*.m"))), len(rows)


def test_nothing_is_misplaced(rows):
    """Phases 1 and 2 exist to make this list empty and keep it empty."""
    misplaced = [r["mrst_path"] for r in rows if r["port_status"] == "misplaced"]
    assert misplaced == [], misplaced


def test_only_the_blocked_entry_is_pending(rows):
    pending = {r["symbol"] for r in rows if r["port_status"] == "pending-review"}
    assert pending == BLOCKED, pending


def test_the_blocked_entry_names_a_reason(rows):
    entry = next(r for r in rows if r["symbol"] in BLOCKED)
    assert entry["notes"], "a pending entry must say which phase retires it"


def test_every_ported_entry_names_a_python_file(rows):
    for row in production(rows):
        if row["port_status"] != "ported":
            continue
        for counterpart in row["python_counterpart"].split("; "):
            assert (REPO / counterpart).is_file(), counterpart


def test_every_ported_entry_is_validated(rows):
    """Code that exists and is never checked against MRST is not parity."""
    unvalidated = [r["mrst_path"] for r in production(rows)
                   if r["port_status"] == "ported"
                   and r["validation_status"] == "unvalidated"]
    assert unvalidated == [], unvalidated


def test_a_pending_entry_is_never_marked_validated(rows):
    """Code that does not exist cannot have been validated."""
    for row in rows:
        if row["port_status"] == "pending-review":
            assert row["validation_status"] == "unvalidated", row["mrst_path"]


def test_the_headline_count_matches_the_rows(rows):
    text = LEDGER.read_text(encoding="utf-8")
    ported = sum(1 for r in rows if r["port_status"] == "ported")
    total = len(production(rows))
    assert f"**{ported} of {total} production files" in text, (ported, total)
