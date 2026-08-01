"""Integrity of the `pymrst/utils` parity ledger.

`docs/utils/parity.csv` is the authoritative record of what has been ported;
`UTILS_PARITY.md` is generated from it and `utils_implementation.md` quotes its
numbers. A ledger nobody checks drifts, and the direction of the drift is
always flattering - a path that has moved, a file deleted, a count edited by
hand to match a claim in prose.

These tests check the ledger against the filesystem rather than against itself.
The two that matter most:

- **every `ported` row must name a file that exists**, so a rename cannot
  silently turn a port into a claim;
- **every `ported` row must sit under `pymrst/utils`**, which is the layout
  contract this whole effort is about. `ported-elsewhere` exists precisely so
  that code filed somewhere else is counted as a deviation rather than as
  success, and this stops the two being confused.

The MRST tree lives outside the repository, so the row-count check skips when
it is absent rather than failing on a machine that has not got it.
"""

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "utils" / "parity.csv"
SUMMARY = ROOT / "UTILS_PARITY.md"
MRST = ROOT.parent / "mrst-2024a-Geomech" / "utils"

COLUMNS = {"mrst_path", "symbol", "subsystem", "port_status",
           "python_counterpart", "port_evidence", "validation_status",
           "validation_files", "notes"}
STATUSES = {"ported", "ported-elsewhere", "consolidated", "pending-review",
            "excluded-directory", "excluded-environment", "excluded-catalog"}
PRESENT = {"ported", "ported-elsewhere", "consolidated"}


@pytest.fixture(scope="module")
def rows():
    assert LEDGER.exists(), f"{LEDGER} is missing; run scripts/audit_utils_parity.py"
    with LEDGER.open() as handle:
        return list(csv.DictReader(handle))


def test_the_schema_is_what_the_tests_assume(rows):
    assert set(rows[0]) == COLUMNS, set(rows[0]) ^ COLUMNS


def test_every_status_is_known(rows):
    unknown = {r["port_status"] for r in rows} - STATUSES
    assert not unknown, f"unrecognised port_status values: {unknown}"


def test_no_mrst_file_is_listed_twice(rows):
    seen, repeated = set(), []
    for row in rows:
        if row["mrst_path"] in seen:
            repeated.append(row["mrst_path"])
        seen.add(row["mrst_path"])
    assert not repeated, repeated


def test_every_ported_file_exists(rows):
    missing = [(r["mrst_path"], path)
               for r in rows if r["port_status"] in PRESENT
               for path in r["python_counterpart"].split("; ") if path
               if not (ROOT / path).exists()]
    assert not missing, f"{len(missing)} counterparts do not exist: {missing[:5]}"


def test_every_present_row_names_a_counterpart(rows):
    empty = [r["mrst_path"] for r in rows
             if r["port_status"] in PRESENT and not r["python_counterpart"]]
    assert not empty, empty


def test_ported_means_at_an_mrst_shaped_path(rows):
    """The layout contract. `ported-elsewhere` is the status for the rest."""
    stray = [(r["mrst_path"], r["python_counterpart"]) for r in rows
             if r["port_status"] == "ported"
             and not r["python_counterpart"].startswith("pymrst/utils/")]
    assert not stray, f"'ported' rows outside pymrst/utils: {stray}"


def test_elsewhere_really_is_elsewhere(rows):
    """The converse: a row filed as a deviation must actually be one."""
    wrong = [(r["mrst_path"], r["python_counterpart"]) for r in rows
             if r["port_status"] == "ported-elsewhere"
             and r["python_counterpart"].startswith("pymrst/utils/")]
    assert not wrong, f"'ported-elsewhere' rows that are in place: {wrong}"


def test_nothing_missing_claims_a_counterpart(rows):
    claimed = [r["mrst_path"] for r in rows
               if r["port_status"] == "pending-review" and r["python_counterpart"]]
    assert not claimed, claimed


def test_every_validation_file_exists(rows):
    missing = [(r["mrst_path"], path) for r in rows
               for path in r["validation_files"].split("; ") if path
               if not (ROOT / path).exists()]
    assert not missing, f"{len(missing)} validation files do not exist: {missing[:5]}"


def test_validated_rows_name_their_evidence(rows):
    silent = [r["mrst_path"] for r in rows
              if r["validation_status"] == "validated-reference"
              and not r["validation_files"]]
    assert not silent, silent


def test_the_summary_agrees_with_the_ledger(rows):
    """`UTILS_PARITY.md` is generated; a hand edit would desynchronise it."""
    assert SUMMARY.exists(), f"{SUMMARY} is missing"
    text = SUMMARY.read_text()
    ported = sum(1 for r in rows if r["port_status"] == "ported")
    in_scope = sum(1 for r in rows
                   if not r["port_status"].startswith("excluded"))
    assert f"**{ported} of {in_scope} in-scope files sit at an MRST-shaped path.**" \
        in text, (f"the summary does not state {ported} of {in_scope}; "
                  f"regenerate it with scripts/audit_utils_parity.py")


def test_the_ledger_covers_every_mrst_file(rows):
    if not MRST.exists():
        pytest.skip(f"the MRST tree is not at {MRST}")
    on_disk = {str(p.relative_to(MRST)) for p in MRST.rglob("*.m")}
    listed = {r["mrst_path"] for r in rows}
    assert listed == on_disk, (
        f"{len(on_disk - listed)} MRST files are unlisted, "
        f"{len(listed - on_disk)} listed files are gone; "
        f"regenerate with scripts/audit_utils_parity.py")


def test_the_gridtools_work_is_finished_but_for_scaffolding(rows):
    """Phase 4's closing claim, kept honest.

    If a `gridtools` file other than MRST's own tests and examples turns up
    unported, the plan's status line is wrong and this says so.
    """
    left = {r["mrst_path"] for r in rows
            if r["subsystem"] == "gridtools"
            and r["port_status"] == "pending-review"}
    scaffolding = {"gridtools/tests/", "gridtools/examples/"}
    substantive = {p for p in left
                   if not any(p.startswith(s) for s in scaffolding)}
    assert not substantive, (
        f"utils_implementation.md claims Phase 4 is complete on substance, "
        f"but these are still unported: {sorted(substantive)}")
