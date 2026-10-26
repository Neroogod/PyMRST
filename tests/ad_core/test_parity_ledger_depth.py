"""The ad-core parity ledger has to be able to fail (Phase 0).

Before the depth pass, `port_status: ported` meant "a file with a matching name
exists". Under that rule `PhysicalModel` - 4 of 49 methods - was ported, and so
was `UniformFacilityModel`, whose counterpart is a class statement and a
docstring. A ledger that cannot distinguish those from `SimpleWell` (21 of 22)
is not evidence; it is a filename listing with a reassuring column.

These tests pin the distinction, and one of them is a ratchet: the number of
shallow rows may fall and must never rise. That is what makes the ledger a gate
on the port rather than a description of it.

The MRST tree lives outside this repository, so the regeneration test skips
where it is absent rather than failing - a contributor without MATLAB must
still be able to run the suite.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO/"docs"/"ad_core"/"parity.csv"
MRST = REPO.parent/"mrst-2024a-Geomech"/"modules"/"ad-core"

STATUSES = {
    "ported", "ported-consolidated", "ported-shallow", "wrapper",
    "pending-review", "excluded-catalog", "excluded-platform",
    "excluded-external", "excluded-external-solver", "excluded-deprecated",
    "excluded-visualization", "excluded-benchmark-driver",
}

# Lower this as depth work lands. Never raise it: a rise means a class lost
# coverage, or that the bar was moved to accommodate one.
MAX_SHALLOW_OR_WRAPPER = 2


def _rows():
    with LEDGER.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _by_symbol():
    return {row["symbol"]: row for row in _rows()}


def test_the_ledger_separates_a_real_port_from_a_wrapper():
    """The test the whole exercise exists for.

    All four rows below were `ported` under the filename rule. Three of them
    should not have been, and `SimpleWell` shows the bar is reachable - without
    it this would only prove the scan is pessimistic.
    """
    rows = _by_symbol()

    assert rows["SimpleWell"]["port_status"] == "ported"
    assert float(rows["SimpleWell"]["depth"]) >= 0.95
    assert rows["FacilityModel"]["port_status"] == "ported"
    assert float(rows["FacilityModel"]["depth"]) >= 0.95

    # UniformFacilityModel was the motivating wrapper and now carries a real
    # override (Phase 2), so it is no longer one. Kept here as the second proof
    # that a row leaves this list by being fixed, not by moving the bar.
    assert rows["UniformFacilityModel"]["port_status"] != "wrapper"
    assert rows["ReservoirModel"]["port_status"] == "ported"

    # PhysicalModel was the motivating case and is now ported at full depth
    # (Phase 1). It stays in this test as the proof that a shallow row can be
    # cleared by porting rather than by moving the bar.
    assert rows["PhysicalModel"]["port_status"] == "ported"
    assert float(rows["PhysicalModel"]["depth"]) == 1.0


def test_no_class_is_called_ported_on_its_filename_alone():
    """The rule that makes `ported` mean something.

    Any classdef big enough to be judged by ratio must clear the depth bar to
    keep the label.
    """
    from scripts.audit_ad_core_parity import DEPTH_BAR, MIN_METHODS_FOR_DEPTH

    for row in _rows():
        if row["port_status"] != "ported" or row["mrst_kind"] != "classdef":
            continue
        total = int(row["mrst_methods"] or 0)
        if total < MIN_METHODS_FOR_DEPTH:
            continue
        assert float(row["depth"]) >= DEPTH_BAR, (
            f"{row['symbol']} is marked ported at depth {row['depth']}")


def test_the_depth_bar_itself_is_pinned():
    """A ratchet on the count alone can be satisfied by moving the bar.

    Dropping `DEPTH_BAR` to 0.05 empties the shallow list, and a test that only
    counts rows passes happily - which is the same move as widening a tolerance
    until a comparison succeeds. Found by making exactly that mutation and
    watching the ratchet below stay green.

    So the bar is pinned here. Changing it is legitimate, but it must be a
    deliberate edit to a test that says what the number is for, not a quiet
    constant change that makes a ledger look finished.
    """
    from scripts.audit_ad_core_parity import DEPTH_BAR, MIN_METHODS_FOR_DEPTH

    assert DEPTH_BAR == 0.6
    assert MIN_METHODS_FOR_DEPTH == 5


def test_per_method_exclusions_are_pinned_and_justified():
    """The third knob that can make a gap vanish.

    `EXCLUDED_METHODS` drops individual methods from the depth count. Left
    unpinned it is simply a way to reach depth 1.00 by deleting the evidence,
    so the contents are fixed here and every entry must carry a reason that
    says *what* is missing and *where the capability lives* - otherwise it
    reads as "PyMRST cannot do this", which in each of these cases is false.

    Three owners, and the grounds differ:

    * `DiagonalAutoDiffBackend` - MRST's MEX block-CSR storage. The algorithm
      is present; only the fixed-block-size memory layout is not, and the
      measured result of adopting it is recorded in the reason itself;
    * `DiagonalJacobian` - two methods that ask about positions *inside* that
      same `n x ndiag` column layout. `setDiagonal` survives the change of
      representation and is ported; these two have no counterpart against a
      triplet store;
    * `ResultHandler` - MATLAB `subsref` plumbing, whose Python equivalent
      (`__getitem__`) is implemented and carries the same branch.
    """
    from scripts.audit_ad_core_parity import EXCLUDED_METHODS

    assert set(EXCLUDED_METHODS) == {"DiagonalAutoDiffBackend",
                                     "DiagonalJacobian", "ResultHandler"}
    assert set(EXCLUDED_METHODS["DiagonalAutoDiffBackend"]) == {
        "getBlockSystemCSR", "applySchurComplementBlockSystemCSR", "getRHS"}
    assert set(EXCLUDED_METHODS["DiagonalJacobian"]) == {
        "compareIndices", "treatSubset"}
    assert set(EXCLUDED_METHODS["ResultHandler"]) == {"getData_private"}

    #: The accepted grounds, each with the word that must appear in a reason
    #: claiming it. Adding a fourth is a deliberate act, which is the point.
    grounds = ("mex", "platform", "layout", "subsref")

    for owner, methods in EXCLUDED_METHODS.items():
        for name, reason in methods.items():
            assert len(reason) > 40, (
                f"{owner}.{name} is excluded without a reason that survives "
                f"reading")
            assert any(g in reason.lower() for g in grounds), (
                f"{owner}.{name}: the accepted grounds are a platform/MEX "
                f"path, a storage layout, or MATLAB subsref plumbing")
            # An exclusion must also say where the capability lives.
            assert ("ported" in reason.lower()
                    or "implemented" in reason.lower()
                    or "pymrst" in reason.lower()), (
                f"{owner}.{name}: say whether the capability exists elsewhere, "
                f"not just that the MRST code cannot be transcribed")


def test_the_shallow_count_is_a_ratchet():
    """Progress may lower this; nothing may raise it."""
    shallow = [r for r in _rows()
               if r["port_status"] in ("ported-shallow", "wrapper")]
    assert len(shallow) <= MAX_SHALLOW_OR_WRAPPER, (
        f"{len(shallow)} shallow/wrapper rows, was {MAX_SHALLOW_OR_WRAPPER}: "
        f"{sorted(r['symbol'] for r in shallow)}")


def test_every_status_is_one_the_scanner_declares():
    unknown = {r["port_status"] for r in _rows()} - STATUSES
    assert not unknown, f"undeclared statuses: {sorted(unknown)}"


def test_every_demoted_row_says_why():
    """A bare 'shallow' sends the reader back to the source to re-derive it."""
    for row in _rows():
        if row["port_status"] in ("ported-shallow", "wrapper"):
            assert row["notes"].strip(), f"{row['symbol']} demoted with no reason"


def test_every_exclusion_carries_its_reason():
    """An exclusion without a stated reason is indistinguishable from an
    oversight, which is the whole difference between the two."""
    for row in _rows():
        if row["port_status"].startswith("excluded"):
            assert row["port_evidence"].strip(), (
                f"{row['symbol']} excluded with no reason given")


def test_absent_methods_are_named_not_just_counted():
    """The list is what makes a shallow row actionable."""
    for row in _rows():
        if row["port_status"] == "ported-shallow":
            assert row["methods_absent"].strip(), (
                f"{row['symbol']} is shallow but names no absent method")


@pytest.mark.skipif(not MRST.is_dir(), reason="MRST reference tree not present")
def test_the_committed_ledger_matches_a_fresh_scan(tmp_path):
    """Anti-staleness. Porting methods without regenerating leaves the ledger
    claiming a depth that is no longer true, which is the failure mode the
    plan-header count already had once."""
    fresh = tmp_path/"parity.csv"
    result = subprocess.run(
        [sys.executable, "scripts/audit_ad_core_parity.py",
         "--csv", str(fresh), "--markdown", str(tmp_path/"out.md")],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    with fresh.open(newline="", encoding="utf-8") as stream:
        regenerated = {r["mrst_path"]: r["port_status"]
                       for r in csv.DictReader(stream)}
    committed = {r["mrst_path"]: r["port_status"] for r in _rows()}
    drifted = {k: (committed[k], regenerated.get(k))
               for k in committed if committed[k] != regenerated.get(k)}
    assert not drifted, (
        f"docs/ad_core/parity.csv is stale; regenerate it: {drifted}")
