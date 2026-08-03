"""The plan's summary line is a claim about the ledger, so it is checked.

The header of `new_improvement_implementation.md` states how many work packages
sit at each status. That number was hand-maintained and drifted six packages
behind `implementation.csv` before anyone noticed - which is the specific way a
progress ledger stops being evidence and becomes decoration.

These tests are cheap and they are not about formatting. A status count that
overstates progress is the one kind of documentation error that changes what
gets worked on next.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/"new_improvement_implementation.md"
LEDGER = ROOT/"docs"/"research_studio"/"implementation.csv"

# Section 18 of the plan. A status outside this set means the ledger and the
# document it belongs to have diverged on their own vocabulary.
STATUSES = {"planned", "designing", "implementing", "validating", "accepted",
            "blocked", "experimental"}


def _rows() -> list[list[str]]:
    with LEDGER.open(newline="") as handle:
        return list(csv.reader(handle))[1:]


def test_the_headline_counts_match_the_ledger():
    """The exact drift that went unnoticed: 12/2/33 against a ledger at 13/8/26."""
    counts = Counter(row[3].strip() for row in _rows())
    header = PLAN.read_text()[:2000]
    stated = re.search(
        r"\*\*(\d+) accepted, (\d+) validating, (\d+) implementing, "
        r"(\d+) planned\*\*", header)
    assert stated, "the plan header no longer states counts in the expected form"

    for number, status in zip(stated.groups(),
                              ("accepted", "validating", "implementing", "planned")):
        assert int(number) == counts[status], (
            f"the plan header claims {number} {status} packages but the ledger "
            f"has {counts[status]}; the header is not maintained by hand - read "
            f"it out of the CSV"
        )


def test_the_stated_package_total_matches_the_ledger():
    rows = _rows()
    stated = re.search(r"\((\d+) work packages", PLAN.read_text()[:2000])
    assert stated, "the plan header no longer states a work-package total"
    assert int(stated.group(1)) == len(rows)


def test_every_status_is_one_the_plan_defines():
    unknown = {row[3].strip() for row in _rows()} - STATUSES
    assert not unknown, f"statuses outside section 18's vocabulary: {sorted(unknown)}"


def test_nothing_is_accepted_without_evidence_recorded():
    """Section 18's rule, enforced rather than trusted.

    'Do not mark a work package accepted until its stated artifacts and tests
    exist.' An accepted row with no unit-test evidence is that rule broken.
    """
    for row in _rows():
        if row[3].strip() == "accepted":
            assert row[5].strip() and row[5].strip().lower() != "none", (
                f"{row[1]} is marked accepted with no unit-test evidence"
            )


def test_the_acceptance_decision_agrees_with_the_status():
    """A row saying 'not accepted' while its status says accepted, or the
    reverse, is a ledger that reads differently depending on which column you
    believe."""
    for row in _rows():
        status, decision = row[3].strip(), row[11].strip().lower()
        if status == "accepted":
            assert not decision.startswith("not accepted"), (
                f"{row[1]}: status accepted, decision says not accepted")
        elif decision.startswith("accepted"):
            pytest.fail(f"{row[1]}: decision says accepted but status is {status}")
