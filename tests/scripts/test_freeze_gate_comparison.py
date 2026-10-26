"""The characterization gate must be able to fail, and only for real reasons.

Two of the eight frozen cases had been failing every run, for reasons that had
nothing to do with the code:

* `blackoil_one_phase` compared decision records with `!=`, and those records
  contain `wall_seconds` - a wall-clock measurement taken inside each linear
  solve. That comparison could never succeed. A gate that always fails is a
  gate nobody reads, and this one had been failing long enough that its output
  was treated as noise.
* `adjoint_uplift` froze `gradient_relative_error` - itself an error measure,
  2.4e-09 - and compared it *relatively* at rtol 1e-10. A run reporting
  2.46e-09 differs by 1.5% while making exactly the same claim: the adjoint
  agrees with finite differences. The frozen number was the wrong thing to
  compare.

Fixing a gate is the most dangerous kind of fix, because "it passes now" is
indistinguishable from "it stopped checking". So the tests that matter here are
the ones proving it still catches real changes, and the ones pinning *what* is
excluded so the exclusion cannot quietly grow.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BASELINES = REPO/"docs"/"research_studio"/"characterization"

sys.path.insert(0, str(REPO))
from scripts.freeze_characterization import (  # noqa: E402
    BOUNDED_METRICS, FINITE_DIFFERENCE_METRICS, FINITE_DIFFERENCE_RTOL,
    NON_DETERMINISTIC_FIELDS, RTOL, _strip_non_deterministic,
)


# -- what is excluded, pinned ----------------------------------------------

def test_only_unreproducible_fields_are_excluded_from_the_comparison():
    """The hole this closes: adding a real field here silently disables part
    of the gate.

    Confirmed by adding `iterations` to the set and watching a changed Newton
    iteration count stop being caught. Nothing else noticed. So the contents
    are pinned, and widening them is a deliberate edit to a test that says why.
    """
    assert NON_DETERMINISTIC_FIELDS == {"wall_seconds", "seconds", "timestamp"}

    for field in NON_DETERMINISTIC_FIELDS:
        assert "iter" not in field and "converged" not in field, (
            f"{field!r} looks like a solver decision, not a timing")


def test_the_bounds_and_tolerances_are_pinned_with_their_reasons():
    """A tolerance that can drift is a tolerance that will."""
    assert RTOL == 1e-10
    assert FINITE_DIFFERENCE_RTOL == 1e-6
    assert FINITE_DIFFERENCE_METRICS == {"finite_difference_gradient"}
    assert BOUNDED_METRICS == {"gradient_relative_error": 1e-6}


def test_stripping_removes_timings_and_nothing_else():
    record = [{"report": {"steps": [{
        "iterations": 3, "converged": True, "cuts": 0, "dt": 21600.0,
        "linear_solves": [{"iterations": 0, "converged": True,
                           "wall_seconds": 0.00053}],
    }]}}]
    stripped = _strip_non_deterministic(record)
    solve = stripped[0]["report"]["steps"][0]["linear_solves"][0]

    assert "wall_seconds" not in solve
    assert solve["iterations"] == 0 and solve["converged"] is True
    step = stripped[0]["report"]["steps"][0]
    assert step["iterations"] == 3 and step["cuts"] == 0 and step["dt"] == 21600.0


def test_stripping_is_stable_when_there_is_nothing_to_strip():
    record = [{"a": 1, "b": [2, 3], "c": {"d": True}}]
    assert _strip_non_deterministic(record) == record


# -- the gate still fails for real reasons ----------------------------------

def _check(case: str) -> bool:
    """True when the gate reports a failure for `case`."""
    result = subprocess.run(
        [sys.executable, "scripts/freeze_characterization.py", "--check", case],
        cwd=REPO, capture_output=True, text=True)
    return "FAIL" in result.stdout


@pytest.mark.slow
def test_a_changed_solver_decision_is_still_caught():
    """The check the timing fix could have broken."""
    path = BASELINES/"blackoil_one_phase.json"
    original = path.read_text()
    try:
        data = json.loads(original)
        data["decisions"][0]["report"]["steps"][0]["iterations"] = 99
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        assert _check("blackoil_one_phase"), (
            "a changed Newton iteration count must still fail the gate")
    finally:
        path.write_text(original)


@pytest.mark.slow
def test_a_changed_state_is_still_caught():
    path = BASELINES/"blackoil_one_phase.json"
    original = path.read_text()
    try:
        data = json.loads(original)
        label = sorted(data["states"])[0]
        field = sorted(data["states"][label])[0]
        values = data["states"][label][field]
        data["states"][label][field] = ([v*1.001 for v in values]
                                        if isinstance(values, list)
                                        else values*1.001)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        assert _check("blackoil_one_phase"), (
            "a 0.1% shift in a state must still fail the gate")
    finally:
        path.write_text(original)


@pytest.mark.slow
def test_the_bounded_metric_is_a_bound_not_a_formality():
    """`gradient_relative_error` is checked against a bound instead of a frozen
    value; the bound has to actually bind."""
    import scripts.freeze_characterization as freeze
    original = dict(freeze.BOUNDED_METRICS)
    try:
        freeze.BOUNDED_METRICS["gradient_relative_error"] = 1e-12
        # exercised through the module, not the subprocess, so the patched
        # bound is the one in force
        baseline = json.loads((BASELINES/"adjoint_uplift.json").read_text())
        failures = freeze.compare(baseline, baseline)
        assert any("exceeds its bound" in f for f in failures)
    finally:
        freeze.BOUNDED_METRICS.clear()
        freeze.BOUNDED_METRICS.update(original)
