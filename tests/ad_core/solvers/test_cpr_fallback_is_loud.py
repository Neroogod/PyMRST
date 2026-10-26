"""CPR must say when it is not running CPR (ad_core).

`CPRSolver.solve` falls back to a direct sparse factorisation whenever the
model does not carry what its pressure stage needs. The answer is correct; the
cost is not. Until now that happened in silence.

It mattered concretely. `TwoPhaseOilWaterModelAD` stores its grid as `.g`, and
the guard checked `hasattr(model, "G")` - so every CPR request on the model
that most of this port's two-phase work runs on became a SuperLU
factorisation. Profiled: 91.5% of a Newton step, scaling as n^1.71 where a
preconditioned solve is roughly linear.

The symptom was "CPR is not faster", which reads as a property of CPR rather
than as CPR never having run. A silent fallback does not just cost time, it
misattributes the cost.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from pymrst.modules.ad_core.solvers.linear_solvers import CPRSolver


def _system(n=6):
    A = sp.csr_matrix(np.diag(np.full(n, 4.0)) + np.diag(np.ones(n - 1), 1))
    return A, np.arange(1.0, n + 1)


def test_a_model_that_misses_the_contract_warns():
    """The defect: the caller meant to get CPR and did not."""
    A, b = _system()
    model = SimpleNamespace(g="grid", fluid=SimpleNamespace())
    with pytest.warns(RuntimeWarning, match="fell back to a direct solve"):
        CPRSolver().solve(A, b, model=model)


def test_passing_no_model_at_all_is_intentional_and_silent():
    """`solve_linear_system(A, b)` documents a direct solve, so warning there
    would put a RuntimeWarning on a deliberate call - and a noisy warning gets
    filtered, which is how a loud fallback becomes silent again."""
    A, b = _system()
    solver = CPRSolver()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = solver.solve(A, b, model=None)
    np.testing.assert_allclose(A @ x, b, rtol=1e-10)
    assert solver.last_fallback, "still recorded, just not shouted about"


def test_the_warning_names_the_missing_attribute():
    """"CPR was slower" is not actionable; "your model has .g, not .G" is."""
    A, b = _system()
    model = SimpleNamespace(g="grid", fluid=SimpleNamespace())

    with pytest.warns(RuntimeWarning) as caught:
        CPRSolver().solve(A, b, model=model)
    message = str(caught[0].message)

    assert ".G" in message
    assert "it has .g" in message, "say what the model does have"
    assert "disgas" in message and ".b" in message


def test_the_lowercase_grid_is_called_out_specifically():
    """The exact defect that hid this: `.g` against a probe for `.G`."""
    missing = CPRSolver._unmet_requirements(
        SimpleNamespace(g=object(), fluid=SimpleNamespace(disgas=False, vapoil=False, b=None)))
    assert len(missing) == 1
    assert "it has .g" in missing[0]


def test_a_model_meeting_the_contract_raises_no_warning(monkeypatch):
    """Drives `solve` rather than the helper.

    An earlier version asserted only on `_unmet_requirements`, so a mutation
    that warned unconditionally inside `solve` went uncaught - the test never
    reached the branch it was about.
    """
    assert CPRSolver._unmet_requirements(
        SimpleNamespace(G=object(),
                        fluid=SimpleNamespace(disgas=False, vapoil=False, b=None))) == []

    A, b = _system()
    solver = CPRSolver()
    monkeypatch.setattr(CPRSolver, "_unmet_requirements",
                        staticmethod(lambda model: []))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            solver.solve(A, b, model=SimpleNamespace())
        except Exception:
            pass                      # the real CPR path needs a real model
    assert not [w for w in caught if "fell back" in str(w.message)], (
        "a compliant model must not be told it fell back")


def test_the_fallback_still_returns_the_right_answer():
    """It is a performance fallback, not a correctness one."""
    A, b = _system()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = CPRSolver().solve(A, b, model=None)
    np.testing.assert_allclose(A @ x, b, rtol=1e-10)


def test_the_fallback_is_recorded_for_inspection():
    """So a caller can check after the fact rather than parsing warnings."""
    A, b = _system()
    solver = CPRSolver()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.solve(A, b, model=None)
    assert solver.last_fallback
    assert solver.last_iterations == 0, "a fallback runs no GMRES iterations"


def test_zero_iterations_is_the_signature_of_a_fallback():
    """How the silent version was eventually spotted: a real CPR solve cannot
    converge in zero GMRES iterations."""
    A, b = _system()
    solver = CPRSolver()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.solve(A, b, model=None)
    assert solver.last_iterations == 0
