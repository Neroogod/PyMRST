"""`TwoPhaseOilWaterModelAD` meets CPR's attribute contract (Phase 3 payoff).

Joining the AD-OO hierarchy put the model *in* the type graph; it did not make
it usable by the solvers written against that graph. `CPRSolver` reads
`model.G`, `model.n_phases`, `model.fluid.disgas`, `.vapoil` and `.b`, and this
model supplied none of them - so every CPR request became a silent SuperLU
factorisation. The measured cost of that silence, on 3D grids:

    n         LU        CPR      speed-up
    16k      133 ms    49 ms       2.7x
    196k   17181 ms   335 ms      51.4x

These tests pin the three things that make that reachable and keep it honest:
the contract is met, the *default* path is untouched, and the requirements list
does not understate itself.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import TwoPhaseOilWaterModelAD
from pymrst.modules.ad_core.solvers.linear_solvers import CPRSolver
from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.adi import combine_equations, init_variables_adi


def _fluid():
    return init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                 n=[2.0, 2.0], c=[1e-10, 1e-9], p_ref=2e7)


def _model(nx=6, ny=6, nz=2, **kwargs):
    g = compute_geometry(cart_grid([nx, ny, nz], [nx*10.0, ny*10.0, nz*10.0]))
    rock = make_rock(g, 100e-15, 0.2)
    wells = add_well(None, g, rock, np.array([0]), well_type='rate', val=1e-3,
                     compi=np.array([1.0, 0.0]), name='I')
    wells = add_well(wells, g, rock, np.array([g.cells.num - 1]),
                     well_type='bhp', val=1.5e7,
                     compi=np.array([0.0, 1.0]), name='P')
    return TwoPhaseOilWaterModelAD(g, rock, _fluid(), wells, **kwargs)


def _system(model):
    state = model.initial_state(2e7, 0.2)
    variables = init_variables_adi(state.pressure, state.sW, state.qWs,
                                   state.qOs, state.bhp)
    eq = combine_equations(*model.assemble(state, *variables, 86400.0))
    return eq.jac[0].tocsr(), -eq.val, state


# -- the contract -----------------------------------------------------------

def test_the_model_meets_cprs_contract():
    """The one assertion the whole phase was for."""
    assert CPRSolver._unmet_requirements(_model()) == []


def test_cpr_actually_runs_and_solves_the_system():
    """Contract satisfied is a claim about attributes; this is the claim about
    arithmetic. A fallback would also 'pass' the test above."""
    model = _model()
    A, b, state = _system(model)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any fallback warning fails the test
        x = CPRSolver().solve(A, b, model, {"pressure": state.pressure})

    residual = np.linalg.norm(A @ x - b)/np.linalg.norm(b)
    assert residual < 1e-6, f"CPR returned a non-solution ({residual:.2e})"


def test_the_requirements_list_is_exhaustive():
    """A requirements list that is not exhaustive is worse than none: a caller
    who satisfies it has been told they are done.

    `vapoil` was missing from the list while `_weighting` read it directly, so
    a fluid could meet every stated requirement and still raise one line in.
    """
    for attribute in ("disgas", "vapoil", "b"):
        fluid = _fluid()
        # Instance attribute shadowing the class one, set to a sentinel that
        # `hasattr` still sees - then deleted, to make the name truly absent.
        model = _model()
        model.fluid = type("Partial", (), {
            name: getattr(fluid, name)
            for name in ("disgas", "vapoil", "b", "b_w", "b_o")
            if name != attribute})()

        missing = CPRSolver._unmet_requirements(model)
        assert any(attribute in entry for entry in missing), (
            f"a fluid with no .{attribute} was reported as meeting the "
            f"contract; got {missing}")


def test_g_and_lowercase_g_cannot_disagree():
    """Two attributes holding one grid can be made to disagree, and a model
    whose G and g are different grids fails where nobody would look."""
    model = _model()
    assert model.G is model.g

    replacement = compute_geometry(cart_grid([2, 2, 1], [1.0, 1.0, 1.0]))
    model.G = replacement
    assert model.g is replacement, "setting G must move g with it"


def test_the_fluid_reports_two_phases_in_water_oil_order():
    """CPR multiplies weight i against equation block i, and `assemble`
    returns `(water, oil, ...)`. Reversed, the weights are right and applied
    to the wrong rows - wrong, but not obviously wrong."""
    fluid = _fluid()
    p = np.array([3.0e7])
    b = fluid.b(p)

    assert len(b) == 2
    np.testing.assert_allclose(np.asarray(b[0]).ravel(), fluid.b_w(p))
    np.testing.assert_allclose(np.asarray(b[1]).ravel(), fluid.b_o(p))
    # Distinct compressibilities, so the two are genuinely distinguishable and
    # a reversed order could not pass by coincidence.
    assert not np.allclose(np.asarray(b[0]).ravel(),
                           np.asarray(b[1]).ravel())


def test_the_fluid_accepts_mrsts_full_black_oil_signature():
    """Callers written against the black-oil fluid pass five arguments. This
    fluid ignores four of them, but must not refuse them."""
    fluid = _fluid()
    p = np.array([3.0e7])
    np.testing.assert_allclose(
        np.asarray(fluid.b(p, None, None, None, None)[0]).ravel(),
        np.asarray(fluid.b(p)[0]).ravel())


def test_the_contract_attributes_did_not_become_constructor_arguments():
    """They are bare class attributes on purpose. As dataclass fields they
    would join `__init__`/`__eq__`/`__repr__`, and every existing
    `ADFluid(...)` call would be answering a different question."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(_fluid())}
    assert "disgas" not in fields and "vapoil" not in fields


# -- the default path is untouched ------------------------------------------

def test_the_default_solver_is_still_a_direct_solve():
    """Defaulting to CPR would move every existing caller's answers by the
    linear solver's tolerance without anyone asking."""
    assert _model().linear_solver is None


def test_the_default_increment_is_bit_identical_to_spsolve():
    """Not 'close to' - the default path must be the same call it always was."""
    model = _model()
    A, b, state = _system(model)

    np.testing.assert_array_equal(model._solve_increment(_eq(A, b), state),
                                  spla.spsolve(A.tocsc(), b))


def test_an_injected_solver_is_used_instead():
    model = _model(linear_solver=CPRSolver())
    A, b, state = _system(model)

    x = model._solve_increment(_eq(A, b), state)
    residual = np.linalg.norm(A @ x - b)/np.linalg.norm(b)
    assert residual < 1e-6

    # And it is genuinely a different code path, not spsolve behind a name.
    assert not np.array_equal(x, spla.spsolve(A.tocsc(), b)), (
        "the injected solver returned the direct solution exactly, which "
        "means it was never called")


def test_an_injected_solver_still_reaches_convergence():
    """An inexact linear solve inside a Newton loop must not stall it."""
    model = _model(linear_solver=CPRSolver())
    state = model.initial_state(2e7, 0.2)
    _, converged, _ = model.step(state, 86400.0)
    assert converged


class _Eq:
    def __init__(self, A, b):
        self.jac = [A]
        self.val = -b


def _eq(A, b):
    return _Eq(A, b)
