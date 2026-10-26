"""CPR's pressure stage must actually call an injected `elliptic_solver`.

`CPRSolverADAD`/`CPRSolver` declared an `elliptic_solver` slot - documented in
`amgcl_solver_ad.py` as "the Python-side equivalent of MRST's
`AMGCL_CPRSolverAD`" - but nothing in the real solve path
(`CPRSolver.solve` -> `_elliptic_solver`) ever read it: only its own
isolated unit tests called `apply_two_stage_preconditioner` directly. A
simulation that set `cpr.elliptic_solver = AMGCLSolverAD()` and ran through
`NonLinearSolver` would silently keep using the built-in pyamg/spilu stage.

These tests pin the fix: an injected solver is actually invoked by
`CPRSolver.solve`, both when passed to the constructor and when assigned to
`CPRSolverADAD.elliptic_solver` after construction, and the solve remains
correct. A spy proves the call happens (a solver that runs and is never
called would pass a residual check as easily as one that runs correctly).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import TwoPhaseOilWaterModelAD
from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverADAD
from pymrst.modules.ad_core.solvers.linear_solvers import CPRSolver
from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.adi import combine_equations, init_variables_adi


def _fluid():
    return init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                 n=[2.0, 2.0], c=[1e-10, 1e-9], p_ref=2e7)


def _model(nx=6, ny=6, nz=2):
    g = compute_geometry(cart_grid([nx, ny, nz], [nx*10.0, ny*10.0, nz*10.0]))
    rock = make_rock(g, 100e-15, 0.2)
    wells = add_well(None, g, rock, np.array([0]), well_type='rate', val=1e-3,
                     compi=np.array([1.0, 0.0]), name='I')
    wells = add_well(wells, g, rock, np.array([g.cells.num - 1]),
                     well_type='bhp', val=1.5e7,
                     compi=np.array([0.0, 1.0]), name='P')
    return TwoPhaseOilWaterModelAD(g, rock, _fluid(), wells)


def _system(model):
    state = model.initial_state(2e7, 0.2)
    variables = init_variables_adi(state.pressure, state.sW, state.qWs,
                                   state.qOs, state.bhp)
    eq = combine_equations(*model.assemble(state, *variables, 86400.0))
    return eq.jac[0].tocsr(), -eq.val, state


class _SpyEllipticSolver:
    """Records every pressure-block system it is handed; solves it exactly
    so the outer CPR/GMRES loop still converges to a correct answer."""

    def __init__(self):
        self.calls = 0
        self.shapes = []

    def solve_linear_system(self, A, b):
        self.calls += 1
        self.shapes.append(A.shape)
        return spla.spsolve(A.tocsc(), b)


def test_constructor_injected_elliptic_solver_is_actually_called():
    model = _model()
    A, b, state = _system(model)
    spy = _SpyEllipticSolver()
    solver = CPRSolver(elliptic_solver=spy)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = solver.solve(A, b, model, {"pressure": state.pressure})

    assert spy.calls > 0, "CPRSolver.solve never invoked the injected solver"
    residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    assert residual < 1e-6, f"CPR returned a non-solution ({residual:.2e})"


def test_post_construction_assignment_reaches_the_real_solve_path():
    """`cpr.elliptic_solver = spy` after construction, not just at
    __init__, must change what CPRSolverADAD.solve_linear_problem actually
    runs - the property must write through to the backend, not shadow it."""
    model = _model()
    A, b, state = _system(model)
    spy = _SpyEllipticSolver()

    cpr = CPRSolverADAD()
    assert cpr.elliptic_solver is None
    cpr.elliptic_solver = spy
    assert cpr.backend.elliptic_solver is spy

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = cpr.solve(A, b, model=model, state={"pressure": state.pressure})

    assert spy.calls > 0, "assigning cpr.elliptic_solver did not reach the backend"
    residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    assert residual < 1e-6, f"CPR returned a non-solution ({residual:.2e})"


def test_without_an_injected_solver_the_builtin_stage_still_runs():
    """The default path (no elliptic_solver) must be untouched - this is
    the ablation that shows the new branch does not swallow the old one."""
    model = _model()
    A, b, state = _system(model)
    solver = CPRSolver()
    assert solver.elliptic_solver is None

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = solver.solve(A, b, model, {"pressure": state.pressure})

    residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    assert residual < 1e-6, f"CPR returned a non-solution ({residual:.2e})"


def test_amgcl_as_the_real_elliptic_solver_matches_the_default_stage():
    """End to end with the actual AMGCL binding, not a spy: the pairing
    `amgcl_solver_ad.py` documents, exercised for real."""
    pyamgcl = pytest.importorskip("pyamgcl")
    from pymrst.modules.ad_core.solvers.amgcl_solver_ad import AMGCLSolverAD

    model = _model()
    A, b, state = _system(model)

    amgcl = AMGCLSolverAD(tolerance=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = CPRSolver(elliptic_solver=amgcl).solve(
            A, b, model, {"pressure": state.pressure})

    assert amgcl.last_iterations >= 1, (
        "AMGCLSolverAD reported no iterations - it was not really called")
    residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    assert residual < 1e-6, f"CPR returned a non-solution ({residual:.2e})"
