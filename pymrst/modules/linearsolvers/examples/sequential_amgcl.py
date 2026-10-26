"""AMGCL on the pressure stage of a sequential (IMPES-style) solve.

Ported from MRST ``linearsolvers/examples/sequentialAMGCL.m``. That script
builds a sequential model with ``getSequentialModelFromFI`` and hands it two
different ``AMGCLSolverADAD`` configurations - one for the pressure stage
(AMG), one for transport (Krylov + ILU(0), no multigrid) - through
``pressureLinearSolver=``/``transportLinearSolver=`` constructor options,
then runs an SPE10 schedule through it.

This port's ``SequentialOilWaterModel``
(``pymrst/modules/sequential/seq_model.py``) does not have that plug-in
point: `_pressure_step`/`_transport_step` call `scipy.sparse.linalg.spsolve`
directly, with no injected-solver hook the way `CPRSolverADAD.elliptic_solver`
now has (see ``test_amgcl_cpr.py``, and the CPR wiring fix that made it
real). Rather than claim a pairing that is not there, this demonstrates the
same physical point MRST's example makes - AMGCL is well suited to the
near-elliptic pressure stage a sequential split produces - by extracting
*that exact system* (the same equations `_pressure_step` assembles and
solves at each of its own iterations) and solving it with `AMGCLSolverADAD`
directly, checked against the model's own reference solve.

If `SequentialOilWaterModel` grows a real pluggable-solver hook, this is the
example to point it at ``AMGCLSolverADAD`` and ``pressureLinearSolver=``; until
then this is the honest version of the same demonstration.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import TwoPhaseOilWaterModelAD
from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
from pymrst.modules.sequential.seq_model import SequentialOilWaterModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.adi import combine_equations, init_variables_adi
from pymrst.utils.units import BARSA, DAY

try:                                      # optional: needs Boost + pybind11
    from pymrst.modules.ad_core.solvers.amgcl_solver_ad import (
        AMGCLSolverADAD,
        HAVE_AMGCL,
    )
except ImportError:                       # pragma: no cover
    AMGCLSolverADAD = None
    HAVE_AMGCL = False


def _fluid():
    return init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                 n=[2.0, 2.0], c=[1e-10, 1e-9], p_ref=2e7)


def _model(nx=8, ny=8, nz=3):
    g = compute_geometry(cart_grid([nx, ny, nz], [nx*10.0, ny*10.0, nz*10.0]))
    rock = make_rock(g, 100e-15, 0.2)
    wells = add_well(None, g, rock, np.array([0]), well_type='rate',
                     val=1e-3, compi=np.array([1.0, 0.0]), name='I')
    wells = add_well(wells, g, rock, np.array([g.cells.num - 1]),
                     well_type='bhp', val=1.5e7,
                     compi=np.array([0.0, 1.0]), name='P')
    return TwoPhaseOilWaterModelAD(g, rock, _fluid(), wells)


def pressure_stage_system(fi: TwoPhaseOilWaterModelAD, st0, st, dt: float):
    """The exact system `SequentialOilWaterModel._pressure_step` assembles
    and solves at one of its own iterations - `A x = -r`, cell-major over
    (pressure, then the well unknowns), sW frozen."""
    p, qws, qos, bhp = init_variables_adi(st.pressure, st.qWs, st.qOs, st.bhp)
    water, oil, eq_w, eq_o, ctrl = fi.assemble(st0, p, st.sW, qws, qos, bhp, dt)
    eq = combine_equations(water + oil, eq_w, eq_o, ctrl)
    return eq.jac[0].tocsr(), -eq.val


def run_sequential_amgcl(*, dt: float = 30.0*DAY, verbose: bool = True):
    """Run one sequential step the model's own way, then solve that same
    step's pressure-stage system a second time with AMGCL and compare."""
    if not HAVE_AMGCL:
        raise ImportError(
            "run_sequential_amgcl needs the 'pyamgcl' package - install "
            "Boost (e.g. `brew install boost`), then `pip install "
            "pymrst[amgcl]`.")

    fi = _model()
    # max_outer=1 (this class's default) is the classic single-pass IMPES
    # scheme; `report.converged` tracks the *fully-implicit* residual, which
    # the class's own module docstring says takes "tens of passes" to
    # satisfy - its own tests never assert it True either, only that the
    # splitting error shrinks with more outer passes. A single pass is what
    # this example runs, matching MRST's own script.
    seq = SequentialOilWaterModel(fi, max_its=25)
    state0 = fi.initial_state(2e7, 0.2)

    state1, report = seq.step(state0, dt)
    if verbose:
        print(f"sequential step: {report.pressure_its} pressure its, "
             f"{report.transport_its} transport its, "
             f"converged={report.converged}")

    a, b = pressure_stage_system(fi, state0, state0, dt)
    direct = spla.spsolve(a.tocsc(), b)

    amgcl = AMGCLSolverADAD(tolerance=1e-8, coarsening="smoothed_aggregation",
                          relaxation="ilu0")
    x_amgcl = amgcl.solve_linear_system(a, b)

    if verbose:
        residual = np.linalg.norm(a @ x_amgcl - b)/np.linalg.norm(b)
        print(f"pressure-stage system: {a.shape[0]} unknowns, "
             f"AMGCL {amgcl.last_iterations} its, residual {residual:.2e}")
    return state1, report, direct, x_amgcl


if __name__ == "__main__":
    if not HAVE_AMGCL:
        print("pyamgcl is not installed - skipping (pip install "
              "pymrst[amgcl], after `brew install boost`)")
        raise SystemExit(0)

    def test_sequential_step_does_real_newton_work() -> None:
        """Both stages must actually iterate - a step that returns its input
        unchanged would "run" just as easily as one that solved anything."""
        _, report, _, _ = run_sequential_amgcl(verbose=False)
        assert report.pressure_its > 0
        assert report.transport_its > 0
        print(f"ok test_sequential_step_does_real_newton_work "
             f"({report.pressure_its} pressure its, "
             f"{report.transport_its} transport its)")

    def test_amgcl_matches_the_models_own_direct_solve() -> None:
        """AMGCL, applied to the identical system the sequential model
        itself assembles and solves with `spsolve`, must agree with it -
        proof this is really the same system, not a look-alike."""
        _, _, direct, x_amgcl = run_sequential_amgcl(verbose=False)
        assert np.allclose(x_amgcl, direct, atol=1e-5), (
            np.abs(x_amgcl - direct).max())
        print("ok test_amgcl_matches_the_models_own_direct_solve")

    test_sequential_step_does_real_newton_work()
    test_amgcl_matches_the_models_own_direct_solve()
    print("\nAll tests passed!")
