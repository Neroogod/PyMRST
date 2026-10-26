"""Validate MRST blackoilTutorialMexAcceleration: the port in
`ad_blackoil/examples/blackoil_tutorial_mex_acceleration.py`.

The reference is MRST 2024a on a 20x20x10 grid of the script's physical size
(`dump_blackoil_tutorial_mex_acceleration_reference.m`). MRST's MEX does not
build on this machine, so two things are compared, each as far as MRST runs:
- the tutorial run with a direct linear solver on both sides (MRST's diagonal
  backend with `useMex` false): Newton iterations and ministeps identical at
  every step, pressures, saturations, bhps and surface rates to `RUN_RTOL`;
- `AMGCL_CPRSolverAD`'s preparation of the first Newton system - everything
  MRST does in MATLAB before handing the system to AMGCL: the matrix, the
  right-hand side, the DRS row weights, the pressure scaling and the
  increment the prepared system gives, to `SYSTEM_RTOL`. The AMGCL solve
  itself is MRST's own C++ code (its `amgcl_matlab.cpp`, compiled for Python
  against the AMGCL it bundles), so with the same input it is MRST's solve.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RUN_RTOL = 1e-10
SYSTEM_RTOL = 1e-12
DIMS = (20, 20, 10)


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def run_check(ref) -> bool:
    from pymrst.modules.ad_blackoil.examples.blackoil_tutorial_mex_acceleration import (
        blackoil_tutorial_mex_acceleration,
    )
    from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
    out = blackoil_tutorial_mex_acceleration(plot=False, dims=DIMS, LinearSolver=BackslashSolverAD())
    its = np.array([r["Iterations"] for r in out.rep0["ControlstepReports"]])
    nmini = np.array([len(r["StepReports"]) for r in out.rep0["ControlstepReports"]])
    ok = report(np.array_equal(its, np.atleast_1d(ref["its"]).astype(int)),
                f"run: Newton iterations identical ({its.sum()}, MRST {np.sum(ref['its'])})")
    ok &= report(np.array_equal(nmini, np.atleast_1d(ref["nmini"]).astype(int)),
                 "run: ministeps identical")
    for name, actual, expected in (
            ("initial pressure", out.state0["pressure"], ref["p0"]),
            ("initial saturations", out.state0["s"], ref["s0"]),
            ("pressure", [s["pressure"] for s in out.states0], ref["p"]),
            ("saturations", [np.asarray(s["s"]).ravel(order="F") for s in out.states0], ref["s"]),
            ("bhp", [[w.bhp for w in row] for row in out.ws0], ref["bhp"]),
            ("qWs", [[w.qWs for w in row] for row in out.ws0], ref["qWs"]),
            ("qOs", [[w.qOs for w in row] for row in out.ws0], ref["qOs"])):
        err = _rel(actual, expected)
        ok &= report(err < RUN_RTOL, f"run: {name} rel. diff {err:.1e}")
    return ok


def cpr_check(ref) -> bool:
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
    from pymrst.modules.ad_core.backends.diagonal.diagonal_auto_diff_backend import (
        DiagonalAutoDiffBackend,
    )
    from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
        get_initialization_regions_black_oil,
    )
    from pymrst.modules.ad_core.initialization.init_state_black_oil_ad import (
        init_state_black_oil_ad,
    )
    from pymrst.modules.ad_core.solvers.amgcl_cpr_solver_ad import AMGCL_CPRSolverAD
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import make_rock, pore_volume
    from pymrst.params.gravity.gravity import gravity
    from pymrst.params.wells_and_bc.vertical_well import vertical_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, POISE, YEAR

    class Capture(AMGCL_CPRSolverAD):
        def solve_linear_system(self, A, b, x0=None):
            self.A_captured, self.b_captured = sp.csr_matrix(A), np.array(b)
            self.weights_captured = np.array(self.amgcl_setup["drs_row_weights"])
            return spla.spsolve(sp.csc_matrix(A), b)

    gravity("reset", "on")
    G = compute_geometry(cart_grid(list(DIMS), [1000.0, 1000.0, 100.0]))
    rock = make_rock(G, 0.1*DARCY, 0.3)
    irate = 0.3*np.sum(pore_volume(G, rock)/(10*YEAR))
    W = vertical_well([], G, rock, 0, 0, None, compi=np.array([1.0, 0, 0]), well_type="rate",
                      val=irate)
    W = vertical_well(W, G, rock, DIMS[0] - 1, DIMS[1] - 1, [0], compi=np.array([1.0, 0, 0]),
                      well_type="bhp", val=100*BARSA)
    fluid = init_simple_adi_fluid(n=[2, 3, 1], rho=[1000, 700, 10],
                                  mu=np.array([1, 5, 0.1])*CENTI*POISE,
                                  c=np.array([0, 1e-6, 1e-4])/BARSA)
    dt = 30*DAY
    schedule = simple_schedule(np.full(5, dt), W=W)
    model = GenericBlackOilModel(G, rock, fluid, disgas=False, vapoil=False)
    model.AutoDiffBackend = DiagonalAutoDiffBackend(use_mex=True)
    model = model.validate_model()
    model.FacilityModel.primaryVariableSet = "bhp"
    region = get_initialization_regions_black_oil(model, [70, 30], datum_pressure=200*BARSA)
    state0, _ = init_state_black_oil_ad(model, region)
    forces = model.get_driving_forces(schedule.controls[0])
    m1 = model.validate_model(forces)
    st0 = m1.validate_state(state0)
    m1, st0 = m1.update_for_changed_controls(st0, forces)
    m1, st = m1.prepare_report_step(st0, st0, dt, forces)
    m1, st = m1.prepare_timestep(st, st0, dt, forces)
    problem, _ = m1.get_equations(st0, st, dt, forces, iteration=1)
    cap = Capture(tolerance=1e-3, block_size=m1.get_number_of_components(),
                  coarsening="aggregation", relaxation="ilu0")
    dx, _, _ = cap.solve_linear_problem(problem, m1)
    A_ref = sp.csr_matrix(ref["A_cpr"])
    ok = report(cap.A_captured.shape == A_ref.shape, f"CPR system {A_ref.shape[0]} unknowns")
    err = abs(cap.A_captured - A_ref).max()/abs(A_ref).max()
    ok &= report(err < SYSTEM_RTOL, f"CPR: the matrix handed to AMGCL, rel. diff {err:.1e}")
    for name, actual, expected in (("right-hand side", cap.b_captured, np.ravel(ref["b_cpr"])),
                                   ("DRS row weights", cap.weights_captured, np.ravel(ref["w_cpr"])),
                                   ("pressure scaling", cap.pressureScaling, ref["pressureScaling"])):
        err = _rel(actual, expected)
        ok &= report(err < SYSTEM_RTOL, f"CPR: {name} rel. diff {err:.1e}")
    full = np.concatenate([np.ravel(np.asarray(d, dtype=float)) for d in dx])
    err = _rel(full, np.ravel(ref["dx_cpr"]))
    ok &= report(err < SYSTEM_RTOL*1e3, f"CPR: increment from the prepared system, rel. diff {err:.1e}")
    return ok


def main() -> int:
    from pymrst.params.gravity.gravity import isolated_gravity
    ref = loadmat(HERE/"ref_blackoil_tutorial_mex_acceleration.mat", squeeze_me=True)
    with isolated_gravity():
        ok = run_check(ref)
        ok &= cpr_check(ref)
    print("\nblackoilTutorialMexAcceleration matches MRST" if ok
          else "\nblackoilTutorialMexAcceleration does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
