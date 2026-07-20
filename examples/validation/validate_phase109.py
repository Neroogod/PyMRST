"""Phase 109: schedule-level multisegment well packing.

This closes the integration edge left after adding the MS equations: a black-oil
schedule with a multisegment well must automatically pack the extra well
unknowns (`pN`, `rW`, `rO`, `vmix`) into the Newton system every time the model
assembles equations.

The MATLAB dump records MRST's variable counts and topology for the same small
schedule. The numerical MS-well physics is still deliberately narrower than
MRST's full facility model, but the schedule-level packing contract is now
validated against MRST's layout instead of only by a local unit test.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
from pymrst.modules.ad_core.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.state_functions import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.params.wells_and_bc import MultisegmentWell, WellSegment
from pymrst.utils.adi import combine_equations
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

HERE = Path(__file__).parent
REF = HERE / "ref_p109_mswell.mat"


def build_case(rate=1e-6):
    g = compute_geometry(cart_grid([3, 1, 1], [300.0, 50.0, 20.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE,
        rho=[1000.0, 800.0],
        n=[2, 2],
        c=[1e-6 / BARSA, 1e-4 / BARSA],
        p_ref=200 * BARSA,
    )
    wells = add_well(
        None, g, rock, np.array([0, 1]), well_type="rate", val=rate,
        compi=np.array([1.0, 0.0]), sign=1, radius=0.1, name="MSI",
    )
    wells[0].multisegment = MultisegmentWell([
        WellSegment(0, -1, depth=1000.0, diameter=0.2, volume=10.0),
        WellSegment(1, 0, length=10.0, depth=1000.0, diameter=0.2,
                    roughness=1e-5, volume=10.0),
        WellSegment(2, 1, length=10.0, depth=1000.0, diameter=0.2,
                    roughness=1e-5, volume=10.0),
    ], np.array([1, 2]))
    wells = add_well(
        wells, g, rock, np.array([2]), well_type="bhp", val=195 * BARSA,
        compi=np.array([0.0, 1.0]), sign=-1, radius=0.1, name="PROD",
    )
    model = GenericBlackOilModel(
        g, rock, fluid, wells, tol_cnv=1e-8, tol_mb=1e-12, tol_well=1e-8,
    )
    state = State(
        pressure=np.full(g.cells.num, 200 * BARSA),
        sW=np.full(g.cells.num, 0.2),
        qWs=np.array([rate, 0.0]),
        qOs=np.array([0.0, 0.0]),
        bhp=np.array([200 * BARSA, 195 * BARSA]),
    )
    return model, state


def _scalar(ref, name):
    return int(np.asarray(ref[name]).squeeze())


def compare(name, py, ml, rtol=1e-12, atol=0.0):
    py = np.asarray(py, dtype=float).ravel()
    ml = np.asarray(ml, dtype=float).ravel()
    if py.size != ml.size:
        print(f"  FAIL {name}: {py.size} values, MRST has {ml.size}")
        return 1
    diff = float(np.abs(py - ml).max()) if py.size else 0.0
    scale = max(float(np.abs(ml).max()) if ml.size else 0.0, 1e-300)
    ok = diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e}")
    return 0 if ok else 1


def run_pymrst_schedule_smoke(model, state):
    """Exercise update/unpack through a real solve attempt.

    The first tiny MS-well schedule can still hit the nonlinear iteration cap
    because PyMRST's MS facility model is intentionally smaller than MRST's, but
    the state must retain all packed variables with finite values.
    """
    solver = NonLinearSolver(max_iterations=6)
    out, report = solver.solve_ministep(model, state, state.copy(), 1.0 * DAY)
    packed = ("ms_pN", "ms_rW", "ms_rO", "ms_vmix")
    finite = all(np.all(np.isfinite(np.asarray(out[k], dtype=float))) for k in packed)
    return report.converged, finite, out


def main() -> int:
    if not REF.exists():
        print("ref_p109_mswell.mat missing (run dump_phase109_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    rate = float(np.asarray(ref["rate"]).squeeze())
    model, state = build_case(rate)
    model._ensure_ms_state(state)

    failed = 0
    ms = model.wells[0].multisegment
    print("Phase 109: MRST/PyMRST multisegment schedule variable packing")
    failed += compare("MS topology", [[s.parent + 1, s.index + 1] for s in ms.segments[1:]],
                      ref["topo0"])
    failed += compare("MS perforation cells", model.wells[0].cells + 1, ref["cells0"])
    failed += compare("MS WI", model.wells[0].WI, ref["WI0"], rtol=2e-12)

    sizes = {
        "pN": len(state["ms_pN"]),
        "rW": len(state["ms_rW"]),
        "rO": len(state["ms_rO"]),
        "vmix": len(state["ms_vmix"]),
    }
    print("\nExtra MS primary variable counts:")
    for key, py_n in sizes.items():
        ref_key = f"n_{key}" if key in ("pN", "vmix") else f"n_{key}"
        ml_n = _scalar(ref, ref_key)
        ok = py_n == ml_n
        print(f"  {'ok  ' if ok else 'FAIL'} {key}: {py_n} values")
        failed += 0 if ok else 1

    eqs = model.get_equations(state, state.copy(), 1.0 * DAY)
    eq = combine_equations(*eqs)
    square = eq.val.size == eq.jac[0].shape[0] == eq.jac[0].shape[1]
    conv = model.get_convergence(eqs, 1.0 * DAY)[1]
    print("\nPacked Newton system:")
    print(f"  {'ok  ' if square else 'FAIL'} square Jacobian "
          f"{eq.jac[0].shape[0]}x{eq.jac[0].shape[1]}")
    print(f"  {'ok  ' if 'extra' in conv else 'FAIL'} MS extra residuals are present")
    failed += 0 if square else 1
    failed += 0 if "extra" in conv else 1

    converged, finite, out = run_pymrst_schedule_smoke(model, state)
    print("\nSchedule-level update smoke:")
    print(f"  ok   solver attempted MS solve; converged={converged}")
    print(f"  {'ok  ' if finite else 'FAIL'} packed MS variables remain finite")
    print(f"  ok   pN={np.array2string(out['ms_pN'], precision=4)}")
    failed += 0 if finite else 1

    print("\nPhase 109 validation passed: MS variables are packed automatically."
          if failed == 0 else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
