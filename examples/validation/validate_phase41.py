"""Phase 41: boundary conditions and sources for AD models, against MATLAB.

PyMRST's AD models could be driven by **wells and nothing else**. A pressure
boundary, a flux boundary, an aquifer standing in for the rest of the field, a
source term - none existed. Every case had to be a closed box with a well in
it, which is not how a reservoir sits in the world, and is not how most of
MRST's own examples are set up.

`dump_phase41_reference.m` runs three cases through MRST's
`GenericBlackOilModelAD`, **none of which has a single well**:

1. pressure in / pressure out - a model driven entirely by its boundary;
2. a flux boundary in, a pressure boundary out;
3. a source and a sink, with no boundary at all.

All three are now runnable in PyMRST, and all three are compared here.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import GenericBlackOilModelAD
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_source, fluxside, make_rock, pside
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

HERE = Path(__file__).parent
REF = HERE / "ref_p41_bc.mat"


def build(ref, case: int):
    nx = int(ref["nx"])
    p_ref = float(ref["pRef"])
    g = compute_geometry(cart_grid([nx, 1, 1], [80.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=p_ref,
    )

    bc = src = None
    if case == 1:
        bc = pside(None, g, "xmin", 250 * BARSA, sat=np.array([1.0, 0.0]))
        bc = pside(bc, g, "xmax", 150 * BARSA, sat=np.array([0.0, 1.0]))
        n_steps = 10
    elif case == 2:
        bc = fluxside(None, g, "xmin", float(ref["flux_rate"]),
                      sat=np.array([1.0, 0.0]))
        bc = pside(bc, g, "xmax", 200 * BARSA, sat=np.array([0.0, 1.0]))
        n_steps = 10
    else:
        r = float(ref["src_rate"])
        src = add_source(None, np.array([0]), np.array([r]),
                         sat=np.array([[1.0, 0.0]]))
        src = add_source(src, np.array([g.cells.num - 1]), np.array([-r]),
                         sat=np.array([[0.0, 1.0]]))
        n_steps = 4

    # The same tolerances the MATLAB dump was run at. The default material-
    # balance tolerance (1e-12) is what limits the flux-BC case: loosen it and
    # the comparison sits at 2e-9, tighten it and it sits at machine precision.
    model = GenericBlackOilModelAD(g, rock, fluid, [], bc=bc, src=src,
                                 tol_cnv=1e-11, tol_mb=1e-14, tol_well=1e-11)
    st = State(pressure=np.full(g.cells.num, p_ref),
               sW=np.full(g.cells.num, 0.2),
               qWs=np.zeros(0), qOs=np.zeros(0), bhp=np.zeros(0))
    return model, st, n_steps


def run_python(ref, case: int):
    model, st, n_steps = build(ref, case)
    solver = NonLinearSolver(max_iterations=40)
    p, sw = [], []
    for i in range(n_steps):
        st, rep = solver.solve_ministep(model, st, st.copy(), 5 * DAY)
        assert rep.converged, f"case {case} step {i}: {rep.residuals[-1]}"
        p.append(st["pressure"].copy())
        sw.append(st["sW"].copy())
    return np.vstack(p), np.vstack(sw)


def compare(name, py, ml, rtol) -> int:
    py = np.asarray(py, float).ravel()
    ml = np.asarray(ml, float).ravel()
    scale = max(np.abs(ml).max(), 1e-300)
    diff = float(np.abs(py - ml).max())
    ok = diff <= rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p41_bc.mat missing (run dump_phase41_reference.m in MATLAB)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    failed = 0

    cases = [
        (1, "pressure in / pressure out - no wells at all", "p1", "s1"),
        (2, "flux in / pressure out", "p2", "s2"),
        (3, "a source and a sink, no boundary", "p3", "s3"),
    ]
    for case, label, pk, sk in cases:
        print(f"\n{case}. {label}:")
        p, sw = run_python(ref, case)
        failed += compare("pressure", p, ref[pk], 1e-9)
        failed += compare("sW", sw, ref[sk], 1e-9)

    print("\nBoundary conditions and sources match MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
