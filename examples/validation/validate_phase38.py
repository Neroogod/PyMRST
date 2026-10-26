"""Phase 38: capillary pressure, against MRST's state-function model.

Until now every PyMRST model assumed `p_w = p_o = p_g` - there was no `Pc`
anywhere in the graph, and `PressureGradient` returned a single gradient that
all phases shared. So water could only ever move where the *oil* pressure
moved it, and capillary imbibition was not merely inaccurate, it was
identically zero.

The chain MRST uses, and PyMRST now mirrors:

    CapillaryPressure (FlowProps)   pc_alpha, with water carrying -pcOW
        -> PhasePressures (PVTProps)   p_alpha = p + pc_alpha
            -> PressureGradient (FluxDisc)   one gradient PER PHASE
                -> PhasePotentialDifference -> PhaseFlux

`dump_phase38_reference.m` runs the same oil-water case through MRST's
`GenericBlackOilModelAD` **twice**, with and without a capillary pressure, and
both are compared here:

- the **no-pc** case must still match, which proves the change is inert where
  it should be (and that Phases 5, 5b, 24 and 27 were not silently disturbed);
- the **pc** case must match, which proves the new chain is right;
- and the two must actually *differ* from each other, or the test would be
  passing on a capillary pressure that does nothing.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import GenericBlackOilModelAD
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

HERE = Path(__file__).parent
REF = HERE / "ref_p38_pc.mat"


def run_python(ref, use_pc: bool):
    """The case of dump_phase38_reference.m, with or without capillarity."""
    nx = int(ref["nx"])
    p_ref = float(ref["pRef"])
    pc0 = float(ref["pc0"])

    g = compute_geometry(cart_grid([nx, nx, 1], [300.0, 300.0, 10.0]))
    rock = make_rock(g, np.asarray(ref["kvals"], float).ravel()[:, None], 0.25)

    # pcOW = p_oil - p_water, decreasing in water saturation (MRST's sign).
    pc_ow = (lambda sw: pc0 * (1.0 - sw)) if use_pc else None
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=p_ref, pc_ow=pc_ow,
    )

    w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                 compi=np.array([1.0, 0.0]), sign=1, radius=0.1)
    w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1,
                 radius=0.1)

    model = GenericBlackOilModelAD(g, rock, fluid, w)
    st = State(
        pressure=np.full(g.cells.num, p_ref),
        sW=np.full(g.cells.num, 0.2),
        qWs=np.array([2e-3, 0.0]), qOs=np.array([0.0, -2e-3]),
        bhp=np.array([p_ref, 150 * BARSA]),
    )

    solver = NonLinearSolver(max_iterations=30)
    p, sw, bhp = [], [], []
    for i in range(8):
        st, rep = solver.solve_ministep(model, st, st.copy(), 30 * DAY)
        assert rep.converged, f"step {i}: {rep.residuals[-1]}"
        p.append(st["pressure"].copy())
        sw.append(st["sW"].copy())
        bhp.append(st["bhp"].copy())
    return np.vstack(p), np.vstack(sw), np.vstack(bhp)


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
        print("ref_p38_pc.mat missing (run dump_phase38_reference.m in MATLAB)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    failed = 0

    print("Without capillary pressure - the chain must be inert:")
    p0, sw0, bhp0 = run_python(ref, use_pc=False)
    failed += compare("pressure", p0, ref["no_p"], 1e-10)
    failed += compare("sW", sw0, ref["no_sw"], 1e-10)
    failed += compare("bhp", bhp0, ref["no_bhp"], 1e-10)

    print("\nWith capillary pressure - the new chain must be right:")
    p1, sw1, bhp1 = run_python(ref, use_pc=True)
    failed += compare("pressure", p1, ref["pc_p"], 1e-10)
    failed += compare("sW", sw1, ref["pc_sw"], 1e-10)
    failed += compare("bhp", bhp1, ref["pc_bhp"], 1e-10)

    # If capillarity made no difference, the two comparisons above would both
    # pass on a term that does nothing. Check it actually bites.
    d_py = np.abs(sw1 - sw0).max()
    d_ml = np.abs(np.asarray(ref["pc_sw"], float)
                  - np.asarray(ref["no_sw"], float)).max()
    print(f"\n  Capillary pressure moves sW by {d_py:.4f} saturation units "
          f"(MRST: {d_ml:.4f}),")
    print("  so the two cases above are genuinely different problems.")
    if d_py < 1e-3:
        print("  FAIL: capillarity had no effect; the test proves nothing")
        failed += 1

    print("\nCapillary pressure matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
