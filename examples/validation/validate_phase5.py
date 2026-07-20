"""Compare PyMRST's fully-implicit oil-water model against MATLAB ad-blackoil.

Both sides solve the same discrete system (equationsOilWater + standard
well model) to tight tolerances; converged states should agree to the
tolerance level, far below any physical significance.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import TwoPhaseOilWaterModelAD
from pymrst.modules.ad_core import simulate_schedule_ad
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

HERE = Path(__file__).parent


def run_python():
    g = compute_geometry(cart_grid([15, 15, 1], [300.0, 300.0, 10.0]))
    kvals = (50 + (11 * np.arange(g.cells.num)) % 200) * MILLI * DARCY
    rock = make_rock(g, kvals[:, None], 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
    )
    w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                 compi=np.array([1.0, 0.0]), sign=1)
    w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
    model = TwoPhaseOilWaterModelAD(g, rock, fluid, w)
    st0 = model.initial_state(200 * BARSA, 0.2)
    states, _ = simulate_schedule_ad(st0, model, np.full(8, 30.0) * DAY)
    return w, states


def main() -> int:
    path = HERE / "ref_p5_ow.mat"
    if not path.exists():
        print("ref_p5_ow.mat missing (run dump_phase5_reference.m in MATLAB)")
        return 1
    ref = scipy.io.loadmat(path, squeeze_me=True)
    wells, states = run_python()

    p_py = np.vstack([s.pressure for s in states])
    sw_py = np.vstack([s.sW for s in states])
    qws_py = np.vstack([s.qWs for s in states])
    qos_py = np.vstack([s.qOs for s in states])
    bhp_py = np.vstack([s.bhp for s in states])

    checks = [
        ("WI (both wells)", np.concatenate([wells[0].WI, wells[1].WI]),
         np.atleast_1d(np.concatenate([np.atleast_1d(ref["WI1"]),
                                       np.atleast_1d(ref["WI2"])])), 1e-12),
        ("pressure history", p_py, ref["p_hist"], 5e-6),
        ("sW history", sw_py, ref["sw_hist"], 5e-6),
        ("qWs history", qws_py, ref["qWs"], 5e-6),
        ("qOs history", qos_py, ref["qOs"], 5e-6),
        ("bhp history", bhp_py, ref["bhp"], 5e-6),
    ]
    failed = 0
    for name, py, ml, rtol in checks:
        py = np.asarray(py, dtype=float).ravel()
        ml = np.asarray(ml, dtype=float).ravel()
        scale = max(np.abs(ml).max(), 1e-300)
        diff = float(np.abs(py - ml).max())
        ok = diff <= rtol * scale
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = "
              f"{diff:.3e} (rel {diff / scale:.2e}, rtol {rtol:.0e})")
        failed += 0 if ok else 1

    print("\nAll compared arrays match!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
