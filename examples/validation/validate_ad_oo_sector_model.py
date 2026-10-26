"""Validate MRST blackoilSectorModelExample: the port in
`ad_blackoil/examples/blackoil_sector_model_example.py`, on the AD-OO
ThreePhaseBlackOilModel.

The reference is MRST 2024a's own run (`dump_ad_oo_sector_model_reference.m`):
SPE1's live-oil fluid on a 50 x 50 sector, an oil-rate producer with a BHP
limit, with constant-pressure boundaries (carrying a dissolution matrix) and
without. Every step's pressure, saturations, `rs`, the producer's rates, BHP
and control mode, and the Newton iteration counts must match.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import ThreePhaseBlackOilModel
from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
from pymrst.modules.ad_core.timesteps.timestep_selectors import rampup_timesteps
from pymrst.modules.ad_core.utils.schedule import simple_schedule
from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
from pymrst.params.gravity.gravity import isolated_gravity
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import pside, vertical_well
from pymrst.solvers.init_res_sol import init_res_sol
from pymrst.utils.units import BARSA, DAY, DARCY, YEAR

HERE = Path(__file__).resolve().parent
RTOL = 1e-10


def _arrays(ws, states, rep):
    return dict(
        p=np.array([s["pressure"] for s in states]),
        rs=np.array([s["rs"] for s in states]),
        sg=np.array([np.asarray(s["s"])[:, 2] for s in states]),
        sw=np.array([np.asarray(s["s"])[:, 0] for s in states]),
        qo=np.array([w[0].qOs for w in ws]), qg=np.array([w[0].qGs for w in ws]),
        bhp=np.array([w[0].bhp for w in ws]),
        bhpctl=np.array([w[0].type == "bhp" for w in ws]),
        its=np.asarray(rep["Iterations"]),
        flux=np.asarray(states[-1].get("flux")))


def run_example():
    """The example as published (`blackoil_sector_model_example`), without
    its figures: one dict per run (open boundary, closed)."""
    from pymrst.modules.ad_blackoil.examples.blackoil_sector_model_example import (
        blackoil_sector_model_example,
    )
    out = blackoil_sector_model_example(plot=False)
    return [_arrays(out.ws, out.states, out.report),
            _arrays(out.ws_c, out.states_c, out.report_c)]


def run_case(model_class=ThreePhaseBlackOilModel):
    """The example's case on `model_class` (for the generic-model check):
    one dict per run (open boundary, closed) of per-step arrays."""
    _, _, fluid, _, _ = setup_spe1()
    fluid = dict(fluid)
    fluid["krW"] = lambda s: s**2
    G = compute_geometry(cart_grid([50, 50, 1], [1000.0, 1000.0, 100.0]))
    rock = make_rock(G, 0.3*DARCY, 0.3)
    s0 = np.array([0.2, 0.8, 0.0])
    p0 = 300*BARSA
    state0 = {k: v for k, v in vars(init_res_sol(G, p0, s0)).items()
              if not k.startswith("_")}
    state0["rs"] = np.full(G.cells.num, 200.0)
    model = model_class(G, rock, fluid, disgas=True)
    T = 10*YEAR
    W = vertical_well(None, G, rock, 24, 24, None,
                      val=-0.25*model.operators.pv.sum()/T, well_type="orat",
                      compi=np.array([1, 1, 1])/3, sign=-1, name="Producer")
    W[0].lims = {"bhp": 100*BARSA}
    bc = None
    for side in ("xmin", "xmax", "ymin", "ymax"):
        bc = pside(bc, G, side, p0, sat=s0)
    bc.dissolution = np.array([[1, 0, 0], [0, 1, 0], [0, 200, 1]], dtype=float)
    dt = rampup_timesteps(T, 30*DAY)
    runs = []
    for sched in (simple_schedule(dt, W=W, bc=bc), simple_schedule(dt, W=W)):
        runs.append(_arrays(*simulate_schedule_ad(state0, model, sched)))
    return runs


def main() -> int:
    path = HERE/"ref_ad_oo_sector_model.mat"
    if not path.exists():
        print(f"{path.name} missing (run dump_ad_oo_sector_model_reference.m)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    with isolated_gravity():
        runs = run_example()
    failed = False
    for k, (label, run) in enumerate(zip(("open boundary", "closed"), runs), 1):
        for key in ("p", "rs", "sg", "sw", "qo", "qg", "bhp"):
            expected = np.asarray(ref[f"{key}{k}"], dtype=float)
            err = np.abs(run[key] - expected).max()/max(np.abs(expected).max(), 1e-300)
            ok = err < RTOL
            failed |= not ok
            print(f"{'ok  ' if ok else 'FAIL'} {label}: {key} rel {err:.2e}")
        for key, what in (("bhpctl", "control mode"), ("its", "Newton iterations")):
            ok = np.array_equal(np.ravel(run[key]).astype(int),
                                np.ravel(ref[f"{key}{k}"]).astype(int))
            failed |= not ok
            print(f"{'ok  ' if ok else 'FAIL'} {label}: {what} identical")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
