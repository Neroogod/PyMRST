"""Validate MRST WellComponentTotalVolumeBalanceCrossflow: the port in
`ad_core/ad_oo/statefunctions/facility/well_component_total_volume_balance_crossflow.py`.

The reference (`dump_well_component_total_volume_balance_crossflow_reference.m`)
has two layers that do not communicate, 100 bar apart, and a rate-controlled
producer and injector perforating both, so both of the function's mixing
rules are used: the producer's (phase mass fractions) and the injector's
(backflow density plus injected stream). The component fluxes of the first
linearisation, and three steps of pressures, saturations, well rates and
bhp, are compared to `RTOL`; the Newton iterations exactly.

Pressure and bhp are compared to `RTOL_LEVEL`. With both wells on rate
control and compressibilities of 1e-9/bar, the pressure level is all but
undetermined and round-off moves it as a whole: the two runs differ by a
uniform 1e-3 Pa, while perturbing the initial pressure by 1e-14 (relative)
moves the Python run alone by 3e-3 to 1.4e-2 Pa.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_core.ad_oo.generic_facility_model import GenericFacilityModel
from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
from pymrst.modules.ad_core.ad_oo.statefunctions.facility.well_component_total_volume_balance_crossflow import (
    WellComponentTotalVolumeBalanceCrossflow,
)
from pymrst.modules.ad_core.utils.schedule import simple_schedule
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params.gravity.gravity import gravity, isolated_gravity
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.add_well import add_well
from pymrst.solvers.init_res_sol import init_res_sol
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

HERE = Path(__file__).resolve().parent
RTOL = 1e-10
RTOL_LEVEL = 1e-9


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def build():
    G = compute_geometry(cart_grid([3, 1, 2], [300.0, 10.0, 20.0]))
    rock = make_rock(G, np.array([100, 100, 1e-6])*MILLI*DARCY, 0.2)
    fluid = init_simple_adi_fluid(phases="WO", mu=np.array([1, 5])*CENTI*POISE,
                                  rho=[1000, 800], n=[2, 2], c=np.array([1e-9, 1e-9])/BARSA,
                                  c_r=1e-10/BARSA)
    model = GenericBlackOilModel(G, rock, fluid, gas=False)
    top = G.cells.centroids[:, 2] < 10
    state0 = init_res_sol(G, 100*BARSA + 100*BARSA*~top, np.array([0.3, 0.7]))
    W = add_well([], G, rock, [0, 3], well_type="rate", val=-1/DAY, name="P",
                 compi=np.array([1, 0]), sign=-1)
    W = add_well(W, G, rock, [2, 5], well_type="rate", val=1/DAY, name="I",
                 compi=np.array([1, 0]), sign=1)
    model.FacilityModel = GenericFacilityModel(model)
    model = model.validate_model({"W": W})
    xflow = WellComponentTotalVolumeBalanceCrossflow(model)
    model.FacilityModel.FacilityFlowDiscretization.set_state_function("ComponentTotalFlux", xflow)
    return G, model, state0, W


def main() -> int:
    ref = loadmat(HERE/"ref_well_component_total_volume_balance_crossflow.mat")
    with isolated_gravity():
        gravity("reset", "off")
        G, model, state0, W = build()
        forces = {"W": W, "bc": None, "src": None}
        st = model.validate_state(deepcopy(state0))
        _, st = model.get_equations(st, st, DAY, forces, iteration=1, resOnly=True)
        st = model.init_state_function_containers(st)
        Q = model.FacilityModel.get_prop(st, "ComponentTotalFlux")
        Q0 = np.column_stack([np.asarray(value(q), dtype=float).ravel() for q in Q])
        schedule = simple_schedule(np.full(3, DAY), W=W)
        ws, states, rep = simulate_schedule_ad(state0, model, schedule)
    err = np.abs(Q0 - ref["Q0"]).max()/np.abs(ref["Q0"]).max()
    ok = report(err < RTOL, f"first-iteration component fluxes rel. diff {err:.1e}")
    its = np.asarray(rep["Iterations"]).astype(int)
    ok &= report(np.array_equal(its, np.ravel(ref["its"]).astype(int)),
                 f"Newton iterations {its.tolist()} (MRST {np.ravel(ref['its']).tolist()})")
    actual = dict(p=np.array([s["pressure"] for s in states]),
                  sw=np.array([np.asarray(s["s"])[:, 0] for s in states]),
                  bhp=np.array([[w.bhp for w in row] for row in ws]),
                  qWs=np.array([[w.qWs for w in row] for row in ws]),
                  qOs=np.array([[w.qOs for w in row] for row in ws]))
    for key, val in actual.items():
        expected = np.asarray(ref[key], dtype=float)
        err = np.abs(val - expected).max()/max(np.abs(expected).max(), 1e-300)
        tol = RTOL_LEVEL if key in ("p", "bhp") else RTOL
        ok &= report(err < tol, f"{key} rel. diff {err:.1e}")
    print("\nWellComponentTotalVolumeBalanceCrossflow matches MRST" if ok
          else "\nWellComponentTotalVolumeBalanceCrossflow does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
