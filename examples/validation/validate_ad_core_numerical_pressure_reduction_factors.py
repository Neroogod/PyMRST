"""Validate MRST NumericalPressureReductionFactors, the port in
`ad_core/ad_oo/statefunctions/flowprops/numerical_pressure_reduction_factors.py`.

The reference is MRST 2024a (`dump_numerical_pressure_reduction_factors_reference.m`):
SPE9 as a `GenericBlackOilModel` with dissolved gas, its first control's wells,
and the states of the `equationsBlackOil` validator (every seventh cell with
free gas, pressure perturbed by up to 1 %), MRST's own arrays.
- First iterate, numeric pressure and no previous weights: the weights, from
  the sparse solve `w*A = b`, numeric, to `RTOL`.
- Second iterate, pressure perturbed again and AD (`getStateAD`), the first
  weights as the previous ones: the weights to `RTOL`, and each weight AD with
  one Jacobian block per primary variable (wells included), the pressure
  block diagonal and equal to MRST's `dw/dp` cell by cell to within
  `ULPS` units of round-off. `dw/dp` is a difference quotient, the change in
  weight over the change in pressure, so one ulp of a weight becomes
  `eps*|w|/|dp|` in it: where `cos(i)` puts `dp` near zero that is 4e-15
  against a largest `dw/dp` of 1.5e-5, which is the 2.7e-10 relative
  difference a flat tolerance would report.

And in use (`dump_sequential_numerical_reduction_reference.m`): the sequential
toy case, a 1D waterflood on a `GenericBlackOilModel` split with
`getSequentialModelFromFI`, with the pressure model's `reductionStrategy` set
to `'numerical'` - which selects this class - over three one-day steps:
iterations identical, pressures and saturations to `RUN_RTOL`.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.numerical_pressure_reduction_factors import (
    NumericalPressureReductionFactors,
)
from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
    convert_deck_schedule_to_mrst,
)
from pymrst.modules.test_suite.dataset_functions.spe9 import setup_spe9
from pymrst.params.gravity.gravity import isolated_gravity
from pymrst.utils.adi import value

HERE = Path(__file__).resolve().parent
RTOL = 1e-10
ULPS = 64
RUN_RTOL = 1e-10


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def check_sequential_run() -> bool:
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.modules.sequential.utils.get_sequential_model_from_fi import (
        get_sequential_model_from_fi,
    )
    from pymrst.params.rock import make_rock, pore_volume
    from pymrst.params.wells_and_bc import fluxside, pside
    from pymrst.solvers.init_res_sol import init_res_sol
    from pymrst.utils.units import BARSA, DARCY, DAY

    ref = loadmat(HERE/"ref_sequential_numerical_reduction.mat", squeeze_me=True)
    n = 10
    G = compute_geometry(cart_grid([n], [float(n)]))
    rock = make_rock(G, DARCY, 0.3)
    fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 2e-3], rho=[1000, 800], n=[2, 2])
    model = GenericBlackOilModel(G, rock, fluid, gas=False)
    rate = np.sum(pore_volume(G, rock))/(10*DAY)
    bc = fluxside(None, G, "xmin", rate, sat=np.array([1.0, 0.0]))
    bc = pside(bc, G, "xmax", 0*BARSA, sat=np.array([0.0, 1.0]))
    schedule = simple_schedule(np.full(3, DAY), bc=bc)
    state0 = init_res_sol(G, 50*BARSA, np.array([0.2, 0.8]))
    seq = get_sequential_model_from_fi(model)
    seq.pressureModel.reductionStrategy = "numerical"
    with isolated_gravity():
        _, states, sim_report = simulate_schedule_ad(state0, seq, schedule, Verbose=False)
    rf = seq.validate_model().pressureModel.parentModel.PVTPropertyFunctions \
        .get_state_function("PressureReductionFactors")
    ok = report(type(rf).__name__ == str(ref["rfclass"]),
                 f"sequential: pressure model uses {type(rf).__name__}")
    its = [r["Iterations"] for r in sim_report["ControlstepReports"]]
    ok &= report(its == list(np.atleast_1d(ref["its"])), f"sequential: iterations {its}")
    for name, field, expected in (("pressure", lambda s: s["pressure"], ref["p"]),
                                  ("sw", lambda s: np.asarray(s["s"])[:, 0], ref["sw"])):
        err = _rel(np.array([field(s) for s in states]), expected)
        ok &= report(err < RUN_RTOL, f"sequential: {name} rel. diff {err:.1e}")
    return ok


def main() -> int:
    ref = loadmat(HERE/"ref_numerical_pressure_reduction_factors.mat", squeeze_me=True)
    with isolated_gravity():
        G, rock, fluid, deck, state0 = setup_spe9()
        model = GenericBlackOilModel(G, rock, fluid, disgas=True, inputdata=deck)
        schedule = convert_deck_schedule_to_mrst(model, deck)
        forces = model.get_driving_forces(schedule.controls[0])
        model = model.validate_model(forces)
        st0 = model.validate_state(state0)
        model, st0 = model.update_for_changed_controls(st0, forces)
        st0["pressure"] = np.array(ref["p0"], dtype=float)
        st0["s"] = np.array(ref["s0"], dtype=float)
        st0["rs"] = np.array(ref["rs0"], dtype=float)
        st = copy_state(st0)
        st["pressure"] = np.array(ref["p"], dtype=float)
        st["s"] = np.array(ref["s"], dtype=float)
        st["rs"] = np.array(ref["rs"], dtype=float)
        dt = float(ref["dt"])
        rf = NumericalPressureReductionFactors(model)
        mass0 = model.get_prop(st0, "ComponentTotalMass")
        stA = copy_state(st)
        stA["reductionFactorProps"] = {"mass0": mass0, "mass": None, "dt": dt,
                                       "pressure": None, "weights": None}
        wA = rf.evaluate_on_domain(model, stA)
        stB = copy_state(stA)
        stB["reductionFactorProps"] = dict(stA["reductionFactorProps"],
                                           pressure=stA["pressure"],
                                           weights=np.column_stack(wA))
        stB["pressure"] = np.array(ref["pB"], dtype=float)
        stB, _, _ = model.get_state_ad(stB)
        wB = rf.evaluate_on_domain(model, stB)

    ok = report(not any(hasattr(w, "jac") for w in wA), "first iterate: weights numeric")
    err = _rel(np.column_stack(wA), ref["wA"])
    ok &= report(err < RTOL, f"first iterate: weights rel. diff {err:.1e}")
    ok &= report(all(hasattr(w, "jac") for w in wB), "second iterate: every weight AD")
    err = _rel(np.column_stack([value(w) for w in wB]), ref["wBval"])
    ok &= report(err < RTOL, f"second iterate: weights rel. diff {err:.1e}")
    njac = [len(w.jac) for w in wB]
    ok &= report(njac == list(ref["njac"]), f"second iterate: {njac[0]} Jacobian blocks "
                 f"(MRST {int(ref['njac'][0])})")
    offdiag = [w.jac[0].nnz - np.count_nonzero(w.jac[0].diagonal()) for w in wB]
    ok &= report(offdiag == [0, 0, 0], "second iterate: pressure block diagonal")
    dB = np.column_stack([w.jac[0].diagonal() for w in wB])
    dp = np.abs(ref["pB"] - ref["p"])
    ulp = np.finfo(float).eps*np.abs(ref["wBval"]).max(axis=1)/dp
    ratio = float((np.abs(dB - ref["dB"])/ulp[:, None]).max())
    ok &= report(ratio < ULPS, f"second iterate: dw/dp within {ratio:.1f} ulps of round-off "
                 f"(rel. diff {_rel(dB, ref['dB']):.1e})")
    others = max(abs(J).max() if J.nnz else 0.0 for w in wB for J in w.jac[1:])
    ok &= report(others == 0, "second iterate: no derivative but pressure")
    ok &= check_sequential_run()
    print("\nNumericalPressureReductionFactors matches MRST" if ok
          else "\nNumericalPressureReductionFactors does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
