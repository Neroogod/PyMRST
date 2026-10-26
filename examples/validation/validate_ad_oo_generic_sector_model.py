"""Validate MRST blackoilSectorModelExample through the AD-OO
GenericBlackOilModel.

`selectModelFromDeck` builds a `GenericBlackOilModel`, so SPE1, SPE9 and the
field examples all run through it: component mass conservation from the
flow discretization, well sources and equations from `GenericFacilityModel`.
With dissolved gas, `validateModel` gives it an `ImmiscibleComponent` for
water, an `OilComponent` and a `GasComponent` (gas carried in the oil phase
through `rs`), so this case validates those components too.
The reference is MRST 2024a's own run of the same sector case
(`dump_ad_oo_generic_sector_model_reference.m`): every step's pressure,
saturations, `rs`, the producer's rates, BHP and control mode, the Newton
iteration counts, and the face fluxes stored after the last step.

The closed case meets a knife edge in MRST itself. At step 37 the producer
switches to BHP control, and the next Newton step lands the BHP on its
100-bar limit to round-off. `SimpleWell.updateWellSol` snaps a BHP that
crosses its limit back to `limit -/+ 0.1 bar`; MRST's solve lands 3e-14 bar
above the limit (snap to 99.9 bar), this port's 4e-14 bar below (no snap).
That one branch costs an iteration and leaves ~1e-5 differences
afterwards. So the closed case is checked twice:

- as run: identical to round-off before step 37, within `KNIFE_EDGE_RTOL`
  after it, and the same iteration counts everywhere except step 37;
- with MRST's side of the round-off taken at a landing on the limit: then
  every step must match to `RTOL`, which confirms the difference is that
  branch and nothing else.

The closed tank ends nearly at rest: its face fluxes (~1e-12 m^3/s) are
differences of ~1e7 Pa pressures, with a round-off floor near 1e-16 m^3/s,
so fluxes are compared with `FLUX_ATOL` as an absolute floor.
"""

from contextlib import contextmanager

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_core.ad_oo import simple_well
from pymrst.params.gravity.gravity import isolated_gravity

from validate_ad_oo_sector_model import HERE, RTOL, run_case

KNIFE_EDGE_STEP = 36             # step 37, 0-based
KNIFE_EDGE_RTOL = 1e-4
FLUX_ATOL = 1e-14                # m^3/s
LIMIT_LANDING_PA = 1e-6
FIELDS = ("p", "rs", "sg", "sw", "qo", "qg", "bhp", "flux")


@contextmanager
def mrst_side_of_limit_landings():
    """A BHP update that lands on its limit to round-off, from more than
    the snap tolerance away, takes the snap branch as MRST's solve did."""
    original = simple_well.SimpleWell.update_well_sol

    def update_well_sol(self, well_sol, variables, dx, resmodel):
        before = well_sol.bhp
        out = original(self, well_sol, variables, dx, resmodel)
        lims = self.W.lims
        if isinstance(lims, dict) and "bhp" in lims:
            dist = before - lims["bhp"]
            tol = 0.1e5
            if abs(out.bhp - lims["bhp"]) < LIMIT_LANDING_PA and abs(dist) > 2*tol:
                out.bhp = lims["bhp"] + tol*np.sign(dist)
        return out

    simple_well.SimpleWell.update_well_sol = update_well_sol
    try:
        yield
    finally:
        simple_well.SimpleWell.update_well_sol = original


def relative_error(key: str, actual: np.ndarray, expected: np.ndarray) -> float:
    scale = max(np.abs(expected).max(), FLUX_ATOL/RTOL if key == "flux" else 1e-300)
    return float(np.abs(actual - expected).max()/scale)


def check(label: str, run: dict, ref: dict, k: int, steps: slice = slice(None),
          rtol: float = RTOL) -> bool:
    ok_all = True
    whole_run = steps == slice(None)
    for key in FIELDS:
        if key == "flux" and not whole_run:
            continue                      # stored for the last step only
        expected = np.asarray(ref[f"{key}{k}"], dtype=float)
        actual = run[key]
        if key != "flux":
            expected, actual = expected[steps], actual[steps]
        err = relative_error(key, actual, expected)
        ok = err < rtol
        ok_all &= ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {key} rel {err:.2e}")
    ok = np.array_equal(np.ravel(run["bhpctl"]).astype(int),
                        np.ravel(ref[f"bhpctl{k}"]).astype(int))
    ok_all &= ok
    print(f"{'ok  ' if ok else 'FAIL'} {label}: control mode identical")
    return ok_all


def check_iterations(label: str, run: dict, ref: dict, k: int,
                     skip: tuple = ()) -> bool:
    its, expected = np.ravel(run["its"]).astype(int), np.ravel(ref[f"its{k}"]).astype(int)
    keep = np.ones(its.size, dtype=bool)
    keep[list(skip)] = False
    ok = its.size == expected.size and np.array_equal(its[keep], expected[keep])
    note = f" (except step {skip[0] + 1})" if skip else ""
    print(f"{'ok  ' if ok else 'FAIL'} {label}: Newton iterations identical{note}")
    return ok


def main() -> int:
    path = HERE/"ref_ad_oo_generic_sector_model.mat"
    if not path.exists():
        print(f"{path.name} missing (run dump_ad_oo_generic_sector_model_reference.m)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    with isolated_gravity():
        open_run, closed_run = run_case(GenericBlackOilModel)
        with mrst_side_of_limit_landings():
            _, closed_mrst_side = run_case(GenericBlackOilModel)
    ok = check("open boundary", open_run, ref, 1)
    ok &= check_iterations("open boundary", open_run, ref, 1)
    before = slice(0, KNIFE_EDGE_STEP)
    ok &= check("closed, before step 37", closed_run, ref, 2, before)
    ok &= check("closed, whole run", closed_run, ref, 2, rtol=KNIFE_EDGE_RTOL)
    ok &= check_iterations("closed", closed_run, ref, 2, skip=(KNIFE_EDGE_STEP,))
    ok &= check("closed, MRST's side at step 37", closed_mrst_side, ref, 2)
    ok &= check_iterations("closed, MRST's side at step 37", closed_mrst_side, ref, 2)
    return int(not ok)


if __name__ == "__main__":
    raise SystemExit(main())
