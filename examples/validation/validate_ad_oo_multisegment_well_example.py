"""Validate MRST multisegmentWellExample: the port in
`ad_blackoil/examples/multisegment_well_example.py`.

The reference is MRST 2024a's run without its figures
(`dump_multisegment_well_example_reference.m`): SPE1 over its whole schedule
with a standard and with a multisegment producer. Every report step's well
rates and bhp (both wells, both runs), the multisegment node pressures,
segment fluxes and node mass fractions, and the completed cells' pressures
are compared to `RTOL` (relative to each quantity's largest magnitude), the
final pressure and gas saturation everywhere likewise, and the Newton
iterations exactly.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.multisegment_well_example import (
    multisegment_well_example,
)
from pymrst.params.gravity.gravity import isolated_gravity

HERE = Path(__file__).resolve().parent
RTOL = 1e-9


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rates(well_sols, i: int) -> np.ndarray:
    return np.array([[ws[i].qWs, ws[i].qOs, ws[i].qGs] for ws in well_sols], dtype=float)


def _iterations(rep) -> np.ndarray:
    return np.array([r["Iterations"] for r in rep["ControlstepReports"]], dtype=int)


def main() -> int:
    ref = loadmat(HERE/"ref_multisegment_well_example.mat")
    with isolated_gravity():
        out = multisegment_well_example(plot=False)
    ms, simple = out.wellSols, out.wellSolsSimple
    c = out.c
    actual = dict(
        qMS=_rates(ms, 1), qSimple=_rates(simple, 1),
        qInjMS=_rates(ms, 0), qInjSimple=_rates(simple, 0),
        bhpMS=np.array([w[1].bhp for w in ms]), bhpSimple=np.array([w[1].bhp for w in simple]),
        bhpInjMS=np.array([w[0].bhp for w in ms]),
        bhpInjSimple=np.array([w[0].bhp for w in simple]),
        nodePressure=np.vstack([w[1].nodePressure for w in ms]),
        segmentFlux=np.vstack([w[1].segmentFlux for w in ms]),
        nodeComp=np.vstack([np.asarray(w[1].nodeComp).ravel(order="F") for w in ms]),
        pCellsMS=np.vstack([np.asarray(s["pressure"])[c] for s in out.states]),
        pCellsSimple=np.vstack([np.asarray(s["pressure"])[c] for s in out.statesSimple]),
        pMS=np.asarray(out.states[-1]["pressure"]),
        sgMS=np.asarray(out.states[-1]["s"])[:, 2],
        pSimple=np.asarray(out.statesSimple[-1]["pressure"]),
        sgSimple=np.asarray(out.statesSimple[-1]["s"])[:, 2])
    ok = report(len(ms) == np.size(ref["bhpMS"]), f"{len(ms)} report steps")
    for name, key in (("multisegment", "itMS"), ("standard", "itSimple")):
        its = _iterations(out.report if key == "itMS" else out.reportSimple)
        expected = np.ravel(ref[key]).astype(int)
        same = np.array_equal(its, expected)
        detail = "" if same else f" (first difference at step {np.flatnonzero(its != expected)[0] + 1})"
        ok &= report(same, f"{name} Newton iterations identical, {its.sum()} in all{detail}")
    for key, value in actual.items():
        expected = np.asarray(ref[key], dtype=float).reshape(np.shape(value))
        err = np.abs(value - expected).max()/max(np.abs(expected).max(), 1e-300)
        ok &= report(err < RTOL, f"{key} rel. diff {err:.1e}")
    print("\nmultisegmentWellExample matches MRST" if ok
          else "\nmultisegmentWellExample does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
