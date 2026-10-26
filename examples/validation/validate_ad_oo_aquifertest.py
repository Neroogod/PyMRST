"""Validate MRST aquifertest: the port in `ad_blackoil/examples/aquifertest.py`.

The reference is MRST 2024a running the script without its figures
(`dump_aquifertest_reference.m`): both published decks, each set up by
`initEclipseProblemAD` (a `GenericBlackOilModel` with an `AquiferModel`) and
simulated over its 21 report steps. For each case:
- the model and aquifer classes, and the initial state, to `EXACT_RTOL`;
- Newton iterations and ministeps identical at every report step;
- pressures, water saturations, the aquifers' pressures and volumes, and the
  wells' bhps and surface rates to `RUN_RTOL`;
- the ECLIPSE restart states `loadAquiferEclipseResult` reads (21, final
  pressure in bar) exactly, and the relative difference the script plots to
  `ERR_RTOL`.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
EXACT_RTOL = 1e-13
RUN_RTOL = 1e-12
#: The relative difference divides by the pressure, and its own differences by
#: the largest of it (3e-4), so it carries the states' round-off up ~1e3.
ERR_RTOL = 1e-10


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def check_case(r, ref) -> bool:
    tag = f"case {r.case}:"
    ok = report(type(r.model).__name__ == str(ref.model_class)
                and type(r.model.AquiferModel).__name__ == str(ref.aquifer_class),
                f"{tag} {ref.model_class} with an {ref.aquifer_class}")
    for name, actual, expected in (("initial pressure", r.state0["pressure"], ref.p0),
                                   ("initial saturations", r.state0["s"], ref.s0)):
        err = _rel(actual, expected)
        ok &= report(err < EXACT_RTOL, f"{tag} {name} rel. diff {err:.1e}")
    reports = r.schedulereport["ControlstepReports"]
    its = np.array([x["Iterations"] for x in reports])
    nmini = np.array([len(x["StepReports"]) for x in reports])
    ok &= report(np.array_equal(its, np.atleast_1d(ref.its)),
                 f"{tag} Newton iterations identical over {its.size} steps ({its.sum()})")
    ok &= report(np.array_equal(nmini, np.atleast_1d(ref.nmini)), f"{tag} ministeps identical")
    states = r.states
    series = {
        "pressure": (np.array([s["pressure"] for s in states]), ref.p),
        "water saturation": (np.array([np.asarray(s["s"])[:, 0] for s in states]), ref.sw),
        "aquifer pressure": (np.array([[a.pressure for a in s["aquiferSol"]] for s in states]), ref.aqp),
        "aquifer volume": (np.array([[a.volume for a in s["aquiferSol"]] for s in states]), ref.aqv),
    }
    if int(ref.nwells) > 0:
        for key in ("bhp", "qWs", "qOs"):
            series[f"well {key}"] = (np.array([[getattr(w, key) for w in ws] for ws in r.wellSols]),
                                     getattr(ref, key))
    for name, (actual, expected) in series.items():
        err = _rel(actual, expected)
        ok &= report(err < RUN_RTOL, f"{tag} {name} rel. diff {err:.1e}")
    ok &= report(len(r.statesEcl) == int(ref.necl)
                 and np.array_equal(r.statesEcl[-1]["pressure"], ref.pecl_end),
                 f"{tag} ECLIPSE states as loadAquiferEclipseResult reads them")
    err = _rel(r.err, ref.err)
    ok &= report(err < ERR_RTOL, f"{tag} relative difference to ECLIPSE rel. diff {err:.1e} "
                                 f"(max |err| {np.abs(r.err).max():.1e}, MRST {np.abs(ref.err).max():.1e})")
    return ok


def main() -> int:
    from pymrst.modules.ad_blackoil.examples.aquifertest import aquifertest
    from pymrst.params.gravity.gravity import isolated_gravity

    ref = loadmat(HERE/"ref_aquifertest.mat", squeeze_me=True, struct_as_record=False)
    with isolated_gravity():
        results = aquifertest(plot=False)
    ok = check_case(results[0], ref["case1"]) & check_case(results[1], ref["case2"])
    print("\naquifertest matches MRST" if ok else "\naquifertest does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
