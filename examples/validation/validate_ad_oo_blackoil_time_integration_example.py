"""Validate MRST blackoilTimeIntegrationExample: the port in
`ad_blackoil/examples/blackoil_time_integration_example.py`.

The reference is MRST 2024a running the script for `name = 'spe1'` without
its figures (`dump_blackoil_time_integration_example_reference.m`), each
packed problem's old output cleared first; the port's are cleared the same
way, and both read back with `getMultiplePackedSimulatorOutputs`. For each of
the three discretizations (fully implicit, explicit,
adaptive implicit), over SPE1's 120 report steps:
- Newton iterations and ministeps identical at every report step, and the
  ministep lengths `getReportMinisteps` returns to `RUN_RTOL`;
- pressures, saturations, `rs` and the wells' bhps and surface rates to
  `RUN_RTOL`;
- for AIM, the fraction of implicit cells at every step identical.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RUN_RTOL = 1e-6
KEYS = ("fim", "explicit", "aim")


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def check(i: int, out, ref) -> bool:
    tag = f"{out.names[i]}:"
    reports = out.reports[i]
    its = np.array([r["Iterations"] for r in reports])
    nmini = np.array([len(r["StepReports"]) for r in reports])
    ok = report(np.array_equal(its, np.atleast_1d(ref.its)),
                f"{tag} Newton iterations identical over {its.size} steps ({its.sum()}, "
                f"MRST {np.sum(ref.its)})")
    ok &= report(np.array_equal(nmini, np.atleast_1d(ref.nmini)),
                 f"{tag} ministeps identical ({nmini.sum()}, MRST {np.sum(ref.nmini)})")
    err = _rel(out.ministeps[i], np.atleast_1d(ref.dt))
    ok &= report(err < RUN_RTOL, f"{tag} ministep lengths rel. diff {err:.1e}")
    states, ws = out.states[i], out.ws[i]
    series = {
        "pressure": (np.array([s["pressure"] for s in states]), ref.p),
        "sw": (np.array([np.asarray(s["s"])[:, 0] for s in states]), ref.sw),
        "sg": (np.array([np.asarray(s["s"])[:, 2] for s in states]), ref.sg),
        "rs": (np.array([s["rs"] for s in states]), ref.rs),
    }
    for key in ("bhp", "qOs", "qGs", "qWs"):
        series[f"well {key}"] = (np.array([[getattr(w, key) for w in row] for row in ws]),
                                 getattr(ref, key))
    for name, (actual, expected) in series.items():
        err = _rel(actual, expected)
        ok &= report(err < RUN_RTOL, f"{tag} {name} rel. diff {err:.1e}")
    if i == 2:
        ok &= report(np.array_equal(out.imp_frac, np.atleast_1d(ref.imp_frac)),
                     f"{tag} implicit-cell fraction identical at every step")
    return ok


def main() -> int:
    from pymrst.modules.ad_blackoil.examples.blackoil_time_integration_example import (
        blackoil_time_integration_example, pack_time_integration_problems,
    )
    from pymrst.modules.ad_core.simulators.sim_runner.clear_packed_simulator_output import (
        clear_packed_simulator_output,
    )
    from pymrst.params.gravity.gravity import isolated_gravity

    ref = loadmat(HERE/"ref_blackoil_time_integration_example.mat", squeeze_me=True,
                  struct_as_record=False)
    with isolated_gravity():
        clear_packed_simulator_output(pack_time_integration_problems("spe1")[0], Prompt=False)
        out = blackoil_time_integration_example("spe1", plot=False)
    ok = True
    for i, key in enumerate(KEYS):
        ok &= check(i, out, ref[key])
    print("\nblackoilTimeIntegrationExample matches MRST" if ok
          else "\nblackoilTimeIntegrationExample does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
