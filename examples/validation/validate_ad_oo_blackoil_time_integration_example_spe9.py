"""Validate MRST blackoilTimeIntegrationExample for `name = 'spe9'`: the port
in `ad_blackoil/examples/blackoil_time_integration_example.py`.

The reference is MRST 2024a running the script for SPE9 without its figures,
the outputs read back with `getMultiplePackedSimulatorOutputs`
(`dump_blackoil_time_integration_example_spe9_reference.m`), each packed
problem's old output cleared first; the port's are cleared the same way.
- Fully implicit and adaptive implicit, over the 35 report steps: Newton
  iterations and ministeps identical at every step, ministep lengths, states
  and well curves to `RUN_RTOL`, and for AIM the fraction of implicit cells
  identical at every step - which is decided by `estimateCompositionCFL` (where
  MRST's linear indexing of the mass matrix first showed) and `estimateSaturationCFL`.
- Explicit: identical iterations and ministeps, and states to `RUN_RTOL`,
  through `EXPLICIT_STRICT_STEPS`. MRST's explicit pressure systems are
  numerically singular (MATLAB warns 306,764 times, RCOND down to 1.8e-23)
  and solved by GMRES/ILU to a tolerance, so from step 8 single Newton
  iterations are decided by noise: MRST itself, with its initial pressure
  changed by one ulp, takes 179 instead of 181 iterations in step 8 and
  parts from its own run by 3.5e-4 in sG - as the port does (180, 4.5e-4).
  The remaining steps are reported, and both runs end on the same failure,
  the CFL limit exhausting the substeps, which is checked.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RUN_RTOL = 1e-6
EXPLICIT_STRICT_STEPS = 7
KEYS = ("fim", "explicit", "aim")


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def _series(states, ws, n: int):
    out = {
        "pressure": np.array([s["pressure"] for s in states[:n]]),
        "sw": np.array([np.asarray(s["s"])[:, 0] for s in states[:n]]),
        "sg": np.array([np.asarray(s["s"])[:, 2] for s in states[:n]]),
        "rs": np.array([s["rs"] for s in states[:n]]),
    }
    for key in ("bhp", "qOs", "qGs", "qWs"):
        out[f"well {key}"] = np.array([[getattr(w, key) for w in row] for row in ws[:n]])
    return out


def _reference(ref, n: int):
    out = {"pressure": ref.p[:n], "sw": ref.sw[:n], "sg": ref.sg[:n], "rs": ref.rs[:n]}
    for key in ("bhp", "qOs", "qGs", "qWs"):
        out[f"well {key}"] = getattr(ref, key)[:n]
    return out


def check(i: int, out, ref) -> bool:
    tag = f"{out.names[i]}:"
    reports = out.reports[i]
    its = np.array([r["Iterations"] for r in reports])
    nmini = np.array([len(r["StepReports"]) for r in reports])
    its_ref, nmini_ref = np.atleast_1d(ref.its), np.atleast_1d(ref.nmini)
    if i == 1:
        n = EXPLICIT_STRICT_STEPS
        ok = report(len(its) < 35 and len(its_ref) < 35,
                    f"{tag} both runs stop early on the substep limit (MRST after {len(its_ref)} "
                    f"steps, port after {len(its)})")
    else:
        n = len(its_ref)
        ok = report(len(its) == n, f"{tag} {len(its)} report steps, MRST {n}")
    ok &= report(np.array_equal(its[:n], its_ref[:n]),
                 f"{tag} Newton iterations identical over steps 1-{n} ({its[:n].sum()}, "
                 f"MRST {its_ref[:n].sum()})")
    ok &= report(np.array_equal(nmini[:n], nmini_ref[:n]),
                 f"{tag} ministeps identical over steps 1-{n} ({nmini[:n].sum()})")
    m = int(nmini_ref[:n].sum())
    err = _rel(np.asarray(out.ministeps[i])[:m], np.atleast_1d(ref.dt)[:m])
    ok &= report(err < RUN_RTOL, f"{tag} ministep lengths rel. diff {err:.1e}")
    actual, expected = _series(out.states[i], out.ws[i], n), _reference(ref, n)
    for name in actual:
        err = _rel(actual[name], expected[name])
        ok &= report(err < RUN_RTOL, f"{tag} {name} rel. diff {err:.1e}")
    if i == 2:
        ok &= report(np.array_equal(out.imp_frac, np.atleast_1d(ref.imp_frac)),
                     f"{tag} implicit-cell fraction identical at every step")
    if i == 1:
        k = min(len(its), len(its_ref))
        late = _rel(_series(out.states[i], out.ws[i], k)["sg"], ref.sg[:k])
        print(f"     {tag} steps {n + 1}-{k} reported: iterations {its[n:k].sum()} vs MRST "
              f"{its_ref[n:k].sum()}, sg rel. diff {late:.1e}")
    return ok


def main() -> int:
    from pymrst.modules.ad_blackoil.examples.blackoil_time_integration_example import (
        blackoil_time_integration_example, pack_time_integration_problems,
    )
    from pymrst.modules.ad_core.simulators.sim_runner.clear_packed_simulator_output import (
        clear_packed_simulator_output,
    )
    from pymrst.params.gravity.gravity import isolated_gravity

    ref = loadmat(HERE/"ref_blackoil_time_integration_example_spe9.mat", squeeze_me=True,
                  struct_as_record=False)
    with isolated_gravity():
        clear_packed_simulator_output(pack_time_integration_problems("spe9")[0], Prompt=False)
        out = blackoil_time_integration_example("spe9", plot=False)
    ok = True
    for i, key in enumerate(KEYS):
        ok &= check(i, out, ref[key])
    print("\nblackoilTimeIntegrationExample (spe9) matches MRST" if ok
          else "\nblackoilTimeIntegrationExample (spe9) does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
