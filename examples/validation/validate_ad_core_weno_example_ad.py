"""Validate MRST wenoExampleAD, the port in `ad_core/examples/weno_example_ad.py`.

The reference is MRST 2024a running the example's computations without its
figures (`dump_weno_example_ad_reference.m`): a quarter five-spot on 50x50
cells over 300 steps, simulated fully implicit and adaptive implicit, each
with single-point upwind and with WENO (`setWENODiscretization`). For each of
the four runs: Newton iterations and ministeps identical at every report step,
ministep lengths, well rates and BHP, and water saturation and pressure every
tenth step to `RUN_RTOL`. The saturation CFL of the first state to `RTOL`.

The two WENO runs' saturations are held to `WENO_SW_RTOL`. They differ from
MRST by 2.1e-8 at step 1 (then less), and that is the direct solver, not the
port: the first Newton systems are identical to 1e-16, but on that system
scipy's SuperLU with its default COLAMD ordering leaves round-off of 1e-14 in
the saturation increments of the cells ahead of the front, where MATLAB's
UMFPACK gives exact zeros. WENO's `cap` (`q <= 0`) decides on those values,
mirrored faces decide differently, and the second Newton step loses the
problem's diagonal symmetry, which MRST keeps to 1e-16. Solved with SuperLU's
natural ordering, which also gives exact zeros, the FIM WENO run matches MRST
to 3e-14 in saturation and 1.6e-14 in water rate with identical iterations;
that ordering is far too slow to be the default.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RTOL = 1e-10
RUN_RTOL = 1e-8
WENO_SW_RTOL = 1e-7
RUNS = (("fim", "FIM SPU", "ws", "states", "report"),
        ("weno", "FIM WENO", "ws_weno", "states_weno", "report_weno"),
        ("aim", "AIM SPU", "ws_e", "states_e", "report_e"),
        ("aim_weno", "AIM WENO", "ws_weno_ex", "states_weno_ex", "report_weno_ex"))


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def check_run(out, ref, key: str, tag: str, ws_name: str, st_name: str, rep_name: str,
              keep: np.ndarray) -> bool:
    r = ref[key]
    reports = getattr(out, rep_name)["ControlstepReports"]
    its = np.array([c["Iterations"] for c in reports])
    nmini = np.array([len(c["StepReports"]) for c in reports])
    dt = np.concatenate([[s["Timestep"] for s in c["StepReports"]] for c in reports])
    ok = report(np.array_equal(its, np.atleast_1d(r.its)),
                f"{tag}: Newton iterations identical at all {its.size} steps ({its.sum()})")
    ok &= report(np.array_equal(nmini, np.atleast_1d(r.nmini)),
                 f"{tag}: ministeps identical ({nmini.sum()})")
    err = _rel(dt, np.atleast_1d(r.dt))
    ok &= report(err < RUN_RTOL, f"{tag}: ministep lengths rel. diff {err:.1e}")
    ws = getattr(out, ws_name)
    for field in ("qWs", "qOs", "bhp"):
        err = _rel([[getattr(w, field) for w in row] for row in ws], getattr(r, field))
        ok &= report(err < RUN_RTOL, f"{tag}: well {field} rel. diff {err:.1e}")
    states = getattr(out, st_name)
    for field, col in (("sw", 0), ("p", None)):
        got = [np.asarray(states[k - 1]["s"])[:, 0] if col == 0
               else np.asarray(states[k - 1]["pressure"]) for k in keep]
        err = _rel(got, getattr(r, field))
        tol = WENO_SW_RTOL if (field == "sw" and "WENO" in tag) else RUN_RTOL
        ok &= report(err < tol, f"{tag}: {field} every tenth step rel. diff {err:.1e}")
    return ok


def main() -> int:
    from pymrst.modules.ad_core.examples.weno_example_ad import weno_example_ad
    ref = loadmat(HERE/"ref_weno_example_ad.mat", squeeze_me=True, struct_as_record=False)
    with contextlib.redirect_stdout(io.StringIO()):
        out = weno_example_ad()
    keep = np.atleast_1d(ref["keep"]).astype(int)
    err = _rel(out.cfl, ref["cfl"])
    ok = report(err < RTOL, f"saturation CFL of the first state rel. diff {err:.1e}")
    for key, tag, ws_name, st_name, rep_name in RUNS:
        ok &= check_run(out, ref, key, tag, ws_name, st_name, rep_name, keep)
    print("\nwenoExampleAD matches MRST" if ok else "\nwenoExampleAD does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
