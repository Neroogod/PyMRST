"""Validate MRST blackoilTutorialSPE9 through the AD-OO port.

The reference is MRST 2024a's tutorial without its figures
(`dump_blackoil_tutorial_spe9_reference.m`); the port's run is read from
`py_blackoil_tutorial_spe9.npz`, which `run_blackoil_tutorial_spe9_python.py`
writes (several minutes).

The tutorial's own run cannot complete here, in MRST or in the port.
`selectLinearSolverAD` picks CPR with a direct pressure solve (no AGMG;
AMGCL needs MRST's MEX build, which fails on this machine). Its GMRES stops
at 25 iterations without reaching 1e-4, and MRST 2024a gives up in report
step 1. The port's CPR reproduces MATLAB's GMRES residuals for the first
Newton iterations of that run; the unconverged increments then take the two
runs apart. So CPR is checked on the first linearised system, and the
simulation (both sides) uses `BackslashSolverAD`.

Checked, section by section:
- the linear solver `selectLinearSolverAD` picks: CPR with a direct
  pressure solve; on the first linearised system, its 25 GMRES iterations,
  final residual and increment, and the direct increment;
- what the sections before the simulation plot: the column of cells, the
  dissolved-gas fraction, the well signs, PROD2's rate targets, and the
  fluid's relative permeabilities (including `relPermWOG`'s three-phase oil
  surface), capillary pressures, formation volume factors, rock
  compressibility and viscosities, all to `EXACT_RTOL`. MRST evaluates
  `bO` and `muO` on matrices, where its `if any(sat)` needs a saturated
  point in every column; there is none at the highest pressures, so the
  saturated points come out 0 (`Bo` infinite). The port does the same;
- the simulation with `BackslashSolverAD`: well controls at every report
  step identical; Newton iterations and ministeps identical, and pressures,
  saturations, `rs`, well rates and BHPs to `RUN_RTOL`, up to report step
  `KNIFE_EDGE_STEP - 1`;
- report step `KNIFE_EDGE_STEP` (18) on MRST's ministeps. Whether its first
  20-day Newton solve converges is decided at the last bit, in MRST itself:
  MRST's report state carries the last iterate's cached
  `ComponentTotalMass` and `Density` into the next step, within an ulp of
  a fresh evaluation, and from MRST's own step-17 state step 18 takes 44
  iterations (fail, cut to 2 x 10 days) with them and 14 (one solve)
  without. The port's state there is 1e-10 from MRST's, so which path it
  takes is chance. The port is therefore run from its own step-17 state on
  MRST's two 10-day ministeps: the state to `KNIFE_RTOL`, and the
  iterations plus the failed attempt's `maxIterations + 1` equal MRST's.
  The report steps after it are reported, not required, as their
  trajectories depend on the path taken;
- the comparison with ECLIPSE: the summary's years, PROD13/PROD18 BHPs and
  gas rates as read by `readEclipseSummaryUnFmt` to `EXACT_RTOL`, the
  simulated ones as `getWellOutput` returns them to `RUN_RTOL`, and the
  control map.

`RUN_RTOL` is the round-off allowance of `validate_ad_oo_simulate_spe9.py`:
SPE9's first linearised system is MRST's to 1e-15, and the linear
solvers' round-off, carried through Newton's chopped updates, grows to
about 1e-6 over the schedule. CPR's GMRES stops at a relative residual of
1e-4, so its iterates agree to the same order.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
EXACT_RTOL = 1e-12
RUN_RTOL = 1e-5
#: CPR stops unconverged after 25 GMRES iterations (residual 3e-3), so its
#: increment carries the two GMRES implementations' rounding, not 1e-15.
CPR_RTOL = 1e-6
KNIFE_EDGE_STEP = 18
KNIFE_RTOL = 1e-9
#: Iterations MRST reports for a ministep that fails: `maxIterations + 1`.
FAILED_ATTEMPT_ITERATIONS = 26
SECTIONS = ("Rs_frac", "sgn", "P", "kr", "krO", "pc", "pvt", "Bo", "muO", "Tcomp", "T",
            "bhp_comp", "gas_comp")
RUN = ("p", "rs_states", "sat", "qWs", "qOs", "qGs", "bhp", "bhp_mrst", "gas_mrst")


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def relative(actual: np.ndarray, expected: np.ndarray) -> float:
    """Largest difference over the largest value; NaNs and infinities
    (MRST's 1./0 where it leaves saturated `bO` at 0) must agree exactly."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if not (np.array_equal(np.isnan(actual), np.isnan(expected))
            and np.array_equal(actual[np.isinf(actual)], expected[np.isinf(actual)])
            and np.array_equal(np.isinf(actual), np.isinf(expected))):
        return np.inf
    keep = np.isfinite(expected)
    scale = max(np.abs(expected[keep]).max(), 1e-300) if keep.any() else 1.0
    return float(np.abs(actual[keep] - expected[keep]).max()/scale) if keep.any() else 0.0


def main() -> int:
    py = np.load(HERE/"py_blackoil_tutorial_spe9.npz")
    ref = loadmat(HERE/"ref_blackoil_tutorial_spe9.mat")
    text = lambda key: str(np.ravel(ref[key])[0])
    ok = True
    ok &= report(str(py["lsolver"]) == text("lsolver") == "CPRSolverAD",
                 f"linear solver {text('lsolver')}")
    ok &= report(str(py["psolver"]) == text("psolver") == "BackslashSolverAD",
                 f"pressure solver {text('psolver')}")
    iterations = int(np.ravel(ref["cpr_iterations"])[0])
    ok &= report(int(py["cpr_iterations"]) == iterations,
                 f"first system: CPR stops after {iterations} GMRES iterations")
    res, res_ref = float(py["cpr_residual"]), float(np.ravel(ref["cpr_residual"])[0])
    ok &= report(abs(res - res_ref) < CPR_RTOL*res_ref + 1e-12,
                 f"first system: CPR residual {res:.6e} (MRST {res_ref:.6e})")
    err = relative(py["cpr_dx"], np.ravel(ref["cpr_dx"]))
    ok &= report(err < CPR_RTOL, f"first system: CPR increment, rel. diff {err:.1e}")
    err = relative(py["direct_dx"], np.ravel(ref["direct_dx"]))
    ok &= report(err < 1e-9, f"first system: direct increment, rel. diff {err:.1e}")
    ok &= report(np.array_equal(py["column"] + 1, np.ravel(ref["column"])),
                 "the column ii == 1 & jj == 1 (MRST's indices are 1-based)")
    for key in SECTIONS:
        actual, expected = py[key], np.asarray(ref[key], dtype=float)
        actual = actual.reshape(expected.shape) if actual.size == expected.size else actual
        err = relative(actual, expected) if actual.shape == expected.shape else np.inf
        ok &= report(err < EXACT_RTOL, f"{key} rel. diff {err:.1e}")
    ok &= report(np.array_equal(py["saturated"], ref["saturated"].astype(bool)),
                 "saturated oil table points")
    strict = KNIFE_EDGE_STEP - 1
    for key in ("its", "nmini"):
        same = np.array_equal(py[key][:strict].astype(int),
                              np.ravel(ref[key])[:strict].astype(int))
        ok &= report(same, f"{key} identical over steps 1-{strict}")
    types = [str(np.ravel(t)[0]) for t in np.ravel(ref["types"])]
    ok &= report(py["types"].tolist() == types, "well controls identical at every step")
    ok &= report(np.array_equal(py["ctrls"], ref["ctrls"]), "the control map")
    names = [str(np.ravel(n)[0]) for n in np.ravel(ref["names"])]
    ok &= report(py["names"].tolist() == names, f"well names, {len(names)} wells")
    for key in RUN:
        actual, expected = py[key], np.asarray(ref[key], dtype=float)
        if actual.shape != expected.shape:
            ok &= report(False, f"{key} shape {actual.shape}, MRST {expected.shape}")
            continue
        err = relative(actual[:strict], expected[:strict])
        ok &= report(err < RUN_RTOL, f"{key} rel. diff {err:.1e} over steps 1-{strict}")
    k = KNIFE_EDGE_STEP - 1
    its = int(py["knife_its"]) + FAILED_ATTEMPT_ITERATIONS
    ok &= report(its == int(np.ravel(ref["its"])[k]),
                 f"step {KNIFE_EDGE_STEP} on MRST's ministeps: {int(py['knife_its'])} + "
                 f"{FAILED_ATTEMPT_ITERATIONS} iterations (MRST {int(np.ravel(ref['its'])[k])})")
    for key, rkey in (("knife_p", "p"), ("knife_rs", "rs_states"), ("knife_sat", "sat"),
                      ("knife_bhp", "bhp")):
        err = relative(py[key], np.asarray(ref[rkey], dtype=float)[k])
        ok &= report(err < KNIFE_RTOL,
                     f"step {KNIFE_EDGE_STEP} on MRST's ministeps: {rkey} rel. diff {err:.1e}")
    print(f"info the run's own step {KNIFE_EDGE_STEP}: {int(py['its'][k])} iterations, "
          f"{int(py['nmini'][k])} ministep(s) (MRST {int(np.ravel(ref['its'])[k])}, "
          f"{int(np.ravel(ref['nmini'])[k])})")
    for key in RUN:
        actual, expected = py[key], np.asarray(ref[key], dtype=float)
        if actual.shape == expected.shape:
            print(f"info {key} rel. diff {relative(actual[k:], expected[k:]):.1e} over steps "
                  f"{KNIFE_EDGE_STEP}-{expected.shape[0]}")
    print("\nblackoilTutorialSPE9 matches MRST" if ok
          else "\nblackoilTutorialSPE9 does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
