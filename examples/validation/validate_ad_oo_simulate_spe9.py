"""Validate MRST simulateSPE9 through the AD-OO port.

simulateSPE9 builds the SPE9 deck (9000 cells, 26 wells) into a packed
problem with `'useMex', true, 'rowMajorAD', true`: the diagonal row-major
AD backend and, since MRST's `useMEX` setting is off by default (so AMGCL
is skipped), GMRES with ILU(0). The references are MRST 2024a's own runs
(`dump_simulate_spe9_reference.m`) with MRST's non-MEX kernels, which give
the same numbers: MRST's C++ kernels do not compile on this machine. Two
runs are compared:
- **published**: the example as shipped (GMRES-ILU(0), tolerance 1e-4);
- **direct**: the same problem with `BackslashSolverAD`.

The Python runs take about 13 minutes together, so they are read from
`py_simulate_spe9_{published,direct}.npz`, which
`run_simulate_spe9_python.py` writes.

Three things are checked.

1. **The model is exact.** The first linearised system (initial state, a
   one-day step; `dump_spe9_first_system_reference.m`) is MRST's to
   round-off: every equation's residual, the Jacobian, and so A\\b.
2. **The runs take the same path.** Newton iterations and ministeps per
   report step, and every well's control at every step, are identical.
3. **The runs agree to the direct solver's round-off, amplified by
   Newton.** SciPy's SuperLU and MATLAB's backslash solve the same SPE9
   system to 1.5e-12 relative of each other. Newton's chopped updates
   (`dpMaxRel` 0.2, `dsMaxAbs` 0.2) do not damp that difference, so after
   report step 1 (19 iterations) the pressures differ by 4e-9. That is
   round-off, not a model difference: changing only SuperLU's column
   ordering moves Python's own step-1 pressure by the same 4e-9, and with
   the MMD ordering Python matches MATLAB to 1.6e-11. Over the 35 steps
   the difference reaches 2e-6 in the well rates; states and wells are
   checked to `ROUNDOFF_RTOL`. The published GMRES runs stop at a relative
   residual of 1e-4 and agree to the same tolerance.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
SYSTEM_RTOL = 1e-12
ROUNDOFF_RTOL = 1e-5
FIELDS = ("p", "rs", "s", "qWs", "qOs", "qGs", "bhp")
EXPECTED_SOLVER = {"published": "GMRES_ILUSolverAD", "direct": "BackslashSolverAD"}


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def relative(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.abs(actual - expected).max()/max(np.abs(expected).max(), 1e-300))


def first_system() -> bool:
    from pymrst.modules.ad_blackoil.examples.utils.local_dataset_path import local_dataset_path
    from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad,
    )
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.utils.adi import value
    from pymrst.utils.units import DAY

    ref = loadmat(HERE/"ref_spe9_first_system.mat")
    fn = local_dataset_path("spe9", "BENCH_SPE9.DATA")/"BENCH_SPE9.DATA"
    with isolated_gravity():
        state0, model, schedule, _ = init_eclipse_problem_ad(fn, useMex=True, rowMajorAD=True)
        forces = model.get_driving_forces(schedule.controls[0])
        model = model.validate_model(forces)
        state = model.validate_state(state0)
        model, state = model.update_for_changed_controls(state, forces)
        model, state = model.prepare_report_step(state, state, DAY, forces)
        model, st = model.prepare_timestep(state, state, DAY, forces)
        problem, _ = model.get_equations(state, st, DAY, forces, iteration=1)
    ok = True
    names = [str(np.ravel(n)[0]) for n in np.ravel(ref["names"])]
    ok &= report(names == list(problem.equation_names), f"equations {names}")
    vals = np.ravel(ref["vals"])
    for name, eq, expected in zip(names, problem.equations, vals):
        err = relative(np.ravel(value(eq)), np.ravel(expected))
        ok &= report(err < SYSTEM_RTOL, f"first system: {name} residual, rel. diff {err:.1e}")
    problem = problem.assemble_system()
    A_ref = sp.csr_matrix(ref["A"])
    err = abs(sp.csr_matrix(problem.A) - A_ref).max()/abs(A_ref).max()
    ok &= report(err < SYSTEM_RTOL, f"first system: Jacobian, rel. diff {err:.1e}")
    # MRST's right-hand side, -r.
    err = relative(np.ravel(problem.b), np.ravel(ref["b"]))
    ok &= report(err < SYSTEM_RTOL, f"first system: right-hand side, rel. diff {err:.1e}")
    return ok


def compare_run(variant: str) -> bool:
    py = np.load(HERE/f"py_simulate_spe9_{variant}.npz")
    ref = loadmat(HERE/f"ref_simulate_spe9_{variant}.mat")
    ok = True
    for key in ("lsolver", "backend", "basename"):
        expected = str(np.ravel(ref[key])[0])
        ok &= report(str(py[key]) == expected, f"{variant}: {key} {expected}")
    ok &= report(str(py["lsolver"]) == EXPECTED_SOLVER[variant],
                 f"{variant}: linear solver is {EXPECTED_SOLVER[variant]}")
    names = [str(np.ravel(n)[0]) for n in np.ravel(ref["names"])]
    ok &= report(py["names"].tolist() == names, f"{variant}: well names, {len(names)} wells")
    for key in ("its", "nmini"):
        same = np.array_equal(py[key].astype(int), np.ravel(ref[key]).astype(int))
        ok &= report(same, f"{variant}: {key} identical over {py[key].size} steps")
    types = [str(np.ravel(t)[0]) for t in np.ravel(ref["types"])]
    ok &= report(py["types"].tolist() == types, f"{variant}: well controls identical at every step")
    for key in FIELDS:
        actual, expected = py[key], np.asarray(ref[key], dtype=float)
        if actual.shape != expected.shape:
            ok &= report(False, f"{variant}: {key} shape {actual.shape} vs {expected.shape}")
            continue
        err = relative(actual, expected)
        ok &= report(err < ROUNDOFF_RTOL, f"{variant}: {key} rel. diff {err:.1e}")
    return ok


def main() -> int:
    ok = first_system()
    for variant in ("published", "direct"):
        ok &= compare_run(variant)
    print("\nsimulateSPE9 matches MRST" if ok else "\nsimulateSPE9 does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
