"""Validate MRST simulateSPE1 and simulateSPE3 through the AD-OO port.

Both examples are three lines in MRST: `initEclipsePackedProblemAD`,
`simulatePackedProblem` and `plotPackedProblem`. Underneath, they exercise:
- the deck readers and `initEclipseProblemAD` (grid, rock, fluid,
  `selectModelFromDeck`, `initStateDeck`, `convertDeckScheduleToMRST`);
- `getNonLinearSolver`'s solver choice and iteration-count timestep
  selector;
- the packed problem's on-disk output, read back by
  `getPackedSimulatorOutput`.

SPE3 is a gas condensate (vaporised oil, no dissolved gas). The reference
is MRST 2024a's own run (`dump_simulate_spe13_reference.m`). Checked per
report step: pressure, saturations, `rs`/`rv`, well rates and BHP, Newton
iterations, ministep counts. Also checked: well names, the base name taken
from the deck's TITLE, and the linear solver and AD backend.

SPE3 meets the BHP-limit knife edge at step 21. The producer switches from
gas rate to BHP, and the next Newton step lands its BHP exactly on the
72.39-bar limit. MRST's solve lands just above and snaps to 72.29 bar; this
port's lands 1e-9 Pa below and does not snap. The rest of the run then
differs within the nonlinear tolerance (~1e-5). SPE3 is therefore checked
as the generic sector case is: as run, to round-off before step 21 and to
`KNIFE_EDGE_RTOL` after; and with MRST's side of the round-off taken at a
limit landing, to `RTOL` throughout.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.spe1.simulate_spe1 import simulate_spe1
from pymrst.modules.ad_blackoil.examples.spe3.simulate_spe3 import simulate_spe3
from pymrst.modules.ad_core.simulators.sim_runner.get_packed_simulator_output import (
    get_packed_simulator_output,
)
from pymrst.modules.ad_core.utils.well_sol_to_vector import well_sol_to_vector
from pymrst.params.gravity.gravity import isolated_gravity

from validate_ad_oo_generic_sector_model import mrst_side_of_limit_landings

HERE = Path(__file__).resolve().parent
RTOL = 1e-10
KNIFE_EDGE_RTOL = 1e-4
SPE3_KNIFE_EDGE_STEP = 20                # step 21, 0-based


def run(name, simulate):
    with isolated_gravity():
        problem = simulate(plot=False)
        ws, states, reports, _ = get_packed_simulator_output(problem)
    qWs, qOs, qGs, bhp = well_sol_to_vector(ws)
    out = dict(
        p=np.array([s["pressure"] for s in states]),
        s=np.array([np.asarray(s["s"]).ravel(order="F") for s in states]),
        qWs=qWs, qOs=qOs, qGs=qGs, bhp=bhp,
        its=np.array([r["Iterations"] for r in reports]),
        nmini=np.array([len(r["StepReports"]) for r in reports]))
    for key in ("rs", "rv"):
        if np.size(states[0].get(key, 0)) > 1:
            out[key] = np.array([s[key] for s in states])
    setup = problem.SimulatorSetup
    meta = dict(names=[w.name for w in ws[0]], basename=problem.BaseName,
                lsolver=type(setup.NonLinearSolver.LinearSolver).__name__,
                backend=type(setup.model.AutoDiffBackend).__name__)
    return out, meta


def compare(name, out, meta, ref, steps=slice(None), rtol=RTOL) -> bool:
    ok_all = True
    for key, actual in out.items():
        expected = np.asarray(ref[key], dtype=float)
        actual, expected = actual[steps], expected[steps]
        if key in ("its", "nmini"):
            ok = np.array_equal(actual.astype(int), np.ravel(expected).astype(int))
            print(f"{'ok  ' if ok else 'FAIL'} {name}: {key} identical")
        else:
            err = np.abs(actual - expected).max()/max(np.abs(expected).max(), 1e-300)
            ok = actual.shape == expected.shape and err < rtol
            print(f"{'ok  ' if ok else 'FAIL'} {name}: {key} rel {err:.2e}")
        ok_all &= ok
    backends = {"AutoDiffBackend": "AutoDiffBackend",
                "DiagonalAutoDiffBackend": "DiagonalAutoDiffBackend"}
    checks = {
        "well names": meta["names"] == list(np.atleast_1d(ref["names"])),
        "base name from TITLE": meta["basename"] == str(ref["basename"]),
        "linear solver": meta["lsolver"] == str(ref["lsolver"]),
        "AD backend": backends.get(meta["backend"]) == str(ref["backend"]),
    }
    for label, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {label}")
        ok_all &= ok
    return ok_all


def main() -> int:
    refs = {}
    for name in ("spe1", "spe3"):
        path = HERE/f"ref_simulate_{name}.mat"
        if not path.exists():
            print(f"{path.name} missing (run dump_simulate_spe13_reference.m)")
            return 1
        refs[name] = loadmat(path, squeeze_me=True)
    out, meta = run("spe1", simulate_spe1)
    ok = compare("spe1", out, meta, refs["spe1"])
    out, meta = run("spe3", simulate_spe3)
    before = slice(0, SPE3_KNIFE_EDGE_STEP)
    ok &= compare("spe3, before step 21", out, meta, refs["spe3"], before)
    ok &= compare("spe3, whole run", out, meta, refs["spe3"], rtol=KNIFE_EDGE_RTOL)
    with mrst_side_of_limit_landings():
        out, meta = run("spe3", simulate_spe3)
    ok &= compare("spe3, MRST's side at the BHP limit", out, meta, refs["spe3"])
    return int(not ok)


if __name__ == "__main__":
    raise SystemExit(main())
