"""Validate MRST demoPackedProblems, the port in
`ad_core/examples/demo_packed_problems.py`.

The reference is MRST 2024a running the script up to its figure
(`dump_demo_packed_problems_reference.m`): two 1D waterfloods packed under one
base name (linear and quadratic relative permeabilities), simulated with
`simulatePackedProblem` and read back with `getMultiplePackedSimulatorOutputs`,
old output cleared first. Both runs ok; names and report times identical;
Newton iterations identical at every step; well rates, BHP and water saturation
at every step to `RUN_RTOL`. The port then clears the output as the script does
(without the prompt) and stops before the background runs, which the
end-to-end test of `simulatePackedProblemBackground` covers.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RUN_RTOL = 1e-10


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def main() -> int:
    from pymrst.modules.ad_core.examples.demo_packed_problems import (
        demo_packed_problems,
        pack_demo_problems,
    )
    from pymrst.modules.ad_core.simulators.sim_runner.clear_packed_simulator_output import (
        clear_packed_simulator_output,
    )
    ref = loadmat(HERE/"ref_demo_packed_problems.mat", squeeze_me=True)
    with contextlib.redirect_stdout(io.StringIO()):
        clear_packed_simulator_output(pack_demo_problems(), Prompt=False)
        out = demo_packed_problems(prompt=False, background=False)
    ok = report(list(out.ok) == [True, True], "both packed problems simulated")
    ok &= report(list(out.names) == list(ref["names"]), f"names {list(out.names)}")
    err = max(_rel(t, r) for t, r in zip(out.report_time, ref["report_time"]))
    ok &= report(err < 1e-15, f"report times rel. diff {err:.1e}")
    for i in (1, 2):
        tag = out.names[i - 1] + ":"
        its = np.array([r["Iterations"] for r in out.all_reports[i - 1]])
        ok &= report(np.array_equal(its, ref[f"its{i}"]),
                     f"{tag} Newton iterations identical at all {its.size} steps ({its.sum()})")
        ws = out.all_ws[i - 1]
        for field in ("qWs", "qOs", "bhp"):
            err = _rel([[getattr(w, field) for w in row] for row in ws], ref[f"{field}{i}"])
            ok &= report(err < RUN_RTOL, f"{tag} well {field} rel. diff {err:.1e}")
        sw = [np.asarray(s["s"])[:, 0] for s in out.all_states[i - 1]]
        err = _rel(sw, ref[f"sw{i}"])
        ok &= report(err < RUN_RTOL, f"{tag} water saturation rel. diff {err:.1e}")
    ok &= report(all(p.OutputHandlers.states.numel_data() == 0 for p in out.problems),
                 "output cleared afterwards")
    print("\ndemoPackedProblems matches MRST" if ok else "\ndemoPackedProblems does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
