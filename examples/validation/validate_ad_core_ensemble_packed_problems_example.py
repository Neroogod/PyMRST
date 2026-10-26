"""Validate MRST ensemblePackedProblemsExample, the port in
`ad_core/examples/ensemble_packed_problems_example.py`.

The reference is MRST 2024a (`dump_ensemble_packed_problems_example_reference.m`):
the script's 50 realizations of a 1D waterflood, each a packed problem. The
porosity fields come from MATLAB's `rng(0)`, which Python cannot reproduce, so
MRST's fields are passed to the port; everything built from them (Carman-Kozeny
permeability, the packed names and descriptions) is the port's own. The
script simulates in background sessions through `PackedProblemManager`; the
reference and the port both simulate in this session (`simulateProblem`),
which gives the same states. Names and descriptions identical; for every
realization, Newton iterations identical at every step, and the outlet water
saturation at every step and the whole field every tenth step to `RUN_RTOL`.
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


def main() -> int:
    from pymrst.modules.ad_core.examples.ensemble_packed_problems_example import (
        ensemble_packed_problems_example,
        setup_ensemble,
    )
    from pymrst.modules.ad_core.simulators.sim_runner.clear_packed_simulator_output import (
        clear_packed_simulator_output,
    )
    ref = loadmat(HERE/"ref_ensemble_packed_problems_example.mat", squeeze_me=True)
    poro = np.asarray(ref["poro"], dtype=float)
    n = poro.shape[1]
    with contextlib.redirect_stdout(io.StringIO()):
        clear_packed_simulator_output(setup_ensemble(n, poro).problems, Prompt=False)
        out = ensemble_packed_problems_example(n, poro=poro, simulate="here")
        clear_packed_simulator_output(out.problems, Prompt=False)
    nstep = out.schedule.step_val.size
    keep = np.atleast_1d(ref["keep"]).astype(int)
    ok = report(list(out.names) == list(ref["names"]), f"{n} names identical")
    ok &= report([p.Description for p in out.problems] == list(ref["descriptions"]),
                 "descriptions identical")
    its = np.array([[r["Iterations"] for r in reps] for reps in out.reports])
    ok &= report(np.array_equal(its, ref["its"].reshape(n, nstep)),
                 f"Newton iterations identical at all {nstep} steps of all realizations "
                 f"({its.sum()})")
    outlet = np.array([[np.asarray(s["s"])[-1, 0] for s in st] for st in out.states])
    ref_outlet = ref["outlet"].reshape(n, nstep)
    err = np.abs(outlet - ref_outlet).max()/np.abs(ref_outlet).max()
    ok &= report(err < RUN_RTOL, f"outlet water saturation rel. diff {err:.1e}")
    sw = np.array([[np.asarray(st[k - 1]["s"])[:, 0] for k in keep] for st in out.states])
    ref_sw = ref["sw"].reshape(n, 100, keep.size).transpose(0, 2, 1)
    err = np.abs(sw - ref_sw).max()/np.abs(ref_sw).max()
    ok &= report(err < RUN_RTOL, f"water saturation every tenth step rel. diff {err:.1e}")
    print("\nensemblePackedProblemsExample matches MRST" if ok
          else "\nensemblePackedProblemsExample does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
