"""Validate MRST timestepControlDemo through the AD-OO port.

The reference is MRST 2024a's demo without its figures
(`dump_timestep_control_demo_reference.m`). Checked: the compressed
schedule; for the base run and each iteration target (4, 8, 15, 25), the
ministeps the `IterationCountTimeStepSelector` took, the total Newton
iterations, and the BHP and gas rate of both wells at every (mini)step.
SPE1 matches MRST to round-off, so the step lengths and well curves are
checked to `RTOL`, and the step counts and iterations exactly.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.spe1.timestep_control_demo import (
    timestep_control_demo,
)
from pymrst.modules.ad_core.utils.well_sol_to_vector import well_sol_to_vector
from pymrst.params.gravity.gravity import isolated_gravity

HERE = Path(__file__).resolve().parent
RTOL = 1e-8


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def relative(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.abs(actual - expected).max()/max(np.abs(expected).max(), 1e-300))


def main() -> int:
    ref = loadmat(HERE/"ref_timestep_control_demo.mat")
    with isolated_gravity():
        out = timestep_control_demo(plot=False)
    ok = True
    ok &= report(np.allclose(out.schedule_small.step_val, np.ravel(ref["small_val"]),
                             rtol=1e-14, atol=0)
                 and np.array_equal(out.schedule_small.step_control + 1,
                                    np.ravel(ref["small_control"])),
                 f"compressed schedule: {out.schedule_small.step_val.size} control periods")
    iterations = np.ravel(ref["iterations"]).astype(int)
    ok &= report(np.array_equal(out.iterations.astype(int), iterations),
                 f"Newton iterations per run {iterations.tolist()}")
    for i, label in enumerate(out.labels):
        expected = np.ravel(ref["timesteps"][i, 0])
        actual = out.timesteps[i]
        same_count = actual.size == expected.size
        ok &= report(same_count, f"{label}: {expected.size} steps")
        if same_count:
            err = relative(actual, expected)
            ok &= report(err < RTOL, f"{label}: step lengths, rel. diff {err:.1e}")
        _, _, qGs, bhp = well_sol_to_vector(out.wsols[i])
        for name, actual_w, expected_w in (("bhp", bhp, ref["bhp"][i, 0]),
                                           ("qGs", qGs, ref["qGs"][i, 0])):
            expected_w = np.asarray(expected_w, dtype=float)
            if actual_w.shape != expected_w.shape:
                ok &= report(False, f"{label}: {name} shape {actual_w.shape} vs "
                                    f"{expected_w.shape}")
                continue
            err = relative(actual_w, expected_w)
            ok &= report(err < RTOL, f"{label}: {name}, rel. diff {err:.1e}")
    print("\ntimestepControlDemo matches MRST" if ok
          else "\ntimestepControlDemo does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
