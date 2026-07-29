"""Validate ``initEclipseProblemAD`` end to end against MRST SPE1."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.ad_core.initialization import init_eclipse_problem_ad


def main() -> int:
    here = Path(__file__).parent
    root = here.parents[2]
    deck = root / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
    ref = scipy.io.loadmat(
        here / "ref_ad_core_init_eclipse_problem.mat", squeeze_me=True
    )
    state, model, schedule, nonlinear = init_eclipse_problem_ad(deck)
    ml_s = np.atleast_2d(np.asarray(ref["s"], float))
    checks = {
        "pressure": (state["pressure"], ref["pressure"], 1e-8),
        "water saturation": (state["sW"], ml_s[:, 0], 1e-12),
        "oil saturation": (state["sO"], ml_s[:, 1], 1e-12),
        "gas saturation": (state["sG"], ml_s[:, 2], 1e-12),
        "rs": (state["rs"], ref["rs"], 1e-10),
        "schedule steps": (schedule.step_val, ref["step_val"], 1e-12),
        "schedule controls": (
            schedule.step_control + 1, ref["step_control"], 0.0
        ),
    }
    failed = 0
    for name, (actual, expected, tolerance) in checks.items():
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = float(np.max(np.abs(actual - expected)))
        scale = max(float(np.max(np.abs(expected))), 1.0)
        ok = error <= tolerance * scale
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    ok = nonlinear.max_iterations == int(ref["max_iterations"])
    print(f"{'ok  ' if ok else 'FAIL'} nonlinear max iterations: {nonlinear.max_iterations}")
    failed += not ok
    print(f"model: {model.__class__.__name__} (MRST {ref['model_class']})")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
