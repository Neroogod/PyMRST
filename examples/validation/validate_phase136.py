"""Phase 136: compositional miscibility example section-data parity."""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.examples.book_ii.compositional_miscibility_example import (
    MRST_MISCIBILITY_PRESSURES_BAR,
    miscibility_formulation_iterations,
    miscibility_front_animation,
    run_compositional_miscibility_example,
    setup_miscibility_test,
)
from pymrst.utils.units import BARSA


def _check(name: str, ok: bool) -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return 0 if ok else 1


def main() -> int:
    state0, model, schedule = setup_miscibility_test(90.0 * BARSA, ncell=7, nstep=4)
    run = run_compositional_miscibility_example(
        pressures_bar=[70.0, 90.0],
        ncell=6,
        nstep=4,
        compare_formulations=False,
    )
    front = miscibility_front_animation(run["packed_problems"], run["states"])
    table = miscibility_formulation_iterations({
        "natural_iterations": np.array([4, 5, 4]),
        "overall_iterations": np.array([3, 3, 4]),
    })

    print("\nPhase 136: compositional miscibility section-data parity")
    failures = 0
    failures += _check("setupMiscibilityTest keeps MRST constants",
                       model.mix.names == ("CO2", "C1", "C5", "C10")
                       and model.g.cells.num == 7
                       and schedule.step.val.size == 4
                       and np.allclose(state0.z[0], [0.33, 0.03, 0.24, 0.40]))
    failures += _check("default full MRST pressure list is recorded",
                       np.allclose(MRST_MISCIBILITY_PRESSURES_BAR,
                                   [70, 90, 100, 110, 120, 130, 150]))
    failures += _check("packed pressure sweep preserves MRST names",
                       run["names"] == ["70", "90"]
                       and [p.directory for p in run["packed_problems"]] == ["MiscibilityDemo"] * 2)
    failures += _check("front animation exposes x-coordinate and report frames",
                       front["x"].shape == (6,)
                       and np.array_equal(front["steps"], np.arange(1, 5))
                       and len(front["frames"]) == 4)
    failures += _check("front frames contain gas saturation per pressure case",
                       set(front["frames"][0]["profiles"]) == {"70", "90"}
                       and all(v.shape == (6,) for v in front["frames"][0]["profiles"].values())
                       and all(np.all(v >= 0.0) for v in front["frames"][0]["profiles"].values()))
    failures += _check("runner returns front animation section",
                       "front_animation" in run
                       and run["front_animation"]["names"] == ("70", "90"))
    failures += _check("coarse timestep formulation comparison has MRST bar data",
                       table["names"] == ("Natural variables", "Overall composition")
                       and table["iterations"].shape[1] == 2
                       and table["cumulative"].shape == table["iterations"].shape)
    failures += _check("comparison branch remains optional for scaled runs",
                       run["comparison"] is None)

    if failures:
        print(f"\nPhase 136 failed ({failures} checks).")
    else:
        print("\nMiscibility example now exposes MRST-shaped animation and iteration data.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
