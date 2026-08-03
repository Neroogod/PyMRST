"""Phase 133: compositionalValidationSimple packed-output parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.compositional_validation_simple import (  # noqa: E402
    compare_natural_overall,
    run_compositional_validation_packed,
    run_compositional_validation_simple,
    setup_compositional_validation_simple_problem,
    ternary_displacement_paths,
    validation_comparison_data,
)
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    setup = setup_compositional_validation_simple_problem(
        use_natural=True,
        nstep=2,
        dt=10.0 * DAY,
    )
    packed = run_compositional_validation_packed(
        use_natural=True,
        nstep=2,
        dt=10.0 * DAY,
    )
    simple = run_compositional_validation_simple(
        use_natural=False,
        nstep=2,
        dt=10.0 * DAY,
    )
    both = compare_natural_overall(nstep=2, dt=10.0 * DAY)
    comparison = validation_comparison_data(packed)
    ternary = ternary_displacement_paths(packed["result"].states, every=1)

    print("\nPhase 133: validation-simple packed output parity")
    failures = 0
    failures += _check("setup builds MRST-style packed problem",
                       setup["problem"].directory == "simple_comp"
                       and setup["problem"].name == "Natural"
                       and setup["schedule"].step.val.size == 2
                       and np.all(setup["schedule"].step.control == 1))
    # Asserted as an invariant rather than as one machine's configuration. The
    # E300 and AD-GPRS histories come from MRST's optional `simplecomp`
    # archive, which lives in the MRST tree rather than in this repo, so
    # whether they load is a property of the checkout and not of the port. The
    # previous form pinned both slots to None and so failed on any machine that
    # actually had the dataset installed - which this one does. The module's own
    # self-test was corrected the same way; this validator was not.
    failures += _check("external simulator slots are present and typed",
                       set(setup["external_references"]) == {"E300", "AD-GPRS"})
    failures += _check("packed runner returns MRST helper tuple shape",
                       packed["names"] == ["Natural"]
                       and len(packed["states"]) == 1
                       and len(packed["reports"]) == 1
                       and packed["result"].iterations.size == 2)
    failures += _check("overall direct runner still returns histories",
                       simple["name"] == "Overall"
                       and simple["pressure"].shape[0] == 2
                       and simple["composition"].shape[1:] == simple["state"].z.shape)
    failures += _check("natural/overall comparison preserves both formulations",
                       set(both) == {"natural", "overall"}
                       and both["natural"]["name"] == "Natural"
                       and both["overall"]["name"] == "Overall")
    failures += _check("comparison data exposes MRST/E300/AD-GPRS slots",
                       comparison["names"] == ("MRST", "E300", "AD-GPRS")
                       and comparison["data"]["MRST"] is packed["result"].states
                       and "MRST" in comparison["available"]
                       # listed as available exactly when its data loaded
                       and all((name in comparison["available"])
                               == (comparison["data"][name] is not None)
                               for name in ("E300", "AD-GPRS")))
    failures += _check("ternary displacement paths match MRST map shape",
                       ternary["vertices"].shape == (4, 2)
                       and len(ternary["paths"]) == 2
                       and all(path.shape == (setup["setup"].grid.cells.num, 2)
                               for path in ternary["paths"])
                       and all(np.all(np.isfinite(path)) for path in ternary["paths"]))

    if failures:
        print(f"\nPhase 133 failed ({failures} checks).")
        return 1
    print("\nValidation-simple example now exposes MRST-shaped packed outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
