"""Phase 234: showNaturalOverall convergence-plot parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.book_ii.show_natural_overall import (  # noqa: E402
    run_show_natural_overall,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    out = run_show_natural_overall(max_change=0.1)
    tight = run_show_natural_overall(max_change=0.05)
    comp = out["convergence_plots"]["composition_pressure"]
    sat = out["convergence_plots"]["saturation_pressure"]
    tight_comp = tight["convergence_plots"]["composition_pressure"]

    print("\nPhase 234: showNaturalOverall convergence-plot parity")
    failures = 0
    failures += _check("two MRST convergence figures are exposed",
                       comp["figure"] == 1
                       and sat["figure"] == 2
                       and comp["xlabel"].startswith("z_")
                       and sat["xlabel"] == "S_L")
    failures += _check("flash contour grids have MRST shape",
                       comp["dp_grid"].shape == (50, 50)
                       and comp["x_grid"].shape == (50, 50)
                       and comp["saturation_contours"].shape == (50, 50)
                       and sat["x_grid"].shape == (50, 50))
    failures += _check("natural and overall paths include initial/final states",
                       comp["natural_x"].shape == (2,)
                       and comp["overall_x"].shape == (2,)
                       and comp["natural_dp"].shape == (2,)
                       and sat["natural_x"].shape == (2,)
                       and sat["overall_x"].shape == (2,))
    failures += _check("solution marker follows overall final state",
                       np.isclose(comp["solution_x"], comp["overall_x"][-1])
                       and np.isclose(comp["solution_dp"], comp["overall_dp"][-1])
                       and np.isclose(sat["solution_x"], sat["overall_x"][-1]))
    failures += _check("legend and axes metadata match MRST script",
                       comp["legend"] == ("Natural variables", "Overall composition", "Solution")
                       and comp["legend_location"] == "southeast"
                       and comp["ylabel"] == r"\Delta p [Pa]")
    failures += _check("line style follows maxChange branch",
                       comp["line_style"] == "-o"
                       and tight_comp["line_style"] == "-")

    if failures:
        print(f"\nPhase 234 failed ({failures} checks).")
        return 1
    print("\nshowNaturalOverall now exposes MRST's convergence contour plot payloads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
