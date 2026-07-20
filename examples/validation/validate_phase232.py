"""Phase 232: larger-problem tutorial plotting-section parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.compositional_larger_problem_tutorial import (  # noqa: E402
    larger_problem_plot_sections,
    larger_problem_tick_labels,
    run_compositional_larger_problem_tutorial,
)
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    natural = run_compositional_larger_problem_tutorial(
        dims=(2, 2, 2),
        use_bc=False,
        include_water=False,
        use_natural=True,
        dt=0.01 * DAY,
    )
    overall = run_compositional_larger_problem_tutorial(
        dims=(2, 2, 2),
        use_bc=False,
        include_water=False,
        use_natural=False,
        dt=0.01 * DAY,
    )
    helper = larger_problem_plot_sections(overall["timings"], use_natural=False)

    nat_plot = natural["plot_sections"]["timing_bar"]
    ov_plot = overall["plot_sections"]["timing_bar"]
    ov_zoom = overall["plot_sections"]["assembly_zoom"]
    has_mex_row = any(row.backend == "Diagonal (MEX+RowMajor)" and row.available
                      for row in overall["timings"])

    print("\nPhase 232: larger-problem tutorial plotting-section parity")
    failures = 0
    failures += _check("figure index follows MRST 1 + useNatural",
                       nat_plot["figure"] == 2 and ov_plot["figure"] == 1)
    failures += _check("timing plot carries MRST legend",
                       ov_plot["legend"] == ("Equation assembly", "Linear solver", "Total time")
                       and ov_plot["legend_location"] == "NorthWest")
    failures += _check("timing plot tick labels follow MRST MEX branch",
                       ov_plot["x_tick_labels"] == larger_problem_tick_labels(has_mex_row))
    failures += _check("timing plot matrix matches returned time matrix",
                       np.allclose(ov_plot["time"], overall["time_matrix"])
                       and ov_plot["time"].shape[1] == 3)
    failures += _check("assembly zoom uses first timing column",
                       np.allclose(ov_zoom["assembly_time"], overall["time_matrix"][:, 0])
                       and np.isclose(ov_zoom["ylim"][0], 0.0)
                       and np.isclose(ov_zoom["ylim"][1],
                                      1.2 * np.max(overall["time_matrix"][:, 0])))
    failures += _check("standalone plot helper is identical to runner payload",
                       helper["timing_bar"]["x_tick_labels"] == ov_plot["x_tick_labels"]
                       and np.allclose(helper["timing_bar"]["time"], ov_plot["time"])
                       and helper["assembly_zoom"]["ylim"] == ov_zoom["ylim"])

    if failures:
        print(f"\nPhase 232 failed ({failures} checks).")
        return 1
    print("\nLarger-problem tutorial now exposes MRST's timing and assembly-zoom plot sections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
