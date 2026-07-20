"""Phase 236: miscibility example plot-section metadata parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.book_ii.compositional_miscibility_example import (  # noqa: E402
    miscibility_plot_sections,
    run_compositional_miscibility_example,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    run = run_compositional_miscibility_example(
        pressures_bar=[70.0, 90.0],
        ncell=6,
        nstep=4,
        compare_formulations=False,
    )
    sections = run["plot_sections"]
    helper = miscibility_plot_sections(run)
    front = sections["front_animation"]
    formulation = sections["formulation_iterations"]

    print("\nPhase 236: miscibility plot-section metadata parity")
    failures = 0
    failures += _check("runner returns MRST miscibility plot sections",
                       set(sections) == {"front_animation", "formulation_iterations"})
    failures += _check("front animation section carries MRST figure metadata",
                       front["figure"] == 1
                       and front["colormap"] == "summer"
                       and front["color_count"] == len(run["names"])
                       and np.isclose(front["line_width"], 2.0))
    failures += _check("front animation section keeps x and per-step profiles",
                       front["x"].shape == (6,)
                       and np.array_equal(front["steps"], np.arange(1, 5))
                       and len(front["frames"]) == 4
                       and set(front["frames"][0]["profiles"]) == {"70", "90"})
    failures += _check("front animation axis labels are explicit",
                       front["xlabel"] == "x" and front["ylabel"] == "S_g")
    failures += _check("formulation branch reports unavailable comparison honestly",
                       formulation["iterations"] is None
                       and formulation["cumulative"] is None
                       and formulation["legend"] == ("Natural variables", "Overall composition")
                       and formulation["bar_width"] == 2.0
                       and formulation["reason"] == "comparison branch not run")
    failures += _check("standalone helper reproduces runner sections",
                       helper["front_animation"]["names"] == front["names"]
                       and helper["front_animation"]["color_count"] == front["color_count"]
                       and helper["formulation_iterations"]["reason"] == formulation["reason"])

    if failures:
        print(f"\nPhase 236 failed ({failures} checks).")
        return 1
    print("\nMiscibility example now exposes MRST animation and iteration-plot metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
