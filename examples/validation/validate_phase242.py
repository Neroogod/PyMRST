"""Phase 242: boundary-condition example full plot metadata parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    boundary_condition_plot_sections,
    run_compositional_boundary_conditions_example,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    case = run_compositional_boundary_conditions_example(nstep=2)
    sections = case["plot_sections"]
    helper = boundary_condition_plot_sections(case)
    perm = sections["permeability"]
    toolbars = sections["toolbars"]
    comparison = sections["comparison_animation"]
    iterations = sections["iteration_plot"]

    print("\nPhase 242: boundary-condition full plot metadata parity")
    failures = 0
    failures += _check("runner returns all MRST plot sections",
                       set(sections) == {"permeability", "toolbars",
                                         "comparison_animation", "iteration_plot"})
    failures += _check("permeability section mirrors MRST figure",
                       perm["title"] == "Permeability [mD]"
                       and perm["colorbar"] is True
                       and perm["grid_cells"] == case["grid"].cells.num
                       and np.array_equal(perm["field"], case["permeability_md"]))
    failures += _check("toolbar sections mirror natural/overall plotToolbar calls",
                       tuple(t["title"] for t in toolbars) == ("Natural variables", "Overall composition")
                       and toolbars[0]["states"] == case["natural_result"].states
                       and toolbars[1]["states"] == case["result"].states
                       and all(t["grid_cells"] == case["grid"].cells.num for t in toolbars))
    failures += _check("comparison animation carries MRST subplot styling",
                       comparison["subplot_layout"] == (3, 2)
                       and comparison["formulations"] == ("Natural variables", "Overall composition")
                       and comparison["edge_color"] == "none"
                       and comparison["axis"] == "tight"
                       and comparison["hide_tick_labels"] is True
                       and comparison["right_bottom_axis"] == "tight off"
                       and comparison["drawnow"] is True)
    failures += _check("comparison rows carry MRST labels and caxis settings",
                       tuple(r["field"] for r in comparison["rows"]) == ("pressure", "gas_saturation", "component")
                       and tuple(r["ylabel"] for r in comparison["rows"]) == ("Pressure", "Gas saturation", "CarbonDioxide")
                       and tuple(r["caxis"] for r in comparison["rows"]) == (None, (0.0, 1.0), (0.0, 1.0)))
    failures += _check("comparison frames reuse validated plot fields",
                       comparison["frames"] == case["plot_fields"]["frames"]
                       and len(comparison["frames"]) == 2)
    failures += _check("iteration plot carries MRST line/legend/axis labels",
                       iterations["line_style"] == "-o"
                       and iterations["legend"] == ("Natural variables", "Overall composition")
                       and iterations["xlabel"] == "Step #"
                       and iterations["ylabel"] == "Nonlinear iterations"
                       and np.array_equal(iterations["data"]["iterations"],
                                          case["iteration_plot"]["iterations"]))
    failures += _check("standalone helper reproduces runner sections",
                       helper["comparison_animation"]["rows"] == comparison["rows"]
                       and helper["iteration_plot"]["line_style"] == "-o")

    if failures:
        print(f"\nPhase 242 failed ({failures} checks).")
        return 1
    print("\nBoundary-condition example now exposes MRST plotToolbar, comparison, and iteration metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
