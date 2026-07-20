"""Phase 228: compositional BC example MRST plotting-section parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    boundary_condition_iteration_plot,
    boundary_condition_phase_saturations,
    boundary_condition_plot_fields,
    run_compositional_boundary_conditions_example,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    print("\nPhase 228: compositional boundary-condition plotting-section parity")
    case = run_compositional_boundary_conditions_example(nstep=2)
    plot_fields = case["plot_fields"]
    iteration_plot = case["iteration_plot"]
    manual_fields = boundary_condition_plot_fields(case)
    manual_iterations = boundary_condition_iteration_plot(case)

    failures = 0
    failures += _check("plot section has MRST formulations",
                       plot_fields["formulations"] == ("Natural variables", "Overall composition"))
    failures += _check("plot section has MRST fields",
                       plot_fields["fields"] == ("pressure", "gas_saturation", "component"))
    failures += _check("CO2 component index follows MRST example",
                       plot_fields["component_index"] == 1
                       and plot_fields["component_name"] == "CarbonDioxide")
    failures += _check("one plot frame per report step",
                       len(plot_fields["frames"]) == case["timesteps"].size == 2)
    first = plot_fields["frames"][0]
    nc = case["grid"].cells.num
    failures += _check("natural pressure frame has one value per cell",
                       first["Natural variables"]["pressure"].shape == (nc,))
    failures += _check("overall pressure frame has one value per cell",
                       first["Overall composition"]["pressure"].shape == (nc,))
    failures += _check("natural gas saturation is bounded",
                       np.all((first["Natural variables"]["gas_saturation"] >= -1e-12)
                              & (first["Natural variables"]["gas_saturation"] <= 1.0 + 1e-12)))
    failures += _check("overall gas saturation is bounded",
                       np.all((first["Overall composition"]["gas_saturation"] >= -1e-12)
                              & (first["Overall composition"]["gas_saturation"] <= 1.0 + 1e-12)))
    failures += _check("natural plotted CO2 equals state composition",
                       np.allclose(
                           first["Natural variables"]["component"],
                           case["natural_result"].states[0].z[:, 1],
                           atol=0.0,
                       ))
    failures += _check("overall plotted CO2 equals state composition",
                       np.allclose(
                           first["Overall composition"]["component"],
                           case["result"].states[0].z[:, 1],
                           atol=0.0,
                       ))
    natural_so, natural_sg = boundary_condition_phase_saturations(
        case["natural_model"],
        case["natural_result"].states[-1],
    )
    failures += _check("phase-saturation helper returns oil/gas columns",
                       natural_so.shape == (nc,) and natural_sg.shape == (nc,))
    failures += _check("iteration plot has MRST names",
                       iteration_plot["names"] == ("Natural variables", "Overall composition"))
    failures += _check("iteration plot stores natural/overall columns",
                       iteration_plot["iterations"].shape == (2, 2)
                       and np.array_equal(iteration_plot["iterations"][:, 0],
                                          case["natural_result"].iterations)
                       and np.array_equal(iteration_plot["iterations"][:, 1],
                                          case["result"].iterations))
    failures += _check("cumulative iterations are cumulative",
                       np.array_equal(iteration_plot["cumulative"],
                                      np.cumsum(iteration_plot["iterations"], axis=0)))
    failures += _check("standalone plot helpers reproduce stored output",
                       manual_fields["frames"][0]["Natural variables"]["pressure"].shape == (nc,)
                       and np.array_equal(manual_iterations["iterations"],
                                          iteration_plot["iterations"]))

    if failures:
        print(f"\nPhase 228 validation failed ({failures} checks).")
    else:
        print("\nBoundary-condition example now exposes MRST-shaped plotting sections.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
