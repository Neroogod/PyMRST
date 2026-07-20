"""Phase 235: compositional accuracy example plot-section metadata parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.book_ii.compositional_accuracy_example import (  # noqa: E402
    accuracy_plot_sections,
    run_compositional_accuracy_example,
)
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    out = run_compositional_accuracy_example(nx=6, nstep=3, dt=5.0 * DAY)
    sections = out["plot_sections"]
    helper = accuracy_plot_sections(out)
    component = sections["component_profiles"]
    gas = sections["gas_saturation"]
    props = sections["properties"]
    cumulative = sections["cumulative_iterations"]

    print("\nPhase 235: compositional accuracy plot-section metadata parity")
    failures = 0
    failures += _check("runner returns all four MRST plot sections",
                       set(sections) == {"component_profiles", "gas_saturation",
                                         "properties", "cumulative_iterations"})
    failures += _check("component profile section has MRST legend and axis metadata",
                       component["legend_location"] == "north"
                       and component["legend_num_columns"] == 3
                       and component["ylim"] == (0.0, 1.0)
                       and component["ylabel"] == "z"
                       and component["font_size"] == 12)
    failures += _check("component legend expands case/component labels",
                       len(component["legend"]) == len(out["names"]) * len(out["component_profiles"]["component_names"])
                       and component["legend"][0].startswith("Explicit "))
    failures += _check("gas saturation section matches MRST plot semantics",
                       gas["legend"] == tuple(out["names"])
                       and gas["legend_location"] == "north"
                       and gas["legend_num_columns"] == 3
                       and gas["ylim"] == (0.0, 1.0)
                       and gas["ylabel"] == "z")
    failures += _check("property section carries MRST dual-axis labels",
                       props["legend"] == (r"\mu_l", r"\mu_v", r"\rho_l", r"\rho_v", "S_g", "z (CO_2)")
                       and props["legend_location"] == "east"
                       and props["legend_num_columns"] == 3
                       and props["left_ylabel"] == "Viscosity [cP]"
                       and props["right_ylabel"] == "Mass density [kg/m^3]")
    failures += _check("cumulative iteration section mirrors MRST final plot",
                       cumulative["legend"] == ("Natural", "Overall")
                       and cumulative["title"] == "Cumulative number of iterations"
                       and cumulative["cumulative"].shape == (3, 2)
                       and np.all(np.diff(cumulative["cumulative"], axis=0) >= 0))
    failures += _check("standalone helper reproduces runner plot sections",
                       helper["component_profiles"]["legend"] == component["legend"]
                       and helper["gas_saturation"]["legend"] == gas["legend"]
                       and np.allclose(helper["cumulative_iterations"]["cumulative"],
                                       cumulative["cumulative"]))

    if failures:
        print(f"\nPhase 235 failed ({failures} checks).")
        return 1
    print("\nAccuracy example now exposes MRST's plotting section metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
