"""Phase 239: compositionalValidationSimple plot-section metadata parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.compositional_validation_simple import (  # noqa: E402
    run_compositional_validation_packed,
    validation_plot_sections,
)
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    packed = run_compositional_validation_packed(use_natural=True, nstep=2, dt=10.0 * DAY)
    sections = packed["plot_sections"]
    helper = validation_plot_sections(packed)
    comp = sections["component_comparison"]
    detailed = sections["pressure_saturation_comparison"]
    toolbars = sections["state_toolbars"]
    ternary = sections["ternary_diagram"]

    print("\nPhase 239: compositionalValidationSimple plot-section metadata parity")
    failures = 0
    failures += _check("runner returns all MRST validation plot sections",
                       set(sections) == {"component_comparison", "pressure_saturation_comparison",
                                         "state_toolbars", "ternary_diagram"})
    failures += _check("component comparison carries MRST legend/axis metadata",
                       comp["names"] == ("MRST", "E300", "AD-GPRS")
                       and comp["available"] == ("MRST",)
                       and comp["markers"] == ("-", "--", "--")
                       and comp["line_widths"] == (0.5, 2.5, 5.0)
                       and comp["legend_location"] == "north"
                       and comp["legend_num_columns"] == 3
                       and comp["ylim"] == (0.0, 1.0)
                       and comp["ylabel"] == "z")
    failures += _check("component legend expands simulator/component labels",
                       len(comp["legend"]) == 3 * len(comp["component_names"])
                       and comp["legend"][0].startswith("MRST "))
    failures += _check("pressure/saturation detailed section mirrors MRST comparison",
                       detailed["available"] == ("MRST",)
                       and detailed["legend_location"] == "northoutside"
                       and detailed["legend_orientation"] == "horizontal"
                       and detailed["ylim"] == (0.0, 1.0)
                       and detailed["MRST"]["components"].shape[1] == len(comp["component_names"])
                       and detailed["E300"] is None)
    failures += _check("state toolbar sections represent all simulator slots",
                       tuple(t["name"] for t in toolbars) == ("MRST", "E300", "AD-GPRS")
                       and toolbars[0]["available"] is True
                       and toolbars[1]["available"] is False
                       and all(t["plot1d"] is True for t in toolbars))
    failures += _check("ternary section carries MRST triangle metadata",
                       ternary["vertices"].shape == (4, 2)
                       and ternary["axis"] == "off"
                       and ternary["colormap"] == "parula"
                       and ternary["component_labels"] == comp["component_names"]
                       and len(ternary["half_labels"]) == 3
                       and all(np.asarray(path).shape[1] == 2 for path in ternary["paths"]))
    failures += _check("standalone helper reproduces runner sections",
                       helper["component_comparison"]["legend"] == comp["legend"]
                       and helper["state_toolbars"][0]["name"] == toolbars[0]["name"]
                       and np.allclose(helper["ternary_diagram"]["vertices"], ternary["vertices"]))

    if failures:
        print(f"\nPhase 239 failed ({failures} checks).")
        return 1
    print("\nValidation-simple now exposes MRST comparison, toolbar, and ternary plot metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
