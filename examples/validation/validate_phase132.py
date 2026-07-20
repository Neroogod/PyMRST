"""Phase 132: MRST-shaped 3D six-component example port."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.compositional_example_3d_six_components import (  # noqa: E402
    extract_plot_fields,
    run_compositional_example_3d_six_components,
    setup_3d_six_component_case,
)
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel  # noqa: E402
from pymrst.utils.units import BARSA, DAY, YEAR  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    case = setup_3d_six_component_case((3, 3, 2))
    full = setup_3d_six_component_case(full_size=True)
    out = run_compositional_example_3d_six_components((2, 2, 2), nstep=1)
    fields = extract_plot_fields(out)

    print("\nPhase 132: 3D six-component example parity")
    failures = 0
    failures += _check("scaled case uses MRST lumped_1 six-component mixture",
                       case["case_name"] == "lumped_1"
                       and case["mixture"].n == 6
                       and case["mixture"].names[1] == "CO2")
    failures += _check("scaled case uses MRST pressure range and initial pressure",
                       np.isclose(case["min_pressure"], 100.0 * BARSA)
                       and np.isclose(case["max_pressure"], 200.0 * BARSA)
                       and np.isclose(case["pressure0"], 150.0 * BARSA))
    failures += _check("full-size option matches MATLAB dimensions",
                       full["dims"] == (20, 20, 2)
                       and full["grid"].cells.num == 800)
    failures += _check("natural-variable model is default",
                       case["use_natural"]
                       and isinstance(case["model"], NaturalVariablesCompositionalModel))
    failures += _check("corner vertical wells match MRST controls",
                       len(case["wells"]) == 2
                       and case["wells"][0].name == "Inj"
                       and case["wells"][0].type == "rate"
                       and case["wells"][0].sign == 1
                       and np.isclose(case["wells"][0].val, 0.0015)
                       and case["wells"][1].name == "Prod"
                       and case["wells"][1].type == "bhp"
                       and case["wells"][1].sign == -1
                       and np.isclose(case["wells"][1].val, case["min_pressure"]))
    failures += _check("all well streams use benchmark injection composition",
                       np.allclose(case["wells"][0].components, case["info"].injection)
                       and np.allclose(case["wells"][1].components, case["info"].injection))
    failures += _check("ramped two-year schedule matches MRST",
                       np.isclose(case["schedule"].step.val.sum(), 2.0 * YEAR)
                       and np.isclose(case["schedule"].step.val[0], 7.0 * DAY / 2.0**5)
                       and np.all(case["schedule"].step.control == 1))
    failures += _check("one report step runs and returns plot fields",
                       out["iterations"].shape == (1,)
                       and fields["components"].shape == (8, 6)
                       and fields["pressure"].shape == (8,)
                       and "sO" in fields and "sG" in fields
                       and np.allclose(fields["components"].sum(axis=1), 1.0)
                       and np.all(np.isfinite(fields["pressure"])))

    if failures:
        print(f"\nPhase 132 failed ({failures} checks).")
        return 1
    print("\n3D six-component example now mirrors MRST's setup structure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
