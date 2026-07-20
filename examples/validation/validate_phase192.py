"""Phase 192: MATLAB-struct-like compositional schedule conversion."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    run_schedule,
    simple_schedule_from_mrst_schedule,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)
from pymrst.params.rock import pore_volume  # noqa: E402
from pymrst.params.wells_and_bc import add_well  # noqa: E402
from pymrst.utils.units import BARSA, DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _struct_schedule(case):
    grid = case["grid"]
    qbc = float(case["bc"].value[:2].sum())
    qsrc = float(pore_volume(grid, case["overall_model"].rock).sum() / (5000.0 * DAY))
    control_a = SimpleNamespace(
        bc=[
            SimpleNamespace(type="flux", side="xmin", value=qbc, sat=[1.0, 0.0],
                            components=[0.1, 0.8, 0.1]),
            SimpleNamespace(type="pressure", side="xmax", value=case["min_pressure"],
                            sat=[1.0, 0.0]),
        ],
        label="object-forcing-a",
    )
    control_b = SimpleNamespace(
        SRC=[
            SimpleNamespace(cell=0, rate=qsrc, sat=[0.0, 1.0],
                            zi=[0.05, 0.9, 0.05]),
            SimpleNamespace(cell=grid.cells.num - 1, rate=-qsrc, sat=[0.5, 0.5]),
        ],
        BC=SimpleNamespace(
            face=case["bc"].face,
            type=["flux", "flux", "pressure", "pressure"],
            value=[case["bc"].value[0], case["bc"].value[1],
                   case["min_pressure"] * 0.95, case["min_pressure"] * 0.95],
            sat=[[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.5, 0.5]],
            compi=[[0.75, 0.2, 0.05], [0.65, 0.3, 0.05],
                   [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ),
        label="object-forcing-b",
    )
    return SimpleNamespace(
        step=SimpleNamespace(val=np.asarray(case["schedule"].step.val[:2]), control=np.array([1, 2])),
        control=np.array([control_a, control_b], dtype=object),
    )


def validate_struct_like_forcing_schedule() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    native = simple_schedule_from_mrst_schedule(_struct_schedule(case), grid=case["grid"])

    failures = 0
    failures += _check("object schedule preserves timesteps and controls",
                       np.allclose(native.step.val, case["schedule"].step.val[:2])
                       and native.step.control.tolist() == [1, 2])
    failures += _check("object controls convert nested BC/SRC structs",
                       native.control[0].bc is not None
                       and np.allclose(native.control[0].bc.components[0], [0.1, 0.8, 0.1])
                       and native.control[1].src is not None
                       and np.allclose(native.control[1].src.components[0], [0.05, 0.9, 0.05]))
    failures += _check("object controls preserve non-forcing metadata",
                       native.control[0].extra == {"label": "object-forcing-a"}
                       and native.control[1].extra == {"label": "object-forcing-b"})

    result = run_schedule(case["overall_model"], case["initial_state"], native)
    failures += _check("object-converted schedule runs and restores forcing",
                       len(result.states) == 2
                       and np.isfinite(result.final_state.pressure).all()
                       and case["overall_model"].bc is case["bc"]
                       and getattr(case["overall_model"], "src", None) is None)
    return failures


def validate_struct_like_deck_group_schedule() -> int:
    case = make_boundary_conditions_case(nx=3, ny=1)
    grid = case["grid"]
    rock = case["overall_model"].rock
    wells = add_well(None, grid, rock, np.array([0]), well_type="rate",
                     val=5.0e-5, compi=np.array([0.0, 1.0]), sign=1,
                     name="INJ")
    wells[0].components = np.array([0.9, 0.05, 0.05])
    wells = add_well(wells, grid, rock, np.array([grid.cells.num - 1]),
                     well_type="bhp", val=75.0 * BARSA,
                     compi=np.array([0.0, 1.0]), sign=-1, name="P1")
    wells[1].components = np.array([0.5, 0.3, 0.2])
    schedule = SimpleNamespace(
        step=SimpleNamespace(val=[case["schedule"].step.val[0]], control=[1]),
        control=[
            SimpleNamespace(
                WELSPECS=[["P1", "PROD"]],
                GCONPROD=[["PROD", "ORAT", 1.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
                W=wells,
                tag="deck-object",
            )
        ],
    )
    native = simple_schedule_from_mrst_schedule(schedule, grid=grid, base_wells=wells)

    failures = 0
    failures += _check("object deck control imports group controls",
                       len(native.control) == 1
                       and native.control[0].group_controls
                       and native.control[0].group_controls[0].name == "PROD")
    failures += _check("object deck control preserves wells and extra metadata",
                       len(native.control[0].wells) == 2
                       and native.control[0].extra.get("tag") == "deck-object"
                       and "WELSPECS" in native.control[0].extra
                       and "GCONPROD" in native.control[0].extra)
    return failures


def main() -> int:
    print("\nPhase 192: MATLAB-struct-like compositional schedule conversion")
    failures = validate_struct_like_forcing_schedule()
    failures += validate_struct_like_deck_group_schedule()
    if failures:
        print(f"\nPhase 192 failed ({failures} checks).")
        return 1
    print("\nMATLAB-style struct schedules now convert through the MRST schedule helper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
