"""Phase 199: GCONINJE late-column context reaches diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.facility import (  # noqa: E402
    facility_group_controls_from_deck_control,
)
from pymrst.modules.compositional.model import OverallCompositionModel  # noqa: E402
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.params.wells_and_bc import add_well  # noqa: E402
from pymrst.utils.units import BARSA, DARCY, MILLI  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _model():
    g = compute_geometry(cart_grid([4, 1, 1], [80.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = None
    for cell, name in [(0, "I1"), (1, "I2")]:
        wells = add_well(
            wells,
            g,
            rock,
            np.array([cell]),
            well_type="bhp",
            val=80.0 * BARSA,
            compi=np.array([0.0, 1.0]),
            sign=1,
            name=name,
        )
        wells[-1].components = np.array([0.9, 0.05, 0.05])
    model = OverallCompositionModel(g, rock, simple_mixture(), temperature=350.0, wells=wells)
    state = model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])
    state.qOs[:] = [1.0e-5, 1.0e-5]
    state.qGs[:] = [5.0e-5, 5.0e-5]
    return model, state


def validate_gconinje_surface_context() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["I1", "GINJ"], ["I2", "GINJ"]],
            "GCONINJE": [
                {
                    "group": "GINJ",
                    "phase": "GAS",
                    "control": "RATE",
                    "surface": 7.0e-5,
                    "reservoir": 9.0e-5,
                    "reinject_target": 2.0e-5,
                    "voidage_target": 3.0e-5,
                    "status": "YES",
                    "guide_rate": 1.1e-4,
                    "guide_phase": "GAS",
                    "trgp": 125.0 * BARSA,
                    "vrep": 0.75,
                    "wet_gas": 4.4e-4,
                }
            ],
        }
    )
    group = model.group_controls[0]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[-1] if model.last_group_switches else {}
    meta = group.extra["GCONINJE"][0]

    failures = 0
    failures += _check("surface GCONINJE limit still redistributes",
                       not ok and report.get("mode") == "grat")
    failures += _check("GCONINJE diagnostic carries late numeric context",
                       report.get("source_keyword") == "GCONINJE"
                       and np.isclose(report.get("surface_rate"), 7.0e-5)
                       and np.isclose(report.get("reservoir_rate"), 9.0e-5)
                       and np.isclose(report.get("reinject_target"), 2.0e-5)
                       and np.isclose(report.get("voidage_target"), 3.0e-5)
                       and np.isclose(report.get("guide_rate"), 1.1e-4)
                       and np.isclose(report.get("trgp"), 125.0 * BARSA)
                       and np.isclose(report.get("vrep"), 0.75)
                       and np.isclose(report.get("wet_gas"), 4.4e-4))
    failures += _check("GCONINJE diagnostic carries guide phase context",
                       report.get("phase") == "GAS"
                       and report.get("control") == "rate"
                       and report.get("guide_phase") == "GAS")
    failures += _check("GCONINJE metadata is stamped with runtime action",
                       meta.get("last_violation") == "grat"
                       and meta.get("last_action") == "redistribute")
    return failures


def validate_gconinje_reservoir_context() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["I1", "GINJ"], ["I2", "GINJ"]],
            "GCONINJE": [["GINJ", "GAS", "RESV", np.inf, 7.0e-5, 1.5e-5, 2.5e-5,
                          "YES", 8.0e-5, "GAS", 130.0 * BARSA, 0.6, 3.3e-4]],
        }
    )
    state.qOs[:] = [1.0e-5, 1.0e-5]
    state.qGs[:] = [4.0e-5, 4.0e-5]
    group = model.group_controls[0]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[-1] if model.last_group_switches else {}

    failures = 0
    failures += _check("reservoir GCONINJE limit still redistributes",
                       not ok and report.get("mode") == "vrat")
    failures += _check("reservoir GCONINJE diagnostic keeps late context",
                       np.isclose(report.get("reservoir_rate"), 7.0e-5)
                       and np.isclose(report.get("reinject_target"), 1.5e-5)
                       and np.isclose(report.get("voidage_target"), 2.5e-5)
                       and np.isclose(report.get("guide_rate"), 8.0e-5)
                       and report.get("guide_phase") == "GAS"
                       and np.isclose(report.get("trgp"), 130.0 * BARSA)
                       and np.isclose(report.get("vrep"), 0.6)
                       and np.isclose(report.get("wet_gas"), 3.3e-4))
    failures += _check("reservoir GCONINJE metadata row is selected",
                       group.extra["GCONINJE"][0].get("last_violation") == "vrat")
    return failures


def main() -> int:
    print("\nPhase 199: GCONINJE late-column diagnostics")
    failures = validate_gconinje_surface_context()
    failures += validate_gconinje_reservoir_context()
    if failures:
        print(f"\nPhase 199 failed ({failures} checks).")
        return 1
    print("\nGCONINJE late MRST columns now reach native group-switch diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
