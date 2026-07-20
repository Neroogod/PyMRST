"""Phase 146: GRUPNET guide rates feed native group allocation."""

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
    g = compute_geometry(cart_grid([6, 1, 1], [120.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=5.0e-5, compi=np.array([0.0, 1.0]), sign=1,
                     name="INJ")
    wells[0].components = np.array([0.9, 0.05, 0.05])
    for cell, name in [(3, "P1"), (4, "P2"), (5, "P3")]:
        wells = add_well(wells, g, rock, np.array([cell]), well_type="bhp",
                         val=55.0 * BARSA, compi=np.array([0.0, 1.0]),
                         sign=-1, name=name)
        wells[-1].components = np.array([0.5, 0.3, 0.2])
    model = OverallCompositionModel(g, rock, simple_mixture(),
                                    temperature=350.0, wells=wells)
    return model, model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])


def _set_oil_rates(state, values) -> None:
    state.qOs[[1, 2, 3]] = np.asarray(values, dtype=float)
    state.qGs[[1, 2, 3]] = 0.0


def validate_grupnet_child_guides_drive_parent_allocation() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "WEST"], ["P2", "WEST"], ["P3", "EAST"]],
            "GRUPTREE": [["WEST", "FIELD"], ["EAST", "FIELD"]],
            "GCONPROD": [["FIELD", "ORAT", 8.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
            # name, guide rate, min VFP, max VFP, network flag, VFP flag, terminal.
            "GRUPNET": [
                ["WEST", 1.0, 0, 0, "YES", "NO", "NODE_W"],
                ["EAST", 3.0, 0, 0, "YES", "NO", "NODE_E"],
            ],
        }
    )
    by_name = {g.name: g for g in model.group_controls}
    _set_oil_rates(state, [-5.0e-5, -5.0e-5, -5.0e-5])
    ok = model._update_group_controls_from_state(state)

    failures = 0
    failures += _check("parent group uses GRUPNET guide allocation",
                       by_name["FIELD"].allocation == "fraction"
                       and np.allclose(by_name["FIELD"].fractions, [1.0, 3.0]))
    failures += _check("parent group limit triggers through guided children", not ok)
    failures += _check("WEST receives one quarter and splits equally",
                       np.isclose(model.wells[1].val, -1.0e-5)
                       and np.isclose(model.wells[2].val, -1.0e-5))
    failures += _check("EAST receives three quarters",
                       np.isclose(model.wells[3].val, -6.0e-5))
    failures += _check("child GRUPNET metadata remains available",
                       by_name["WEST"].extra["GRUPNET"]["terminal"] == "NODE_W"
                       and by_name["EAST"].extra["GRUPNET"]["terminal"] == "NODE_E")
    return failures


def validate_wgrupcon_takes_precedence_over_grupnet() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"], ["P3", "PROD"]],
            "GCONPROD": [["PROD", "ORAT", 6.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
            "WGRUPCON": [["P1", "YES", 1.0, "PROD"], ["P2", "YES", 2.0, "PROD"], ["P3", "YES", 3.0, "PROD"]],
            "GRUPNET": [["PROD", 99.0, 0, 0, "YES", "NO", "NODE"]],
        }
    )
    group = model.group_controls[0]
    _set_oil_rates(state, [-5.0e-5, -5.0e-5, -5.0e-5])
    ok = model._update_group_controls_from_state(state)

    failures = 0
    failures += _check("WGRUPCON fractions remain the imported allocation",
                       group.allocation == "fraction"
                       and np.allclose(group.fractions, [1.0, 2.0, 3.0]))
    failures += _check("WGRUPCON allocation still drives well targets",
                       not ok
                       and np.allclose(
                           [model.wells[1].val, model.wells[2].val, model.wells[3].val],
                           [-1.0e-5, -2.0e-5, -3.0e-5],
                       ))
    return failures


def validate_grupnet_requires_network_flag() -> int:
    groups = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "WEST"], ["P2", "EAST"]],
            "GRUPTREE": [["WEST", "FIELD"], ["EAST", "FIELD"]],
            "GCONPROD": [["FIELD", "ORAT", 6.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
            "GRUPNET": [
                ["WEST", 1.0, 0, 0, "NO", "NO", "NODE_W"],
                ["EAST", 3.0, 0, 0, "NO", "NO", "NODE_E"],
            ],
        }
    )
    field = {g.name: g for g in groups}["FIELD"]
    failures = 0
    failures += _check("GRUPNET guide rates are ignored when network flag is off",
                       field.allocation == "rate" and field.fractions is None)
    return failures


def main() -> int:
    print("\nPhase 146: imported GRUPNET guide allocation")
    failures = validate_grupnet_child_guides_drive_parent_allocation()
    failures += validate_wgrupcon_takes_precedence_over_grupnet()
    failures += validate_grupnet_requires_network_flag()
    if failures:
        print(f"\nPhase 146 failed ({failures} checks).")
        return 1
    print("\nImported GRUPNET guide rates now seed native group allocations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
