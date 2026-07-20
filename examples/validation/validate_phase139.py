"""Phase 139: deck-imported compositional facility group trees."""

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
    facility_group_controls_from_deck_schedule,
)
from pymrst.modules.compositional.model import OverallCompositionModel  # noqa: E402
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.params.wells_and_bc import add_well  # noqa: E402
from pymrst.utils.units import BARSA, DARCY, MILLI  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _deck_control():
    return {
        # MRST readWellKW stores GRUPTREE as child, parent.
        "GRUPTREE": [
            ["WEST", "FIELD"],
            ["EAST", "FIELD"],
        ],
        # WELSPECS stores well name, group name in the first two columns.
        "WELSPECS": [
            ["P1", "WEST"],
            ["P2", "WEST"],
            ["P3", "EAST"],
        ],
        # GCONPROD: group, control, oil, water, gas, liquid, ..., status.
        "GCONPROD": [
            ["FIELD", "ORAT", 9.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"],
        ],
        # WGRUPCON guide rates become deterministic direct-member fractions.
        "WGRUPCON": [
            ["P1", "YES", 1.0, "WEST", 1.0],
            ["P2", "YES", 3.0, "WEST", 1.0],
            ["P3", "YES", 1.0, "EAST", 1.0],
        ],
    }


def _model():
    g = compute_geometry(cart_grid([6, 1, 1], [120.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=5.0e-5, compi=np.array([0.0, 1.0]), sign=1,
                     name="INJ")
    wells[0].components = np.array([0.9, 0.05, 0.05])
    for cell, name, bhp in [(3, "P1", 55.0), (4, "P2", 54.0), (5, "P3", 53.0)]:
        wells = add_well(wells, g, rock, np.array([cell]), well_type="bhp",
                         val=bhp * BARSA, compi=np.array([0.0, 1.0]),
                         sign=-1, name=name)
        wells[-1].components = np.array([0.5, 0.3, 0.2])
    model = OverallCompositionModel(g, rock, simple_mixture(),
                                    temperature=350.0, wells=wells)
    return model, model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])


def validate_control_conversion() -> int:
    groups = facility_group_controls_from_deck_control(_deck_control())
    by_name = {g.name: g for g in groups}

    failures = 0
    failures += _check("deck converter returns leaf groups before parent",
                       [g.name for g in groups] == ["EAST", "WEST", "FIELD"])
    failures += _check("WELSPECS wells attach to deck groups",
                       by_name["WEST"].wells == ["P1", "P2"]
                       and by_name["EAST"].wells == ["P3"])
    failures += _check("GRUPTREE becomes parent subgroup references",
                       set(by_name["FIELD"].wells) == {"group:WEST", "group:EAST"})
    failures += _check("GCONPROD oil target becomes orat limit",
                       np.isclose(by_name["FIELD"].lims["orat"], 9.0e-5))
    failures += _check("WGRUPCON guide rates become child fractions",
                       by_name["WEST"].allocation == "fraction"
                       and np.allclose(by_name["WEST"].fractions, [1.0, 3.0]))
    return failures


def validate_schedule_conversion() -> int:
    schedule = {"control": [_deck_control(), {"GCONINJE": [["GINJ", "GAS", "RATE", 2.0e-5]]}]}
    converted = facility_group_controls_from_deck_schedule(schedule)

    failures = 0
    failures += _check("schedule converter returns one group list per control",
                       len(converted) == 2)
    failures += _check("GCONINJE gas target maps to grat",
                       converted[1][0].name == "GINJ"
                       and np.isclose(converted[1][0].lims["grat"], 2.0e-5))
    return failures


def validate_imported_tree_switches_model_controls() -> int:
    model, state = _model()
    state.qOs[1:] = [-6.0e-5, -4.0e-5, -5.0e-5]
    state.qGs[1:] = 0.0
    model.group_controls = facility_group_controls_from_deck_control(_deck_control())

    ok = model._update_group_controls_from_state(state)
    failures = 0
    failures += _check("imported FIELD group trips aggregate producer limit", not ok)
    failures += _check("imported parent without fractions allocates by current leaf rates",
                       np.isclose(model.wells[1].val, -3.6e-5)
                       and np.isclose(model.wells[2].val, -2.4e-5)
                       and np.isclose(model.wells[3].val, -3.0e-5))
    failures += _check("producer controls are switched to deck group limit type",
                       all(model.wells[ix].type == "orat" for ix in [1, 2, 3]))
    return failures


def main() -> int:
    print("\nPhase 139: deck-imported compositional facility group trees")
    failures = validate_control_conversion()
    failures += validate_schedule_conversion()
    failures += validate_imported_tree_switches_model_controls()
    if failures:
        print(f"\nPhase 139 failed ({failures} checks).")
        return 1
    print("\nDeck-imported compositional facility group trees map to native controls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
