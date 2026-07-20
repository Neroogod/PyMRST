"""Phase 154: WGRUPCON fifth-column guide multiplier parity."""

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


def validate_positional_wgrupcon_multiplier() -> int:
    group = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"], ["P3", "PROD"]],
            "GCONPROD": [["PROD", "ORAT", 1.2e-4, np.inf, np.inf, np.inf, "NONE", "YES"]],
            # MRST readWGrupCon: well, status, guide, group, numeric multiplier.
            "WGRUPCON": [
                ["P1", "YES", 1.0, "PROD", 2.0],
                ["P2", "YES", 3.0, "PROD", 0.5],
                ["P3", "YES", 4.0, "PROD"],
            ],
        }
    )[0]
    metadata = group.extra["WGRUPCON"]

    failures = 0
    failures += _check("positional WGRUPCON multiplier changes imported fractions",
                       group.allocation == "fraction"
                       and np.allclose(group.fractions, [2.0, 1.5, 4.0]))
    failures += _check("WGRUPCON multiplier metadata is preserved",
                       np.allclose([row["guide_multiplier"] for row in metadata], [2.0, 0.5, 1.0])
                       and np.allclose([row["effective_guide_rate"] for row in metadata], [2.0, 1.5, 4.0]))
    return failures


def validate_mapping_wgrupcon_multiplier_aliases() -> int:
    group = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            "WGRUPCON": [
                {"well": "P1", "status": "YES", "guide_rate": 2.0, "group": "PROD", "guide_multiplier": 4.0},
                {"well": "P2", "status": "YES", "guide": 3.0, "grp": "PROD", "factor": 2.0},
            ],
        }
    )[0]

    failures = 0
    failures += _check("mapping WGRUPCON multiplier aliases are accepted",
                       group.allocation == "fraction"
                       and np.allclose(group.fractions, [8.0, 6.0]))
    failures += _check("mapping WGRUPCON raw records remain available",
                       group.extra["WGRUPCON"][0]["raw"][0] == "P1"
                       and group.extra["WGRUPCON"][1]["raw"][0] == "P2")
    return failures


def validate_multiplier_drives_group_target_split() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"], ["P3", "PROD"]],
            "GCONPROD": [["PROD", "ORAT", 7.5e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
            "WGRUPCON": [
                ["P1", "YES", 1.0, "PROD", 2.0],
                ["P2", "YES", 3.0, "PROD", 0.5],
                ["P3", "YES", 4.0, "PROD", 1.0],
            ],
        }
    )
    state.qOs[1:] = [-5.0e-5, -5.0e-5, -5.0e-5]
    state.qGs[1:] = 0.0
    ok = model._update_group_controls_from_state(state)

    failures = 0
    failures += _check("multiplied WGRUPCON group limit trips", not ok)
    failures += _check("multiplied guide fractions drive producer targets",
                       np.allclose(
                           [model.wells[1].val, model.wells[2].val, model.wells[3].val],
                           [-2.0e-5, -1.5e-5, -4.0e-5],
                       ))
    return failures


def main() -> int:
    print("\nPhase 154: WGRUPCON fifth-column guide multiplier parity")
    failures = validate_positional_wgrupcon_multiplier()
    failures += validate_mapping_wgrupcon_multiplier_aliases()
    failures += validate_multiplier_drives_group_target_split()
    if failures:
        print(f"\nPhase 154 failed ({failures} checks).")
        return 1
    print("\nWGRUPCON fifth-column guide multipliers now affect native allocation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
