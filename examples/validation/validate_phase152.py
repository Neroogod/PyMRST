"""Phase 152: WGRUPCON status and metadata parity."""

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


def _deck_control():
    return {
        "WELSPECS": [["P1", "PROD"], ["P2", "PROD"], ["P3", "PROD"]],
        "GCONPROD": [["PROD", "ORAT", 8.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
        "WGRUPCON": [
            ["P1", "YES", 1.0, "PROD"],
            ["P2", "NO", 99.0, "PROD"],
            ["P3", "YES", 3.0, "PROD"],
        ],
    }


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


def validate_wgrupcon_metadata_and_status() -> int:
    group = facility_group_controls_from_deck_control(_deck_control())[0]
    metadata = group.extra.get("WGRUPCON", [])

    failures = 0
    failures += _check("WGRUPCON rows remain available as group metadata",
                       len(metadata) == 3
                       and [row["well"] for row in metadata] == ["P1", "P2", "P3"])
    failures += _check("inactive WGRUPCON status is preserved",
                       [row["status"] for row in metadata] == [True, False, True])
    failures += _check("inactive WGRUPCON guide does not seed allocation fractions",
                       group.allocation == "fraction"
                       and group.wells == ["P1", "P2", "P3"]
                       and np.allclose(group.fractions, [1.0, 0.0, 3.0]))
    return failures


def validate_inactive_wgrupcon_guide_does_not_receive_group_target() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(_deck_control())
    state.qOs[1:] = [-5.0e-5, -5.0e-5, -5.0e-5]
    state.qGs[1:] = 0.0
    ok = model._update_group_controls_from_state(state)

    failures = 0
    failures += _check("WGRUPCON group limit trips producer target", not ok)
    failures += _check("inactive WGRUPCON guide receives zero allocated target",
                       np.allclose(
                           [model.wells[1].val, model.wells[2].val, model.wells[3].val],
                           [-2.0e-5, -0.0, -6.0e-5],
                       ))
    failures += _check("all producer controls switch to group limit mode",
                       [model.wells[ix].type for ix in [1, 2, 3]] == ["orat", "orat", "orat"])
    return failures


def validate_all_inactive_wgrupcon_falls_back_to_rate_allocation() -> int:
    group = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            "GCONPROD": [["PROD", "ORAT", 8.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
            "WGRUPCON": [["P1", "NO", 1.0, "PROD"], ["P2", "NO", 3.0, "PROD"]],
        }
    )[0]

    failures = 0
    failures += _check("all-inactive WGRUPCON guides fall back to rate allocation",
                       group.allocation == "rate" and group.fractions is None)
    failures += _check("all-inactive WGRUPCON metadata is still retained",
                       [row["status"] for row in group.extra["WGRUPCON"]] == [False, False])
    return failures


def main() -> int:
    print("\nPhase 152: WGRUPCON status and metadata parity")
    failures = validate_wgrupcon_metadata_and_status()
    failures += validate_inactive_wgrupcon_guide_does_not_receive_group_target()
    failures += validate_all_inactive_wgrupcon_falls_back_to_rate_allocation()
    if failures:
        print(f"\nPhase 152 failed ({failures} checks).")
        return 1
    print("\nImported WGRUPCON guide controls now preserve status and metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
