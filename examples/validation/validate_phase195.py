"""Phase 195: GRUPNET allocation provenance reaches runtime diagnostics."""

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
    wells = None
    for cell, name in [(3, "P1"), (4, "P2"), (5, "P3")]:
        wells = add_well(
            wells,
            g,
            rock,
            np.array([cell]),
            well_type="bhp",
            val=55.0 * BARSA,
            compi=np.array([0.0, 1.0]),
            sign=-1,
            name=name,
        )
        wells[-1].components = np.array([0.5, 0.3, 0.2])
    model = OverallCompositionModel(g, rock, simple_mixture(), temperature=350.0, wells=wells)
    state = model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])
    state.qOs[:] = [-5.0e-5, -5.0e-5, -5.0e-5]
    state.qGs[:] = 0.0
    return model, state


def _groups():
    return facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "WEST"], ["P2", "WEST"], ["P3", "EAST"]],
            "GRUPTREE": [["WEST", "FIELD"], ["EAST", "FIELD"]],
            "GCONPROD": [["FIELD", "ORAT", 8.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
            "GRUPNET": [
                ["WEST", 1.0, 0, 0, "YES", "NO", "NODE_W"],
                ["EAST", 3.0, 0, 0, "YES", "NO", "NODE_E"],
            ],
        }
    )


def validate_grupnet_allocation_metadata_is_imported() -> int:
    by_name = {group.name: group for group in _groups()}
    field = by_name["FIELD"]
    metadata = field.extra.get("GRUPNET_ALLOCATION", {})
    members = metadata.get("members", [])

    failures = 0
    failures += _check(
        "parent group records GRUPNET as allocation source",
        metadata.get("source_keyword") == "GRUPNET"
        and metadata.get("allocation") == "fraction"
        and np.allclose(metadata.get("fractions", []), [1.0, 3.0]),
    )
    failures += _check(
        "child group guide metadata is preserved on allocation source",
        [row.get("group") for row in members if row.get("used")] == ["EAST", "WEST"]
        or [row.get("group") for row in members if row.get("used")] == ["WEST", "EAST"],
    )
    failures += _check(
        "network terminals remain attached to allocation source",
        sorted(row.get("terminal") for row in members if row.get("used")) == ["NODE_E", "NODE_W"],
    )
    return failures


def validate_grupnet_allocation_reaches_switch_report() -> int:
    model, state = _model()
    model.group_controls = _groups()
    ok = model._update_group_controls_from_state(state)
    field = {group.name: group for group in model.group_controls}["FIELD"]
    report = model.last_group_switches[-1] if model.last_group_switches else {}
    metadata = field.extra.get("GRUPNET_ALLOCATION", {})

    failures = 0
    failures += _check("group limit still triggers", not ok and report.get("group") == "FIELD")
    failures += _check(
        "violated constraint source remains GCONPROD",
        report.get("source_keyword") == "GCONPROD" and report.get("mode") == "orat",
    )
    failures += _check(
        "allocation source is reported separately as GRUPNET",
        report.get("allocation_source_keyword") == "GRUPNET"
        and report.get("allocation") == "fraction"
        and np.allclose(report.get("allocation_fractions", []), [1.0, 3.0]),
    )
    failures += _check(
        "runtime report lists guided child groups and terminals",
        sorted(report.get("allocation_guided_groups", [])) == ["EAST", "WEST"]
        and sorted(report.get("allocation_terminals", [])) == ["NODE_E", "NODE_W"],
    )
    failures += _check(
        "allocation metadata is stamped with runtime action",
        metadata.get("last_violation") == "orat"
        and metadata.get("last_action") == "redistribute"
        and metadata.get("last_affected_wells") == ["P1", "P2", "P3"],
    )
    failures += _check(
        "GRUPNET allocation still drives redistributed targets",
        np.isclose(model.wells[0].val, -1.0e-5)
        and np.isclose(model.wells[1].val, -1.0e-5)
        and np.isclose(model.wells[2].val, -6.0e-5),
    )
    return failures


def main() -> int:
    print("\nPhase 195: GRUPNET allocation diagnostics")
    failures = validate_grupnet_allocation_metadata_is_imported()
    failures += validate_grupnet_allocation_reaches_switch_report()
    if failures:
        print(f"\nPhase 195 failed ({failures} checks).")
        return 1
    print("\nGRUPNET-derived group allocation now reaches runtime diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
