"""Phase 302: MRST-style facility group-tree diagnostics."""

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
    grid = compute_geometry(cart_grid([6, 1, 1], [120.0, 10.0, 10.0]))
    rock = make_rock(grid, 100.0 * MILLI * DARCY, 0.25)
    wells = None
    for cell, name in [(1, "P1"), (2, "P2"), (4, "P3"), (5, "P4")]:
        wells = add_well(
            wells,
            grid,
            rock,
            np.array([cell]),
            well_type="bhp",
            val=55.0 * BARSA,
            compi=np.array([0.0, 1.0]),
            sign=-1,
            name=name,
        )
        wells[-1].components = np.array([0.5, 0.3, 0.2])
    model = OverallCompositionModel(grid, rock, simple_mixture(), temperature=350.0, wells=wells)
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "GRUPTREE": [["EAST", "FIELD"], ["WEST", "FIELD"]],
            "WELSPECS": [
                ["P1", "EAST"],
                ["P2", "EAST"],
                ["P3", "WEST"],
                ["P4", "WEST"],
            ],
            "GCONPROD": [["FIELD", "ORAT", 9.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
            "GRUPNET": [
                ["EAST", 2.0, 0, 0, "YES", "NO", "NODE_E"],
                ["WEST", 1.0, 0, 0, "YES", "NO", "NODE_W"],
            ],
        }
    )
    state = model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])
    state.qOs[:] = [-6.0e-5, -2.0e-5, -4.0e-5, -2.0e-5]
    state.qGs[:] = 0.0
    return model, state


def validate_tree_summary_before_switch() -> int:
    model, state = _model()
    tree = model.facility_group_tree_summary(state)
    field = tree["by_name"]["FIELD"]

    failures = 0
    failures += _check("tree summary exposes FIELD as root",
                       tree["roots"] == ["FIELD"])
    failures += _check("FIELD direct members are child groups",
                       field["direct_wells"] == []
                       and field["child_groups"] == ["EAST", "WEST"])
    failures += _check("FIELD resolves all leaf producers in MRST well order",
                       field["leaf_wells"] == ["P1", "P2", "P3", "P4"]
                       and field["leaf_indices"] == [0, 1, 2, 3])
    failures += _check("child groups know their parent and direct wells",
                       tree["by_name"]["EAST"]["parent"] == "FIELD"
                       and tree["by_name"]["EAST"]["direct_wells"] == ["P1", "P2"]
                       and tree["by_name"]["WEST"]["parent"] == "FIELD"
                       and tree["by_name"]["WEST"]["direct_wells"] == ["P3", "P4"])
    failures += _check("state-aware summary reports aggregate producer rates",
                       np.isclose(field["rates"]["oil"], -1.4e-4)
                       and np.isclose(field["rates"]["total"], -1.4e-4))
    failures += _check("summary reports active violated group limit",
                       field["violated_limit"] == "orat")
    failures += _check("GRUPNET parent guide allocation is previewed",
                       field["allocation_preview"]["wells"] == ["P1", "P2", "P3", "P4"]
                       and np.allclose(
                           field["allocation_preview"]["weights"],
                           [0.5, 1.0 / 6.0, 2.0 / 9.0, 1.0 / 9.0],
                       ))
    return failures


def validate_tree_summary_after_switch() -> int:
    model, state = _model()
    ok = model._update_group_controls_from_state(state)
    tree = model.facilityGroupTreeSummary(state)
    field = tree["by_name"]["FIELD"]
    expected = -9.0e-5 * np.array([0.5, 1.0 / 6.0, 2.0 / 9.0, 1.0 / 9.0])

    failures = 0
    failures += _check("FIELD group limit switches through nested tree",
                       not ok and model.last_group_switches[-1]["group"] == "FIELD")
    failures += _check("parent guide rates and child current rates drive targets",
                       np.allclose([w.val for w in model.wells], expected))
    failures += _check("runtime report keeps GRUPNET allocation provenance",
                       model.last_group_switches[-1]["allocation_source_keyword"] == "GRUPNET"
                       and model.last_group_switches[-1]["allocation_guided_groups"] == ["EAST", "WEST"])
    failures += _check("camelCase tree summary alias carries updated preview",
                       field["allocation_preview"]["wells"] == ["P1", "P2", "P3", "P4"]
                       and np.isclose(sum(field["allocation_preview"]["weights"]), 1.0))
    return failures


def main() -> int:
    print("\nPhase 302: MRST-style facility group-tree diagnostics")
    failures = validate_tree_summary_before_switch()
    failures += validate_tree_summary_after_switch()
    if failures:
        print(f"\nPhase 302 failed ({failures} checks).")
        return 1
    print("\nFacility group trees now expose MRST-style summary and allocation diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
