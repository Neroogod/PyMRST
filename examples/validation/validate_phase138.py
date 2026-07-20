"""Phase 138: hierarchical compositional facility group controls.

MRST facility controls are tree-shaped rather than only flat well lists. This
native PyMRST regression checks that compositional group limits can reference
subgroups, allocate a parent target over direct child groups, and still end at
deterministic per-well controls.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.model import (  # noqa: E402
    FacilityGroupControl,
    OverallCompositionModel,
)
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
    for cell, name, bhp in [(3, "P1", 55.0), (4, "P2", 54.0), (5, "P3", 53.0)]:
        wells = add_well(wells, g, rock, np.array([cell]), well_type="bhp",
                         val=bhp * BARSA, compi=np.array([0.0, 1.0]),
                         sign=-1, name=name)
        wells[-1].components = np.array([0.5, 0.3, 0.2])
    model = OverallCompositionModel(g, rock, simple_mixture(),
                                    temperature=350.0, wells=wells)
    return model, model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])


def _set_oil_rates(state, values):
    for well_ix, q in zip([1, 2, 3], values):
        state.qOs[well_ix] = q
        state.qGs[well_ix] = 0.0


def validate_parent_fraction_allocates_to_child_groups() -> int:
    model, state = _model()
    _set_oil_rates(state, [-6.0e-5, -4.0e-5, -5.0e-5])
    west = FacilityGroupControl("WEST", wells=["P1", "P2"], lims={}, allocation="equal")
    east = FacilityGroupControl("EAST", wells=["P3"], lims={}, allocation="equal")
    field = FacilityGroupControl(
        "FIELD",
        wells=["group:WEST", "EAST"],
        lims={"orat": 9.0e-5},
        allocation="fraction",
        fractions=[2.0 / 3.0, 1.0 / 3.0],
    )
    model.group_controls = [west, east, field]

    ok = model._update_group_controls_from_state(state)
    failures = 0
    failures += _check("parent group limit triggers through subgroup names", not ok)
    failures += _check("WEST child receives two thirds split equally",
                       np.isclose(model.wells[1].val, -3.0e-5)
                       and np.isclose(model.wells[2].val, -3.0e-5))
    failures += _check("EAST child receives one third",
                       np.isclose(model.wells[3].val, -3.0e-5))
    failures += _check("state guesses follow hierarchical allocation",
                       np.allclose(state.qOs[[1, 2, 3]], [-3.0e-5, -3.0e-5, -3.0e-5]))
    return failures


def validate_flat_leaf_fractions_still_work() -> int:
    model, state = _model()
    _set_oil_rates(state, [-5.0e-5, -5.0e-5, -5.0e-5])
    child = FacilityGroupControl("CHILD", wells=["P1", "P2"], lims={})
    parent = FacilityGroupControl(
        "FIELD",
        wells=["CHILD", "P3"],
        lims={"orat": 6.0e-5},
        allocation="fraction",
        fractions=[0.2, 0.3, 0.5],
    )
    model.group_controls = [child, parent]

    ok = model._update_group_controls_from_state(state)
    failures = 0
    failures += _check("leaf-length fractions remain supported", not ok)
    failures += _check("leaf fractions map to flattened wells",
                       np.allclose(model.wells[1].val, -1.2e-5)
                       and np.allclose(model.wells[2].val, -1.8e-5)
                       and np.allclose(model.wells[3].val, -3.0e-5))
    return failures


def validate_inactive_leaf_renormalization() -> int:
    model, state = _model()
    model.wells[2].status = False
    _set_oil_rates(state, [-5.0e-5, -5.0e-5, -5.0e-5])
    child = FacilityGroupControl("CHILD", wells=["P1", "P2"], lims={}, allocation="equal")
    parent = FacilityGroupControl(
        "FIELD",
        wells=["CHILD", "P3"],
        lims={"orat": 4.0e-5},
        allocation="fraction",
        fractions=[0.5, 0.5],
    )
    model.group_controls = [child, parent]

    ok = model._update_group_controls_from_state(state)
    failures = 0
    failures += _check("inactive subgroup leaves are ignored", not ok)
    failures += _check("active leaves receive renormalized direct-member shares",
                       np.isclose(model.wells[1].val, -2.0e-5)
                       and model.wells[2].type == "bhp"
                       and np.isclose(model.wells[3].val, -2.0e-5))
    return failures


def validate_group_cycle_detection() -> int:
    model, state = _model()
    _set_oil_rates(state, [-5.0e-5, -5.0e-5, -5.0e-5])
    a = FacilityGroupControl("A", wells=["B"], lims={"orat": 1.0e-5})
    b = FacilityGroupControl("B", wells=["A"], lims={})
    model.group_controls = [a, b]
    try:
        model._update_group_controls_from_state(state)
    except ValueError as err:
        return _check("cyclic group references fail loudly",
                      "cyclic facility group reference" in str(err))
    return _check("cyclic group references fail loudly", False)


def main() -> int:
    print("\nPhase 138: hierarchical facility group controls")
    failures = validate_parent_fraction_allocates_to_child_groups()
    failures += validate_flat_leaf_fractions_still_work()
    failures += validate_inactive_leaf_renormalization()
    failures += validate_group_cycle_detection()
    if failures:
        print(f"\nPhase 138 failed ({failures} checks).")
        return 1
    print("\nHierarchical compositional facility group controls are deterministic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
