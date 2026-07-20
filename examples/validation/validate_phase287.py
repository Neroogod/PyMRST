"""Phase 287: native guide-rate group allocation parity."""

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


def _model(nprod: int = 3):
    g = compute_geometry(cart_grid([nprod + 2, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=5.0e-5, compi=np.array([0.0, 1.0]), sign=1,
                     name="INJ")
    wells[0].components = np.array([0.9, 0.05, 0.05])
    for local_ix in range(nprod):
        cell = g.cells.num - nprod + local_ix
        name = f"P{local_ix + 1}"
        wells = add_well(wells, g, rock, np.array([cell]), well_type="bhp",
                         val=(55.0 - local_ix) * BARSA,
                         compi=np.array([0.0, 1.0]), sign=-1,
                         name=name)
        wells[-1].components = np.array([0.5, 0.3, 0.2])
    model = OverallCompositionModel(g, rock, simple_mixture(),
                                    temperature=350.0, wells=wells)
    return model, model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])


def _set_oil_rates(state, rates: list[float]) -> None:
    for offset, rate in enumerate(rates, start=1):
        state.qOs[offset] = rate
        state.qGs[offset] = 0.0


def validate_native_guide_payload() -> int:
    model, state = _model(2)
    _set_oil_rates(state, [-5.0e-5, -5.0e-5])
    group = FacilityGroupControl(
        "PROD",
        wells=["P1", "P2"],
        lims={"orat": 6.0e-5},
        allocation="guide",
        extra={
            "GUIDE": {
                "members": [
                    {"well": "P1", "guide_rate": 1.0},
                    {"well": "P2", "guide_rate": 3.0},
                ]
            }
        },
    )
    model.group_controls = [group]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[0] if model.last_group_switches else {}
    meta = group.extra.get("NATIVE_ALLOCATION", {})

    failures = 0
    failures += _check("native GUIDE payload switches as guide allocation",
                       not ok
                       and report.get("allocation") == "guide"
                       and np.allclose(report.get("allocation_weights"), [0.25, 0.75]))
    failures += _check("native GUIDE payload drives producer targets",
                       np.isclose(model.wells[1].val, -1.5e-5)
                       and np.isclose(model.wells[2].val, -4.5e-5))
    failures += _check("native guide allocation metadata is stamped",
                       meta.get("last_violation") == "orat"
                       and meta.get("last_action") == "redistribute"
                       and meta.get("last_affected_wells") == ["P1", "P2"])
    return failures


def validate_imported_wgrupcon_metadata_alias() -> int:
    model, state = _model(2)
    _set_oil_rates(state, [-5.0e-5, -5.0e-5])
    group = FacilityGroupControl(
        "PROD",
        wells=["P1", "P2"],
        lims={"orat": 8.0e-5},
        allocation="guided",
        extra={
            "WGRUPCON_ALLOCATION": {
                "members": [
                    {
                        "member": "P1",
                        "well": "P1",
                        "guide_rate": 2.0,
                        "effective_guide_rate": 2.0,
                        "used": True,
                    },
                    {
                        "member": "P2",
                        "well": "P2",
                        "guide_rate": 6.0,
                        "effective_guide_rate": 6.0,
                        "used": True,
                    },
                ]
            }
        },
    )
    model.group_controls = [group]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[0] if model.last_group_switches else {}

    failures = 0
    failures += _check("guided alias consumes WGRUPCON allocation metadata",
                       not ok
                       and report.get("allocation_source_keyword") == "WGRUPCON"
                       and report.get("allocation") == "guided"
                       and report.get("allocation_guided_wells") == ["P1", "P2"]
                       and np.allclose(report.get("allocation_effective_guide_rates"), [2.0, 6.0]))
    failures += _check("WGRUPCON metadata guide rates drive targets",
                       np.isclose(model.wells[1].val, -2.0e-5)
                       and np.isclose(model.wells[2].val, -6.0e-5))
    return failures


def validate_subgroup_guide_fractions() -> int:
    model, state = _model(3)
    _set_oil_rates(state, [-3.0e-5, -3.0e-5, -4.0e-5])
    west = FacilityGroupControl("WEST", wells=["P1", "P2"], lims={}, allocation="equal")
    east = FacilityGroupControl("EAST", wells=["P3"], lims={})
    field = FacilityGroupControl(
        "FIELD",
        wells=["group:WEST", "group:EAST"],
        lims={"orat": 8.0e-5},
        allocation="guide_rate",
        extra={"GUIDE": {"fractions": [1.0, 3.0]}},
    )
    model.group_controls = [west, east, field]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[0] if model.last_group_switches else {}

    failures = 0
    failures += _check("subgroup guide fractions distribute through leaf wells",
                       not ok
                       and report.get("allocation") == "guide_rate"
                       and np.allclose(report.get("allocation_weights"), [0.125, 0.125, 0.75]))
    failures += _check("subgroup guide fractions drive nested targets",
                       np.isclose(model.wells[1].val, -1.0e-5)
                       and np.isclose(model.wells[2].val, -1.0e-5)
                       and np.isclose(model.wells[3].val, -6.0e-5))
    return failures


def validate_guide_falls_back_to_current_rates() -> int:
    model, state = _model(2)
    _set_oil_rates(state, [-2.0e-5, -8.0e-5])
    group = FacilityGroupControl(
        "PROD",
        wells=["P1", "P2"],
        lims={"orat": 5.0e-5},
        allocation="guide",
    )
    model.group_controls = [group]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[0] if model.last_group_switches else {}

    failures = 0
    failures += _check("guide allocation falls back to current rates without guides",
                       not ok
                       and report.get("allocation") == "guide"
                       and np.allclose(report.get("allocation_weights"), [0.2, 0.8]))
    failures += _check("fallback guide allocation keeps rate-share targets",
                       np.isclose(model.wells[1].val, -1.0e-5)
                       and np.isclose(model.wells[2].val, -4.0e-5))
    return failures


def main() -> int:
    print("\nPhase 287: native guide-rate group allocation")
    failures = validate_native_guide_payload()
    failures += validate_imported_wgrupcon_metadata_alias()
    failures += validate_subgroup_guide_fractions()
    failures += validate_guide_falls_back_to_current_rates()
    if failures:
        print(f"\nPhase 287 failed ({failures} checks).")
        return 1
    print("\nNative guide-rate allocation now mirrors imported MRST guide semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
