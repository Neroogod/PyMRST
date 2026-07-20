"""Phase 201: native group-allocation diagnostics."""

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
    g = compute_geometry(cart_grid([5, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=5.0e-5, compi=np.array([0.0, 1.0]), sign=1,
                     name="INJ")
    wells[0].components = np.array([0.9, 0.05, 0.05])
    wells = add_well(wells, g, rock, np.array([3]), well_type="bhp",
                     val=55.0 * BARSA, compi=np.array([0.0, 1.0]), sign=-1,
                     name="P1")
    wells[1].components = np.array([0.5, 0.3, 0.2])
    wells = add_well(wells, g, rock, np.array([4]), well_type="bhp",
                     val=54.0 * BARSA, compi=np.array([0.0, 1.0]), sign=-1,
                     name="P2")
    wells[2].components = np.array([0.45, 0.35, 0.2])
    model = OverallCompositionModel(g, rock, simple_mixture(),
                                    temperature=350.0, wells=wells)
    return model, model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])


def _set_rates(state, q1: float, q2: float) -> None:
    state.qOs[1] = q1
    state.qOs[2] = q2
    state.qGs[1] = 0.0
    state.qGs[2] = 0.0


def validate_rate_allocation_diagnostics() -> int:
    model, state = _model()
    _set_rates(state, -4.0e-5, -6.0e-5)
    group = FacilityGroupControl("PROD", wells=["P1", "P2"], lims={"orat": 5.0e-5})
    model.group_controls = [group]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[0] if model.last_group_switches else {}
    meta = group.extra.get("NATIVE_ALLOCATION", {})

    failures = 0
    failures += _check("native rate allocation reports its source",
                       not ok
                       and report.get("allocation_source_keyword") == "NATIVE"
                       and report.get("allocation") == "rate")
    failures += _check("native rate allocation reports active wells and weights",
                       report.get("allocation_wells") == ["P1", "P2"]
                       and np.allclose(report.get("allocation_weights"), [0.4, 0.6]))
    failures += _check("native rate allocation metadata is stamped",
                       meta.get("last_violation") == "orat"
                       and meta.get("last_action") == "redistribute"
                       and meta.get("last_affected_wells") == ["P1", "P2"])
    failures += _check("native rate allocation targets are unchanged",
                       np.isclose(model.wells[1].val, -2.0e-5)
                       and np.isclose(model.wells[2].val, -3.0e-5))
    return failures


def validate_equal_allocation_diagnostics() -> int:
    model, state = _model()
    _set_rates(state, -1.0e-5, -9.0e-5)
    model.group_controls = [
        FacilityGroupControl(
            "PROD",
            wells=["P1", "P2"],
            lims={"orat": 6.0e-5},
            allocation="equal",
        )
    ]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[0] if model.last_group_switches else {}

    failures = 0
    failures += _check("native equal allocation reports equal weights",
                       not ok
                       and report.get("allocation_source_keyword") == "NATIVE"
                       and report.get("allocation") == "equal"
                       and np.allclose(report.get("allocation_weights"), [0.5, 0.5]))
    failures += _check("native equal allocation targets are unchanged",
                       np.isclose(model.wells[1].val, -3.0e-5)
                       and np.isclose(model.wells[2].val, -3.0e-5))
    return failures


def validate_fraction_allocation_diagnostics() -> int:
    model, state = _model()
    _set_rates(state, -5.0e-5, -5.0e-5)
    group = FacilityGroupControl(
        "PROD",
        wells=["P1", "P2"],
        lims={"orat": 6.0e-5},
        allocation="fraction",
        fractions=[0.25, 0.75],
    )
    model.group_controls = [group]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[0] if model.last_group_switches else {}
    meta = group.extra.get("NATIVE_ALLOCATION", {})

    failures = 0
    failures += _check("native fraction allocation reports raw fractions",
                       not ok
                       and report.get("allocation_source_keyword") == "NATIVE"
                       and np.allclose(report.get("allocation_fractions"), [0.25, 0.75]))
    failures += _check("native fraction allocation reports normalized weights",
                       np.allclose(report.get("allocation_weights"), [0.25, 0.75])
                       and np.allclose(meta.get("last_weights"), [0.25, 0.75]))
    failures += _check("native fraction allocation targets are unchanged",
                       np.isclose(model.wells[1].val, -1.5e-5)
                       and np.isclose(model.wells[2].val, -4.5e-5))
    return failures


def main() -> int:
    print("\nPhase 201: native group-allocation diagnostics")
    failures = validate_rate_allocation_diagnostics()
    failures += validate_equal_allocation_diagnostics()
    failures += validate_fraction_allocation_diagnostics()
    if failures:
        print(f"\nPhase 201 failed ({failures} checks).")
        return 1
    print("\nNative group-allocation modes now reach runtime diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
