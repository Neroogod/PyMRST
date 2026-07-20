"""Phase 126: compositional facility group allocation modes.

PyMRST's compact compositional group-control layer now supports deterministic
per-well target allocation for group limits: current-rate shares, equal shares,
and explicit fractions. This is a native regression for the richer facility-tree
control behavior that MRST schedules rely on when aggregate group constraints
are converted into per-well controls.
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


def _set_rates(st, q1: float, q2: float) -> None:
    st.qOs[1] = q1
    st.qOs[2] = q2
    st.qGs[1] = 0.0
    st.qGs[2] = 0.0


def validate_rate_allocation() -> int:
    model, state = _model()
    _set_rates(state, -4.0e-5, -6.0e-5)
    model.group_controls = [
        FacilityGroupControl("PROD", wells=["P1", "P2"], lims={"orat": 5.0e-5})
    ]
    ok = model._update_group_controls_from_state(state)
    failures = 0
    failures += _check("rate allocation triggers group switch", not ok)
    failures += _check("P1 current-rate share target",
                       np.isclose(model.wells[1].val, -2.0e-5))
    failures += _check("P2 current-rate share target",
                       np.isclose(model.wells[2].val, -3.0e-5))
    failures += _check("state guesses follow rate allocation",
                       np.isclose(state.qOs[1], -2.0e-5)
                       and np.isclose(state.qOs[2], -3.0e-5))
    return failures


def validate_equal_allocation() -> int:
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
    failures = 0
    failures += _check("equal allocation triggers group switch", not ok)
    failures += _check("P1 equal target", np.isclose(model.wells[1].val, -3.0e-5))
    failures += _check("P2 equal target", np.isclose(model.wells[2].val, -3.0e-5))
    return failures


def validate_fraction_allocation() -> int:
    model, state = _model()
    _set_rates(state, -5.0e-5, -5.0e-5)
    model.group_controls = [
        FacilityGroupControl(
            "PROD",
            wells=["P1", "P2"],
            lims={"orat": 6.0e-5},
            allocation="fraction",
            fractions=[0.25, 0.75],
        )
    ]
    ok = model._update_group_controls_from_state(state)
    failures = 0
    failures += _check("fraction allocation triggers group switch", not ok)
    failures += _check("P1 fraction target", np.isclose(model.wells[1].val, -1.5e-5))
    failures += _check("P2 fraction target", np.isclose(model.wells[2].val, -4.5e-5))
    failures += _check("fraction state guesses updated",
                       np.isclose(state.qOs[1], -1.5e-5)
                       and np.isclose(state.qOs[2], -4.5e-5))
    return failures


def validate_inactive_fraction_renormalization() -> int:
    model, state = _model()
    model.wells[1].status = False
    _set_rates(state, -5.0e-5, -5.0e-5)
    model.group_controls = [
        FacilityGroupControl(
            "PROD",
            wells=["P1", "P2"],
            lims={"orat": 4.0e-5},
            allocation="fraction",
            fractions=[0.9, 0.1],
        )
    ]
    ok = model._update_group_controls_from_state(state)
    failures = 0
    failures += _check("inactive-well fraction allocation triggers", not ok)
    failures += _check("inactive well left unchanged", model.wells[1].type == "bhp")
    failures += _check("active well receives renormalized group target",
                       np.isclose(model.wells[2].val, -4.0e-5))
    return failures


def main() -> int:
    print("\nPhase 126: facility group allocation modes")
    failures = validate_rate_allocation()
    failures += validate_equal_allocation()
    failures += validate_fraction_allocation()
    failures += validate_inactive_fraction_renormalization()
    if failures:
        print(f"\nPhase 126 failed ({failures} checks).")
        return 1
    print("\nCompositional facility group allocation modes are deterministic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
