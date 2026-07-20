"""Phase 145: conservative enforcement of imported GECON economics."""

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
    g = compute_geometry(cart_grid([4, 1, 1], [80.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=5.0e-5, compi=np.array([0.0, 1.0]), sign=1,
                     name="INJ")
    wells[0].components = np.array([0.9, 0.05, 0.05])
    for cell, name in [(2, "P1"), (3, "P2")]:
        wells = add_well(wells, g, rock, np.array([cell]), well_type="bhp",
                         val=55.0 * BARSA, compi=np.array([0.0, 1.0]),
                         sign=-1, name=name)
        wells[-1].components = np.array([0.5, 0.3, 0.2])
    model = OverallCompositionModel(g, rock, simple_mixture(),
                                    temperature=350.0, wells=wells)
    return model, model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])


def _set_production(state, *, qo, qg=0.0, qw=None) -> None:
    state.qOs[1:] = np.asarray(qo, dtype=float)
    state.qGs[1:] = np.asarray(qg if np.ndim(qg) else [qg, qg], dtype=float)
    if qw is not None:
        assert state.qWs is not None
        state.qWs[1:] = np.asarray(qw, dtype=float)


def validate_minimum_oil_shut_in_from_deck_gecon() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            # name, min oil, min gas, max water cut, max GOR, max WGR, ...
            "GECON": [["PROD", 3.0e-5, np.inf, np.inf, np.inf, np.inf, "NONE", "NO", 0]],
        }
    )
    _set_production(state, qo=[-1.0e-5, -1.0e-5])
    ok = model._update_group_controls_from_state(state)
    group = model.group_controls[0]

    failures = 0
    failures += _check("metadata-only GECON group can trigger economics", not ok)
    failures += _check("GECON min oil violation shuts all active producers",
                       not model.wells[1].status and not model.wells[2].status)
    failures += _check("shut-in producers have zero state rate guesses",
                       np.allclose(state.qOs[1:], 0.0)
                       and np.allclose(state.qGs[1:], 0.0))
    failures += _check("violated GECON criterion is recorded",
                       group.extra["GECON"].get("last_violation") == "min_oil_rate")
    return failures


def validate_ratio_shut_in_and_safe_case() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            "GECON": [["PROD", 0.0, 0.0, np.inf, 2.0, np.inf, "NONE", "NO", 0]],
        }
    )
    _set_production(state, qo=[-1.0e-5, -1.0e-5], qg=[-4.0e-5, -1.0e-5])
    ok = model._update_group_controls_from_state(state)

    safe_model, safe_state = _model()
    safe_model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            "GECON": [["PROD", 0.0, 0.0, np.inf, 10.0, np.inf, "NONE", "NO", 0]],
        }
    )
    _set_production(safe_state, qo=[-2.0e-5, -2.0e-5], qg=[-1.0e-5, -1.0e-5])
    safe_ok = safe_model._update_group_controls_from_state(safe_state)

    failures = 0
    failures += _check("GECON max GOR violation shuts the group",
                       not ok
                       and safe_model.group_controls[0].status is True
                       and model.group_controls[0].extra["GECON"].get("last_violation")
                       == "max_gas_oil_ratio")
    failures += _check("non-violating GECON group remains active",
                       safe_ok and all(w.status for w in safe_model.wells))
    return failures


def validate_rate_limit_still_applies_without_economics() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            "GCONPROD": [["PROD", "ORAT", 2.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
            "GECON": [["PROD", 0.0, 0.0, np.inf, np.inf, np.inf, "NONE", "NO", 0]],
        }
    )
    _set_production(state, qo=[-2.0e-5, -2.0e-5])
    ok = model._update_group_controls_from_state(state)

    failures = 0
    failures += _check("ordinary group rate limit still triggers", not ok)
    failures += _check("rate limit switches wells instead of shutting them",
                       all(w.status for w in model.wells[1:])
                       and [w.type for w in model.wells[1:]] == ["orat", "orat"]
                       and np.allclose([w.val for w in model.wells[1:]], [-1.0e-5, -1.0e-5]))
    failures += _check("GECON metadata is retained after rate switch",
                       "GECON" in model.group_controls[0].extra
                       and "last_violation" not in model.group_controls[0].extra["GECON"])
    return failures


def main() -> int:
    print("\nPhase 145: imported GECON economics enforcement")
    failures = validate_minimum_oil_shut_in_from_deck_gecon()
    failures += validate_ratio_shut_in_and_safe_case()
    failures += validate_rate_limit_still_applies_without_economics()
    if failures:
        print(f"\nPhase 145 failed ({failures} checks).")
        return 1
    print("\nImported GECON rows now enforce conservative group shut-in criteria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
