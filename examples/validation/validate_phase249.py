"""Phase 249: separator allocation diagnostics in schedule reports."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.ad_core.models.facilities.separators import EOSSeparator, SeparatorGroup  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.examples._shared import run_schedule, run_steps, simple_schedule  # noqa: E402
from pymrst.modules.compositional.model import OverallCompositionModel  # noqa: E402
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.params.wells_and_bc import add_well  # noqa: E402
from pymrst.utils.units import ATM, BARSA, DARCY, DAY, MILLI  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class DummyState:
    def __init__(self, diagnostics):
        self.separator_group_diagnostics = diagnostics


class DummyStepModel:
    def step(self, state, dt, bc=None):
        return state, True, 1


def _well_model(nx: int = 8) -> OverallCompositionModel:
    grid = compute_geometry(cart_grid([nx, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(grid, 100 * MILLI * DARCY, 0.25)
    wells = add_well(
        None,
        grid,
        rock,
        np.array([0]),
        well_type="rate",
        val=5.0e-5,
        compi=np.array([0.0, 1.0]),
        sign=1,
        name="Injector",
    )
    wells[0].components = np.array([0.9, 0.05, 0.05])
    wells = add_well(
        wells,
        grid,
        rock,
        np.array([nx - 1]),
        well_type="bhp",
        val=55 * BARSA,
        compi=np.array([0.0, 1.0]),
        sign=-1,
        name="Producer",
    )
    wells[1].components = np.array([0.5, 0.3, 0.2])
    model = OverallCompositionModel(grid, rock, simple_mixture(), temperature=350.0, wells=wells)
    model.separator_group = SeparatorGroup(EOSSeparator(pressure=ATM, T=300.0))
    return model


def validate_real_separator_report() -> int:
    model = _well_model()
    state0 = model.initial_state(70 * BARSA, [0.5, 0.3, 0.2])
    result = run_schedule(model, state0, simple_schedule([30 * DAY]))
    report = result.reports[0]
    state_diag = result.states[0].separator_group_diagnostics
    report_diag = report["SeparatorAllocationDiagnostics"]

    failures = 0
    failures += _check("real separator schedule emits one report with diagnostics",
                       len(result.reports) == 1
                       and state_diag is not None
                       and report_diag is not None)
    failures += _check("report diagnostics preserve producer allocation payload",
                       report_diag["producer_indices"] == [1]
                       and report_diag["producer_names"] == ["Producer"]
                       and np.allclose(
                           report_diag["producer_component_mass_streams"],
                           state_diag["producer_component_mass_streams"],
                       )
                       and np.allclose(
                           report_diag["allocated_surface_rates"],
                           state_diag["allocated_surface_rates"],
                       ))
    failures += _check("report diagnostics are deep-copied from state diagnostics",
                       report_diag is not state_diag
                       and report_diag["allocation_shares"] is not state_diag["allocation_shares"])
    failures += _check("separator diagnostics do not imply a schedule control switch",
                       report["SeparatorChanged"] is False and report["FacilityChanged"] is False)
    return failures


def validate_run_steps_deepcopy_default() -> int:
    diagnostics = {
        "producer_indices": [0],
        "allocation_shares": np.array([1.0]),
    }
    state = DummyState(diagnostics)
    result = run_steps(DummyStepModel(), state, [1.0])
    report_diag = result.reports[0]["SeparatorAllocationDiagnostics"]
    report_diag["allocation_shares"][0] = 0.0

    failures = 0
    failures += _check("run_steps carries separator diagnostics when present",
                       report_diag["producer_indices"] == [0])
    failures += _check("run_steps report diagnostics are independent copies",
                       np.allclose(state.separator_group_diagnostics["allocation_shares"], [1.0]))
    return failures


def main() -> int:
    print("\nPhase 249: separator allocation diagnostics in schedule reports")
    failures = validate_real_separator_report()
    failures += validate_run_steps_deepcopy_default()
    if failures:
        print(f"\nPhase 249 failed ({failures} checks).")
        return 1
    print("\nSchedule reports now carry separator allocation diagnostics when states provide them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
