"""Phase 250: compact separator allocation summary in schedule reports."""

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


class PlainStepModel:
    def step(self, state, dt, bc=None):
        return state + float(dt), True, 1


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


def validate_real_separator_summary() -> int:
    model = _well_model()
    state0 = model.initial_state(70 * BARSA, [0.5, 0.3, 0.2])
    result = run_schedule(model, state0, simple_schedule([30 * DAY]))
    report = result.reports[0]
    diagnostics = report["SeparatorAllocationDiagnostics"]
    summary = report["SeparatorAllocationSummary"]

    failures = 0
    failures += _check("real separator report carries compact allocation summary",
                       diagnostics is not None and summary is not None)
    failures += _check("summary preserves producer order and names",
                       summary["ProducerIndices"] == diagnostics["producer_indices"]
                       and summary["ProducerNames"] == diagnostics["producer_names"])
    failures += _check("summary preserves allocation arrays",
                       np.allclose(summary["AllocationShares"], diagnostics["allocation_shares"])
                       and np.allclose(summary["AllocatedSurfaceRates"], diagnostics["allocated_surface_rates"])
                       and np.allclose(
                           summary["PositiveSurfaceRatesTotal"],
                           diagnostics["positive_surface_rates_total"],
                       ))
    failures += _check("summary total surface rates equal negative allocated sum",
                       np.allclose(
                           summary["PositiveSurfaceRatesTotal"],
                           -summary["AllocatedSurfaceRates"].sum(axis=0),
                       ))
    failures += _check("summary arrays are independent report payloads",
                       summary["AllocationShares"] is not diagnostics["allocation_shares"]
                       and summary["AllocatedSurfaceRates"] is not diagnostics["allocated_surface_rates"])
    return failures


def validate_default_summary_absent() -> int:
    result = run_steps(PlainStepModel(), 0.0, [0.25])
    report = result.reports[0]
    failures = 0
    failures += _check("plain reports keep separator diagnostics empty",
                       report["SeparatorAllocationDiagnostics"] is None
                       and report["SeparatorAllocationSummary"] is None)
    return failures


def main() -> int:
    print("\nPhase 250: compact separator allocation summary in schedule reports")
    failures = validate_real_separator_summary()
    failures += validate_default_summary_absent()
    if failures:
        print(f"\nPhase 250 failed ({failures} checks).")
        return 1
    print("\nSchedule reports now expose compact separator allocation summaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
