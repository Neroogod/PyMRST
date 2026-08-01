"""Phase 320: base ReservoirModel MRST naming helper parity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.reservoir_model import ReservoirModel  # noqa: E402
from pymrst.modules.ad_core.statefunctions.pvt_property_functions import PhaseFluid  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _fluid() -> PhaseFluid:
    return PhaseFluid(
        phases=["water", "oil", "gas"],
        b=lambda p, rs=None, sat=None, rv=None, sat_g=None: [1.0, 1.0, 1.0],
        mu=lambda p, rs=None, sat=None, rv=None, sat_g=None: [1.0, 2.0, 3.0],
        kr=lambda s, **kw: [np.ones(1), np.ones(1), np.ones(1)],
        rho_s=np.array([1000.0, 800.0, 1.0]),
    )


def validate_reservoir_model_helpers() -> int:
    grid = SimpleNamespace(cells=SimpleNamespace(num=1))
    rock = SimpleNamespace()
    # `grad` is required by PressureGradient, which ReservoirModel now builds.
    # This mock predates that and supplied no operators at all; the checks
    # below only exercise name aliases, so the operator itself is never called.
    operators = SimpleNamespace(grad=None)
    model = ReservoirModel(
        grid,
        rock,
        _fluid(),
        operators,
        pv_ref=np.ones(1),
        gdz=np.zeros(1),
    )

    failures = 0
    failures += _check("getComponentNames aliases component_names",
                       model.getComponentNames() == model.component_names()
                       and model.get_number_of_components() == model.n_components
                       and model.getNumberOfComponents() == model.n_components)
    failures += _check("getPhaseNames returns MRST short phase names",
                       model.getPhaseNames() == ["W", "O", "G"]
                       and model.get_phase_names() == ["W", "O", "G"])
    failures += _check("phase-count helpers mirror n_phases",
                       model.getNumberOfPhases() == 3
                       and model.get_number_of_phases() == 3)
    failures += _check("getActivePhases returns all active phases",
                       np.array_equal(model.getActivePhases(), np.array([True, True, True])))
    failures += _check("getPhaseIndex accepts short and long names using zero-based indexing",
                       model.getPhaseIndex("W") == 0
                       and model.getPhaseIndex("oil") == 1
                       and model.get_phase_index("G") == 2)
    try:
        model.getPhaseIndex("polymer")
    except ValueError:
        unknown_fails = True
    else:
        unknown_fails = False
    failures += _check("getPhaseIndex rejects unknown phase names", unknown_fails)
    return failures


def main() -> int:
    print("\nPhase 320: base ReservoirModel MRST naming helper parity")
    failures = validate_reservoir_model_helpers()
    if failures:
        print(f"\nPhase 320 failed ({failures} checks).")
        return 1
    print("\nReservoirModel naming helpers are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
