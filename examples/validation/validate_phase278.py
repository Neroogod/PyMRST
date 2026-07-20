"""Phase 278: MRST-style natural-variable model public surface."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.models.contents import list_models  # noqa: E402
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel  # noqa: E402
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.utils.units import BARSA, DARCY, MILLI  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _model(nx: int = 4):
    grid = compute_geometry(cart_grid([nx, 1, 1], [40.0, 10.0, 10.0]))
    rock = make_rock(grid, 100.0 * MILLI * DARCY, 0.25)
    return NaturalVariablesCompositionalModel(grid, rock, simple_mixture(), temperature=350.0)


def main() -> int:
    model = _model()
    state = model.initial_state(np.linspace(55.0 * BARSA, 45.0 * BARSA, 4), [0.5, 0.3, 0.2])
    pure_liquid, pure_vapour, two_phase = model.getFlag(state)
    vars_, names, origin = model.getPrimaryVariables(state)
    refreshed = model.updateSecondaryProperties(state)
    flashed = model.flashPhases(state, state, None, False, 1)
    dirty = state.copy()
    dirty.x *= 2.0
    dirty.y *= 3.0
    dirty.sO += 0.05
    validated = model.validateState(dirty)
    catalog = list_models()

    print("\nPhase 278: MRST-style natural-variable model public surface")
    failures = 0
    failures += _check("phase/component names and indices follow active phase order",
                       model.getComponentNames() == tuple(model.mix.names)
                       and model.getPhaseNames() == ("oil", "gas")
                       and model.getLiquidIndex() == 0
                       and model.getVaporIndex() == 1
                       and model.getNumberOfPhases() == 2)
    failures += _check("getFlag/getTwoPhaseFlag return boolean cell masks",
                       pure_liquid.shape == state.pressure.shape
                       and pure_vapour.shape == state.pressure.shape
                       and two_phase.shape == state.pressure.shape
                       and np.array_equal(model.getTwoPhaseFlag(state), two_phase))
    failures += _check("getPrimaryVariables exposes MRST natural-variable names",
                       names[0] == "pressure"
                       and all(name.startswith("v_") for name in names[1:3])
                       and "sL" in names and "sV" in names
                       and names[-3:] == ["qOs", "qGs", "bhp"]
                       and len(vars_) == len(names) == len(origin))
    failures += _check("secondary-property refresh and flashPhases keep normalized compositions",
                       np.allclose(refreshed.z.sum(axis=1), 1.0)
                       and np.allclose(flashed.z.sum(axis=1), 1.0)
                       and np.allclose(refreshed.x.sum(axis=1), 1.0)
                       and np.allclose(refreshed.y.sum(axis=1), 1.0))
    failures += _check("validateState normalizes natural variables and saturations",
                       np.allclose(validated.x.sum(axis=1), 1.0)
                       and np.allclose(validated.y.sum(axis=1), 1.0)
                       and np.allclose(validated.sO + validated.sG, 1.0)
                       and np.allclose(validated.z.sum(axis=1), 1.0))
    failures += _check("model catalog records natural-variable core surface implementation",
                       catalog["GenericNaturalVariablesModel"]["status"] == "implemented_core_surface")

    if failures:
        print(f"\nPhase 278 failed ({failures} checks).")
        return 1
    print("\nNaturalVariablesCompositionalModel now exposes MRST-style flags, primary variables and state refresh helpers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
