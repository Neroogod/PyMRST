"""Phase 290: compositional states carry MRST-style wellSol.cdp."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.model import OverallCompositionModel  # noqa: E402
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.params.wells_and_bc import add_well  # noqa: E402
from pymrst.utils.units import BARSA, DARCY, MILLI  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _model():
    gravity = np.array([0.0, 0.0, 9.80665])
    g = compute_geometry(cart_grid([1, 1, 3], [10.0, 10.0, 30.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = add_well(
        None,
        g,
        rock,
        np.array([0, 1, 2]),
        well_type="bhp",
        val=65.0 * BARSA,
        compi=np.array([1.0, 0.0]),
        sign=-1,
        name="P1",
        gravity=gravity,
        ref_depth=0.0,
    )
    model = OverallCompositionModel(
        g,
        rock,
        simple_mixture(),
        temperature=350.0,
        wells=wells,
        gravity=float(np.linalg.norm(gravity)),
    )
    return model


def validate_wellsol_cdp_storage() -> int:
    model = _model()
    state = model.initial_state(75.0 * BARSA, [0.5, 0.3, 0.2])
    state.qOs[0] = -1.0e-5
    state.qGs[0] = 0.0
    state.bhp[0] = 65.0 * BARSA
    model._store_sequential_total_flux(state)
    ws = state.wellSol[0] if state.wellSol else None

    failures = 0
    failures += _check("sequential compositional state exposes wellSol",
                       ws is not None and ws.name == "P1")
    failures += _check("wellSol carries pressure, flux and components",
                       ws is not None
                       and np.isclose(ws.pressure, state.bhp[0])
                       and np.allclose(ws.flux, state.well_flux)
                       and np.allclose(ws.components, state.well_components))
    failures += _check("wellSol.cdp is finite and per perforation",
                       ws is not None
                       and ws.cdp.shape == (3,)
                       and np.all(np.isfinite(ws.cdp)))
    failures += _check("wellSol.cdp stores a nonzero accumulated column",
                       ws is not None
                       and ws.cdp[0] > 0.0
                       and ws.cdp[0] < ws.cdp[1] < ws.cdp[2])
    return failures


def validate_wellsol_copy_is_deep() -> int:
    model = _model()
    state = model.initial_state(75.0 * BARSA, [0.5, 0.3, 0.2])
    model._store_sequential_total_flux(state)
    copied = state.copy()
    copied.wellSol[0].cdp[0] = -1.0
    copied.wellSol[0].flux[0, 0] = 123.0

    failures = 0
    failures += _check("CompositionalState.copy deep-copies wellSol.cdp",
                       not np.isclose(copied.wellSol[0].cdp[0], state.wellSol[0].cdp[0]))
    failures += _check("CompositionalState.copy deep-copies wellSol.flux",
                       not np.isclose(copied.wellSol[0].flux[0, 0], state.wellSol[0].flux[0, 0]))
    return failures


def main() -> int:
    print("\nPhase 290: compositional wellSol.cdp")
    failures = validate_wellsol_cdp_storage()
    failures += validate_wellsol_copy_is_deep()
    if failures:
        print(f"\nPhase 290 failed ({failures} checks).")
        return 1
    print("\nCompositional sequential states now expose MRST-style wellSol.cdp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
