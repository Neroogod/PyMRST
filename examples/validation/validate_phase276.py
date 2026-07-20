"""Phase 276: getImpesWeightsOverallComposition utility parity."""

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
from pymrst.modules.compositional.utils import (  # noqa: E402
    getImpesWeightsOverallComposition,
    get_impes_weights_overall_composition,
    list_utils,
)
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.utils.adi import init_variables_adi  # noqa: E402
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _model(nx: int = 5):
    g = compute_geometry(cart_grid([nx, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    return OverallCompositionModel(g, rock, simple_mixture(), temperature=350.0)


def main() -> int:
    model = _model()
    nc, ncomp = model.g.cells.num, model.mix.n
    p0 = np.linspace(78.0 * BARSA, 62.0 * BARSA, nc)
    z0 = np.tile([0.5, 0.3, 0.2], (nc, 1))
    state = model.initial_state(p0 + np.linspace(0.5, -0.5, nc) * BARSA, z0)
    dt = 10.0 * DAY

    w = getImpesWeightsOverallComposition(model, state, dt)
    w_alias = get_impes_weights_overall_composition(model, state, dt)

    prim = init_variables_adi(
        state.pressure,
        *[state.z[:, i] for i in range(ncomp - 1)],
        np.zeros(0),
        np.zeros(0),
        np.zeros(0),
    )
    _eqs, _well_eqs, _ctrl, ref = model.assemble(
        state,
        prim[0],
        list(prim[1:1 + ncomp - 1]),
        prim[1 + ncomp - 1],
        prim[1 + ncomp],
        prim[2 + ncomp],
        dt,
        pressure_reduction=True,
        return_pressure_weights=True,
    )
    ref_matrix = np.column_stack(ref)
    ref_list = getImpesWeightsOverallComposition(model, state, dt, asList=True)
    catalog = list_utils()

    print("\nPhase 276: getImpesWeightsOverallComposition utility parity")
    failures = 0
    failures += _check("standalone utility returns one column per component",
                       w.shape == (nc, ncomp) and np.all(np.isfinite(w)))
    failures += _check("utility weights match model pressure-reduction weights",
                       np.allclose(w, ref_matrix))
    failures += _check("Pythonic alias matches MATLAB-style name", np.allclose(w_alias, w))
    failures += _check("asList returns MRST equation-order vectors",
                       isinstance(ref_list, list) and len(ref_list) == ncomp
                       and all(arr.shape == (nc,) for arr in ref_list)
                       and np.allclose(np.column_stack(ref_list), w))
    failures += _check("utility catalog marks IMPES weights implemented",
                       catalog["getImpesWeightsOverallComposition"]["status"] == "implemented_ad_accumulation_weights"
                       and catalog["getImpesWeightsOverallComposition"]["python_module"]
                       == "get_impes_weights_overall_composition")

    if failures:
        print(f"\nPhase 276 failed ({failures} checks).")
        return 1
    print("\ngetImpesWeightsOverallComposition now exposes PyMRST's MRST-style AD accumulation pressure weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
