"""Phase 357: saturation CFL on a Buckley-Leverett column, against MRST (AC21).

AC21 ports `estimateSaturationCFL` and the two-phase `getFractionalFlowMagnitude`
to `ad_core/utils/cfl.py`. The MATLAB companion (`dump_phase357_reference.m`)
runs MRST's own routines on a 1D water/oil Buckley-Leverett column with a fixed
saturation profile and a fixed interior-face flux, and dumps:

- `F`   - the fractional-flow magnitude |df_w/ds_w| per cell,
- `cfl` - the per-cell CFL number,
- the ingredients `sw`, interior flux `v_int`, neighbours `N`, `pv`, `dt`.

This feeds the *identical* inputs to the PyMRST kernel and compares F and cfl
number for number, so the check isolates the CFL arithmetic from how a flux is
assembled. The neighbour list and pore volume are compared too, to confirm the
two codes discretise the 1D column the same way (so the dumped interior flux
lines up with PyMRST's own face ordering).
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_core.utils.cfl import (
    estimate_saturation_cfl,
    fractional_flow_derivative,
)
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

REF = Path(__file__).with_name("ref_p357_cfl.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase357_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    sw = ref["sw"].ravel()
    v_int = np.atleast_2d(ref["v_int"])
    N_ref = ref["N"].astype(int) - 1                 # 1-based -> 0-based
    pv_ref = ref["pv"].ravel()
    dt = float(ref["dt"].ravel()[0])
    F_ref = ref["F"].ravel()
    cfl_ref = ref["cfl"].ravel()

    # -- rebuild the same model the dump used ---------------------------------
    nx = sw.size
    g = compute_geometry(cart_grid([nx, 1, 1], [200.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 1.0]) * CENTI * POISE, rho=[1000.0, 1000.0],
        n=[2, 2], c=[0.0, 0.0], p_ref=200 * BARSA)
    model = GenericBlackOilModel(g, rock, fluid, [])

    failed = 0
    print("Discretisation matches MRST:")
    failed += _check("neighbour list N identical",
                     np.array_equal(model.operators.N, N_ref))
    failed += _check("pore volume identical",
                     np.allclose(model.operators.pv, pv_ref, rtol=1e-12))

    print("\nFractional-flow magnitude |df_w/ds_w| vs MRST:")
    F = fractional_flow_derivative(model, sw)
    err_F = np.max(np.abs(F - F_ref))
    failed += _check("F matches number for number", err_F < 1e-9,
                     f"max|dF| = {err_F:.3e}")

    print("\nSaturation CFL vs MRST (same sw, same interior flux):")
    st = State(pressure=np.full(nx, 200.0 * BARSA), sW=sw,
               qWs=np.zeros(0), qOs=np.zeros(0), bhp=np.zeros(0))
    cfl = estimate_saturation_cfl(model, st, dt, flux=v_int)
    err_cfl = np.max(np.abs(cfl - cfl_ref))
    failed += _check("CFL matches number for number", err_cfl < 1e-12,
                     f"max|dCFL| = {err_cfl:.3e}")
    failed += _check(f"CFL peak reproduced ({cfl.max():.4e})",
                     np.isclose(cfl.max(), cfl_ref.max(), rtol=1e-12))

    print("\nSaturation CFL matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
