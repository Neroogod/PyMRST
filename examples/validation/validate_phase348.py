"""Phase 348 (AC2 completion): WENO upwind reconstruction, vs MATLAB.

The one AC-phase piece that could not be validated by internal equivalence -
WENO is genuinely new physics, not a re-homing of an existing PyMRST path - so
it needs a MATLAB reference. `dump_phase348_weno.m` runs MRST's actual
`WENOUpwindDiscretizationAD` on a 1D grid and saves the face reconstruction of
two fields (a smooth ramp and a shock front) in both upwind directions;
`ref_p348_weno.mat` is that dump.

PyMRST's `WENOUpwindDiscretizationAD` must reproduce it face for face. The shock
case pins the smoothness weighting (the oscillatory stencil is down-weighted at
the front); the ramp case pins the second-order interior reconstruction and the
one-sided boundary stencils. Tolerance is machine precision - the two codes run
the identical algorithm.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.statefunctions.flux.upwind.weno import (
    WENOUpwindDiscretizationAD,
)
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.params.rock import make_rock

HERE = Path(__file__).parent
REF = HERE / "ref_p348_weno.mat"


def main():
    print("Phase 348 (AC2): WENO upwind reconstruction vs MATLAB")
    if not REF.exists():
        print("  missing (run dump_phase348_weno.m in MATLAB first)")
        print("Phase 348 skipped")
        return

    ref = scipy.io.loadmat(REF)
    nx = 20
    g = compute_geometry(cart_grid([nx, 1, 1], [float(nx), 1.0, 1.0]))
    rock = make_rock(g, 1e-13, 1.0)
    ops = setup_operators_tpfa(g, rock)
    model = type("M", (), {"G": g, "operators": ops})()

    # centroid ordering must match, or the fields would not line up
    assert np.allclose(g.cells.centroids[:, 0], ref["cell_centroids"].ravel())

    weno = WENOUpwindDiscretizationAD(model, 1)
    ramp = ref["ramp"].ravel()
    front = ref["front"].ravel()
    n_if = ops.T.size
    out = ref["out"][0, 0]
    cases = {1: ramp, 2: front}
    flags = {1: np.ones(n_if, bool), 2: np.zeros(n_if, bool)}
    labels = {1: "ramp", 2: "front"}

    worst = 0.0
    for ci in (1, 2):
        for fi in (1, 2):
            mref = np.asarray(out[f"weno_c{ci}_f{fi}"], dtype=float).ravel()
            mine = np.asarray(
                weno.face_upstream(model, None, flags[fi], cases[ci]),
                dtype=float).ravel()
            d = float(np.max(np.abs(mine - mref)))
            worst = max(worst, d)
            print(f"  {labels[ci]:6s} flag={'>' if fi == 1 else '<'}: "
                  f"max |PyMRST - MRST| = {d:.2e}")
    assert worst < 1e-12, worst
    print(f"Phase 348 passed - WENO reproduces MRST to {worst:.1e}.")


if __name__ == "__main__":
    main()
