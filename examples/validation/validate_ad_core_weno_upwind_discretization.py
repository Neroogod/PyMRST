"""Validate MRST WENOUpwindDiscretization and getBoundaryFacesMerged, the ports
in `ad_core/ad_oo/statefunctions/flux/`.

The reference is MRST 2024a (`dump_weno_upwind_discretization_reference.m`):
the face values of a smooth field and of a step, with their Jacobians, from
`faceUpstream` with a fixed upwind direction per face, on a 1D grid of 20
cells, a 2D grid of 7x5 cells of 2 m x 0.5 m (so the SVD frame of each cell
is anisotropic and the ghost points are not at unit distance) and a 3D grid
of 4x3x2. The step makes the cap act (face values of 0 fall back to the cell
value). The boundary faces, their cells and the reflected centroids
identical; the face values and Jacobians to `RTOL`. The triangulations come
from different Qhull builds and the per-cell SVD frames can differ by sign or,
for equal singular values, by a rotation; the reconstruction does not depend
on either, so they show only as round-off.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.get_boundary_faces_merged import (
    get_boundary_faces_merged,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.weno_upwind_discretization import (
    WENOUpwindDiscretization,
)
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params.rock import make_rock
from pymrst.utils.adi import init_variables_adi

HERE = Path(__file__).resolve().parent
RTOL = 1e-12
GRIDS = (([20], [20.0]), ([7, 5], [14.0, 2.5]), ([4, 3, 2], [4.0, 3.0, 2.0]))


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float).reshape(np.shape(a))
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def main() -> int:
    ref = loadmat(HERE/"ref_weno_upwind_discretization.mat", squeeze_me=True)
    ok = True
    for k, (dims, size) in enumerate(GRIDS, start=1):
        G = compute_geometry(cart_grid(dims, size))
        fluid = init_simple_adi_fluid(phases="WO", n=[2, 2], rho=[1000, 800], mu=[1, 5])
        model = GenericBlackOilModel(G, make_rock(G, 1, 1), fluid, gas=False)
        with contextlib.redirect_stdout(io.StringIO()):
            weno = WENOUpwindDiscretization(model)
        tag = f"{len(dims)}D {'x'.join(map(str, dims))}:"
        bfaces, bcells, bfc = get_boundary_faces_merged(G)
        ok &= report(np.array_equal(bfaces + 1, np.atleast_1d(ref[f"bfaces_g{k}"]))
                     and np.array_equal(bcells + 1, np.atleast_1d(ref[f"bcells_g{k}"]))
                     and _rel(bfc, ref[f"bfc_g{k}"]) == 0,
                     f"{tag} boundary faces, cells and reflected centroids identical")
        x = G.cells.centroids
        smooth = 0.5 + 0.3*np.sin(x[:, 0]/3)*np.cos(x[:, 1:].sum(axis=1)/2)
        step = (x[:, 0] > x[:, 0].mean()).astype(float)
        nf = model.operators.N.shape[0]
        flag = (np.arange(1, nf + 1)*7) % 5 < 2
        assert np.array_equal(flag, np.asarray(ref[f"flag_g{k}"], dtype=bool))
        for f, (name, field) in enumerate((("smooth", smooth), ("step", step)), start=1):
            (q,) = init_variables_adi(field)
            v = weno.face_upstream(model, None, flag, q)
            err_v = _rel(v.val, ref[f"val_g{k}_f{f}"])
            err_j = _rel(v.jac[0].toarray(), ref[f"jac_g{k}_f{f}"])
            ok &= report(err_v < RTOL and err_j < RTOL,
                         f"{tag} {name}: face values rel. diff {err_v:.1e}, "
                         f"Jacobian {err_j:.1e}")
    print("\nWENOUpwindDiscretization matches MRST" if ok
          else "\nWENOUpwindDiscretization does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
