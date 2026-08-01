"""Phase 364: flux<->velocity conversions against MRST (Phase 25 core leftovers).

Pins the Phase 25 utility ports - `face_flux_to_cell_velocity`,
`face_flux_to_cell_flux`, `cell_flux_to_face_flux` - against MRST's own
`faceFlux2cellVelocity` / `faceFlux2cellFlux` / `cellFlux2faceFlux`
(`dump_phase364_reference.m`) on a 4x3x2 grid with a deterministic face-flux
field. PyMRST feeds the identical grid and flux and must reproduce the cell
velocities, half-face fluxes and averaged face fluxes number for number.

Cell and face centroids are compared first, to confirm both codes order the grid
the same way (so the flux arrays line up).
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.utils.cell_flux_to_face_flux import cell_flux_to_face_flux
from pymrst.utils.face_flux_to_cell_flux import face_flux_to_cell_flux
from pymrst.utils.face_flux_to_cell_velocity import face_flux_to_cell_velocity

REF = Path(__file__).with_name("ref_p364_flux.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase364_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    face_flux = ref["faceFlux"].ravel()

    g = compute_geometry(cart_grid([4, 3, 2], [4.0, 3.0, 2.0]))

    failed = 0
    print("Grid ordering (must match before comparing fluxes):")
    failed += _check("cell centroids identical",
                     np.allclose(g.cells.centroids, ref["cell_centroids"], atol=1e-12))
    failed += _check("face centroids identical",
                     np.allclose(g.faces.centroids, ref["face_centroids"], atol=1e-12))

    print("\nflux<->velocity conversions vs MRST:")
    v = face_flux_to_cell_velocity(g, face_flux)
    err_v = np.max(np.abs(v - ref["v"]))
    failed += _check("faceFlux2cellVelocity matches", err_v < 1e-12,
                     f"max|dv| = {err_v:.3e}")

    cf = face_flux_to_cell_flux(g, face_flux)
    err_cf = np.max(np.abs(cf - ref["cf"].ravel()))
    failed += _check("faceFlux2cellFlux matches", err_cf < 1e-12,
                     f"max|dcf| = {err_cf:.3e}")

    ff = cell_flux_to_face_flux(g, cf)
    err_ff = np.max(np.abs(ff - ref["ff"].ravel()))
    failed += _check("cellFlux2faceFlux matches", err_ff < 1e-12,
                     f"max|dff| = {err_ff:.3e}")

    print("\nFlux/velocity conversions match MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
