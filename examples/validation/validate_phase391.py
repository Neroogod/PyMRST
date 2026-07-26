"""Phase 391: NFVM harmonic points and runnable 2-D NTPFA flux chain."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.nfvm import NTPFA, findHAP
from pymrst.modules.nfvm.solvers.utils.compute_flux import computeFlux
from pymrst.modules.nfvm.solvers.utils.find_os_flux import findOSflux
from pymrst.modules.nfvm.solvers.utils.trans_ntpfa import TransNTPFA
from pymrst.params.rock import make_rock


def _twisted_grid(n=10, amplitude=0.3):
    g = cart_grid([n, n], [float(n), float(n)])
    coords = np.asarray(g.nodes.coords, dtype=float)
    x, y = coords[:, 0].copy(), coords[:, 1].copy()
    coords[:, 0] += amplitude*np.sin(np.pi*x/n)*np.sin(np.pi*y/n)
    coords[:, 1] += 0.5*amplitude*np.sin(2*np.pi*x/n)*np.sin(np.pi*y/n)
    g.nodes.coords = coords
    return compute_geometry(g)


def main():
    # On an isotropic Cartesian grid the HAP construction collapses exactly
    # to face centroids in both supported dimensions.
    hap_errors = []
    hull_fractions = []
    for dims in ([4, 3], [3, 3, 2]):
        g0 = compute_geometry(cart_grid(dims, np.asarray(dims, dtype=float)))
        h0 = findHAP(g0, make_rock(g0, 1.0, 0.2))
        hap_errors.append(float(np.max(np.abs(h0.coords - g0.faces.centroids))))
        hull_fractions.append(h0.fraction)

    g = _twisted_grid()
    K = np.array([[2.0, 0.3], [0.3, 1.0]])
    rock = make_rock(
        g, np.tile([2.0, 0.3, 1.0], (g.cells.num, 1)), 0.2
    )
    ntpfa = NTPFA.from_grid(g, rock)
    gradient = np.array([3.0, 2.0])
    pressure = np.asarray(g.cells.centroids) @ gradient + 5.0
    internal = np.all(g.faces.neighbors >= 0, axis=1)
    faces = np.flatnonzero(internal)
    flux = -ntpfa.unscaled_gradient(pressure)
    exact_flux = -(np.asarray(g.faces.normals)[faces] @ (K @ gradient))

    # Boundary HAPs encode homogeneous Neumann data, so the patch test uses
    # faces whose one-sided stencils do not touch the outer cell layer.
    centers = np.asarray(g.faces.centroids)[faces]
    core = (
        (centers[:, 0] > 1.2) & (centers[:, 0] < 8.8)
        & (centers[:, 1] > 1.2) & (centers[:, 1] < 8.8)
    )
    patch_error = float(np.max(np.abs(flux[core] - exact_flux[core])))
    constant_error = float(np.max(np.abs(ntpfa.unscaled_gradient(
        np.full(g.cells.num, 17.0)
    ))))

    # Exercise the individual MRST-layout utility entry points too.
    hap = findHAP(g, rock)
    stencils = findOSflux(g, rock, hap)
    trans = TransNTPFA(g, pressure, stencils)
    layout_flux = computeFlux(g, pressure, trans)[internal]
    layout_error = float(np.max(np.abs(layout_flux - flux)))

    print("Phase 391: NFVM HAP/NTPFA vertical slice")
    print(f"  Cartesian HAP centroid error: {max(hap_errors):.3e}")
    print(f"  Cartesian outside-hull fraction: {max(hull_fractions):.3e}")
    print(f"  twisted full-tensor linear patch: {patch_error:.3e}")
    print(f"  constant-pressure flux: {constant_error:.3e}")
    print(f"  utility/class flux parity: {layout_error:.3e}")

    failed = (
        max(hap_errors) > 1e-12
        or max(hull_fractions) != 0.0
        or patch_error > 1e-10
        or constant_error > 1e-12
        or layout_error > 1e-12
    )
    print("\nPhase 391 passed." if not failed else "\nPhase 391 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
