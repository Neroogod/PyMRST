"""Phase 393: MRST AvgMPFA arithmetic one-sided flux blend."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.nfvm import AvgMPFA
from pymrst.modules.nfvm.solvers.utils.avg_trans_ntpfa import AvgTransNTPFA
from pymrst.params.rock import make_rock


def _grid(n=8, amplitude=0.25):
    g = cart_grid([n, n], [float(n), float(n)])
    coords = np.asarray(g.nodes.coords, dtype=float)
    x, y = coords.T.copy()
    coords[:, 0] += amplitude*np.sin(np.pi*x/n)*np.sin(np.pi*y/n)
    coords[:, 1] += 0.5*amplitude*np.sin(2*np.pi*x/n)*np.sin(np.pi*y/n)
    g.nodes.coords = coords
    return compute_geometry(g)


def main():
    g = _grid()
    K = np.array([[2.0, 0.3], [0.3, 1.0]])
    rock = make_rock(
        g, np.tile([2.0, 0.3, 1.0], (g.cells.num, 1)), 0.2
    )
    avg = AvgMPFA.from_grid(g, rock)
    gradient = np.array([3.0, 2.0])
    pressure = np.asarray(g.cells.centroids) @ gradient + 5.0
    internal = np.all(g.faces.neighbors >= 0, axis=1)
    faces = np.flatnonzero(internal)
    flux = avg.unscaled_gradient(pressure)
    exact = np.asarray(g.faces.normals)[faces] @ (K @ gradient)
    centers = np.asarray(g.faces.centroids)[faces]
    core = (
        (centers[:, 0] > 1.2) & (centers[:, 0] < 6.8)
        & (centers[:, 1] > 1.2) & (centers[:, 1] < 6.8)
    )
    patch_error = float(np.max(np.abs(flux[core] - exact[core])))
    constant_error = float(np.max(np.abs(
        avg.unscaled_gradient(np.full(g.cells.num, 13.0))
    )))

    maps = AvgTransNTPFA(g, avg.OSflux)
    utility_flux = 0.5*(maps[0] @ pressure - maps[1] @ pressure)
    utility_error = float(np.max(np.abs(utility_flux - flux)))

    print("Phase 393: AvgMPFA")
    print(f"  twisted full-tensor linear patch: {patch_error:.3e}")
    print(f"  constant-pressure flux: {constant_error:.3e}")
    print(f"  AvgTransNTPFA/class parity: {utility_error:.3e}")

    failed = (
        patch_error > 1e-10
        or constant_error > 1e-11
        or utility_error > 1e-12
    )
    print("\nPhase 393 passed." if not failed else "\nPhase 393 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
