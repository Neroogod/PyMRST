"""Phase 392: NFVM HAP correction and 3-D NTPFA one-sided stencils."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.nfvm import NTPFA, correct_hap, findHAP
from pymrst.modules.nfvm.solvers.utils.correct_hap import correctHAP
from pymrst.params.rock import make_rock


def _distorted_3d(n=6):
    g = cart_grid([n, n, n], [float(n)]*3)
    coords = np.asarray(g.nodes.coords, dtype=float)
    x, y, z = coords.T.copy()
    envelope = np.sin(np.pi*x/n)*np.sin(np.pi*y/n)*np.sin(np.pi*z/n)
    coords[:, 0] += 0.12*envelope
    coords[:, 1] += 0.08*np.sin(2*np.pi*x/n)*np.sin(np.pi*y/n)*np.sin(np.pi*z/n)
    g.nodes.coords = coords
    return compute_geometry(g)


def main():
    # Explicit correction clips every displaced HAP to the same MRST face
    # radius convention without changing interpolation weights.
    g2 = compute_geometry(cart_grid([5, 4], [5.0, 4.0]))
    rock2 = make_rock(
        g2, np.tile([4.0, 1.2, 1.0], (g2.cells.num, 1)), 0.2
    )
    hap = findHAP(g2, rock2)
    weights_before = hap.weights.copy()
    ratio = 0.25
    corrected = correctHAP(g2, hap, ratio=ratio)
    radius = 0.5*np.asarray(g2.faces.areas)
    displacement = np.linalg.norm(
        corrected.coords - g2.faces.centroids, axis=1
    )
    radius_violation = float(np.max(displacement - ratio*radius))
    weight_error = float(np.max(np.abs(corrected.weights - weights_before)))

    g = _distorted_3d()
    K = np.array([
        [2.0, 0.2, 0.1],
        [0.2, 1.0, 0.15],
        [0.1, 0.15, 0.8],
    ])
    rock = make_rock(
        g,
        np.tile([2.0, 0.2, 0.1, 1.0, 0.15, 0.8], (g.cells.num, 1)),
        0.2,
    )
    ntpfa = NTPFA.from_grid(g, rock)
    gradient = np.array([3.0, 2.0, 1.0])
    pressure = np.asarray(g.cells.centroids) @ gradient + 5.0
    internal = np.all(g.faces.neighbors >= 0, axis=1)
    faces = np.flatnonzero(internal)
    flux = -ntpfa.unscaled_gradient(pressure)
    exact = -(np.asarray(g.faces.normals)[faces] @ (K @ gradient))
    centers = np.asarray(g.faces.centroids)[faces]
    core = np.all((centers > 1.2) & (centers < 4.8), axis=1)
    patch_error = float(np.max(np.abs(flux[core] - exact[core])))
    constant_error = float(np.max(np.abs(
        ntpfa.unscaled_gradient(np.full(g.cells.num, 11.0))
    )))

    print("Phase 392: NFVM correction and 3-D NTPFA")
    print(f"  HAP correction radius violation: {radius_violation:.3e}")
    print(f"  HAP interpolation-weight change: {weight_error:.3e}")
    print(f"  distorted 3-D full-tensor patch: {patch_error:.3e}")
    print(f"  3-D constant-pressure flux: {constant_error:.3e}")

    failed = (
        radius_violation > 1e-12
        or weight_error > 0.0
        or patch_error > 1e-10
        or constant_error > 1e-11
    )
    print("\nPhase 392 passed." if not failed else "\nPhase 392 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
