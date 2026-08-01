"""Elasticity on a complex grid with holes (MRST `example_2D_complex_grid`).

A `mixed4` grid - Cartesian, PEBI and triangular blocks glued together - with
a group of cells cut out of it, so the domain has interior holes and a ragged
internal boundary. It is loaded by gravity plus an overburden that changes
sign across the mid-line, and is held only on its two vertical sides.

The point is that none of this needs special treatment: one assembly handles
polygons of any shape and a domain of any connectivity.

The removed-cell indices are MRST's, translated to zero-based. They select
cells by position in the glued grid's ordering, so they are only meaningful
for this exact construction; the module checks they are in range rather than
failing obscurely if the ordering ever changes.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.remove_cells import remove_cells
from pymrst.modules.vemmech.utils.calculate_stress_vem import calculate_stress_vem
from pymrst.modules.vemmech.utils.square_grid import square_grid
from pymrst.modules.vemmech.utils.vem_div import vem_div
from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c_2d
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast
from pymrst.utils.mcolon import mcolon
from pymrst.utils.units import BARSA

# MRST's cut-out, one-based, kept verbatim then shifted
REMOVED_CELLS_ONE_BASED = (
    list(range(140, 152)) + list(range(233, 236)) + [249, 250, 93, 117, 118]
)


@dataclass
class ComplexGridResult:
    grid: object
    u: np.ndarray
    divergence: np.ndarray
    stress: np.ndarray
    removed: int


def _side_nodes(grid, axis: int, value: float):
    """Nodes of the boundary faces whose centroid sits at `value` on `axis`."""
    centroids = np.asarray(grid.faces.centroids, dtype=float)
    faces = np.flatnonzero(np.abs(centroids[:, axis] - value)
                           < np.finfo(float).eps ** 0.5)
    if faces.size == 0:
        return faces, np.zeros(0, dtype=np.int64)
    inodes = mcolon(grid.faces.nodePos[faces], grid.faces.nodePos[faces + 1])
    return faces, np.unique(np.asarray(grid.faces.nodes, dtype=np.int64)[inodes])


def run_example_2d_complex_grid(
    *,
    L: Sequence[float] = (1.0, 1.0),
    cart_dims: Sequence[int] = (4, 4),
    disturb: float = 0.02,
    E: float = 4.0e7,
    nu: float = 0.4,
    density: float = 3000.0,
    gravity: float = 10.0,
    force: float = 50.0 * BARSA,
    offset: float = 100.0 * BARSA,
    remove: Sequence[int] = None,
) -> ComplexGridResult:
    """Solve the holed complex-grid problem and return the fields."""
    L = [float(v) for v in L]
    grid = square_grid(list(cart_dims), L, grid_type="mixed4", disturb=disturb)

    removed = (np.asarray(REMOVED_CELLS_ONE_BASED, dtype=np.int64) - 1
               if remove is None else np.asarray(remove, dtype=np.int64))
    removed = removed[removed < grid.cells.num]
    if removed.size:
        grid = remove_cells(grid, removed)
    grid = create_augmented_grid(compute_geometry(grid))

    # held on the two vertical sides; top and bottom are free
    left_faces, left = _side_nodes(grid, 0, 0.0)
    right_faces, right = _side_nodes(grid, 0, L[0])
    nodes = np.concatenate([left, right])
    mask = np.ones((nodes.size, 2), dtype=bool)

    # overburden on the top, changing sign across the mid-line
    top_faces, _ = _side_nodes(grid, 1, L[1])
    centroids = np.asarray(grid.faces.centroids, dtype=float)[top_faces]
    magnitude = force * np.sign(centroids[:, 0] - L[0] / 2.0) + offset
    traction = magnitude[:, None] * np.array([0.0, -1.0])

    el_bc = ElasticBC(disp_nodes=nodes, disp_mask=mask,
                      disp_vals=np.zeros((nodes.size, 2)),
                      force_faces=top_faces, force=traction)
    C = enu2c_2d(np.full(grid.cells.num, E), np.full(grid.cells.num, nu))
    load = lambda x: -(gravity * density) * np.tile(  # noqa: E731
        [0.0, 1.0], (np.asarray(x).shape[0], 1))

    u, _ = vem_lin_elast(grid, C, el_bc, load)
    divergence = (vem_div(grid) @ u.reshape(-1)) / grid.cells.volumes
    stress = calculate_stress_vem(grid, u.reshape(-1), C)
    if isinstance(stress, tuple):
        stress = stress[0]
    return ComplexGridResult(grid=grid, u=u, divergence=divergence,
                             stress=np.asarray(stress), removed=int(removed.size))


if __name__ == "__main__":
    result = run_example_2d_complex_grid()
    print(f"complex mixed4 grid, {result.removed} cells removed -> "
          f"{result.grid.cells.num} cells")
    sides = np.diff(result.grid.cells.facePos)
    print(f"  cell face counts    : {sides.min()}..{sides.max()} "
          f"({np.unique(sides).size} distinct)")
    print(f"  x displacement      : {result.u[:, 0].min():+.4e} .. "
          f"{result.u[:, 0].max():+.4e}")
    print(f"  y displacement      : {result.u[:, 1].min():+.4e} .. "
          f"{result.u[:, 1].max():+.4e}")
    print(f"  divergence          : {result.divergence.min():+.4e} .. "
          f"{result.divergence.max():+.4e}")
    print(f"  stress array        : {result.stress.shape}")
    assert np.all(np.isfinite(result.u))
    print("\nExample ran.")
