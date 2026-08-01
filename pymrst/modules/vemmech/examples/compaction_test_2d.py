"""2-D compaction against its analytic solution (MRST `compactionTest2D`).

A column compacting under its own weight has a closed-form vertical
displacement, so this example is a genuine verification rather than a smoke
test: the computed field is compared against the analytic one.

The grid is deliberately awkward - horizontal faces triangulated, then every
node twisted - to show the answer does not depend on cells being rectangles.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.pebi import _sort_cell_faces as sort_cell_faces
from pymrst.gridprocessing.testgrids.twister import twister
from pymrst.gridprocessing.triangulate_faces import triangulate_faces
from pymrst.modules.vemmech.examples.utils.make_compaction_test import (
    CompactionOptions,
    make_compaction_test,
)
from pymrst.modules.vemmech.utils.square_grid import square_grid
from pymrst.modules.vemmech.utils.vem_div import vem_div
from pymrst.modules.vemmech.vem import create_augmented_grid, enu2c_2d
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast

# MRST triangulates the horizontal faces, tagged 3 and 4 one-based
_HORIZONTAL_TAGS = (2, 3)
# density * gravity / 2, the constant in the analytic profile
_GRAVITY_FACTOR = 100.0 * 300.0 / 2.0


@dataclass
class CompactionResult:
    grid: object
    u: np.ndarray
    analytic: np.ndarray
    divergence: np.ndarray
    analytic_divergence: np.ndarray
    relative_error: float


def analytic_displacement(z, *, start: float, top: float, stiffness: float,
                          face_force: float, gravity_load: bool):
    """MRST's closed-form vertical displacement of a compacting column."""
    z = np.asarray(z, dtype=float)
    weight = (_GRAVITY_FACTOR * ((top - start) ** 2 - (z - top) ** 2)
              if gravity_load else 0.0)
    return face_force * (z - start) / stiffness - weight / stiffness


def analytic_divergence(z, *, top: float, stiffness: float, face_force: float,
                        gravity_load: bool):
    """Derivative of `analytic_displacement`, which is the divergence here."""
    z = np.asarray(z, dtype=float)
    weight = -2.0 * (z - top) * _GRAVITY_FACTOR if gravity_load else 0.0
    return face_force / stiffness - weight / stiffness


def run_compaction_test_2d(
    *,
    E: float = 0.3e9,
    nu: float = 0.4,
    cart_dims: Sequence[int] = (10, 10),
    L: Sequence[float] = (15.0, 15.0),
    grid_type: str = "cartgrid",
    triangulate: bool = True,
    twist: bool = True,
    disturb: float = 0.05,
    gravity_load: bool = True,
    top_load: bool = False,
    free_top: bool = True,
) -> CompactionResult:
    """Solve the 2-D compaction problem and compare against the analytic field."""
    grid = square_grid(list(cart_dims), list(L), grid_type=grid_type)
    if triangulate:
        tags = np.asarray(grid.cells.tags, dtype=np.int64)
        faces = np.unique(np.asarray(grid.cells.faces,
                                     dtype=np.int64)[np.isin(tags, _HORIZONTAL_TAGS)])
        grid = sort_cell_faces(triangulate_faces(grid, faces))
    if twist:
        grid = twister(grid, disturb)
    grid = create_augmented_grid(compute_geometry(grid))

    C = enu2c_2d(np.full(grid.cells.num, E), np.full(grid.cells.num, nu))
    options = CompactionOptions(gravity_load=gravity_load, top_load=top_load,
                                free_top=free_top)
    el_bc, load = make_compaction_test(grid, options)
    u, _ = vem_lin_elast(grid, C, el_bc, load)

    dim = grid.griddim
    face_force = (abs(float(el_bc.force[0, dim - 1])) if el_bc.force.size
                  else 0.0)
    centroids = np.asarray(grid.faces.centroids, dtype=float)[:, dim - 1]
    start, top = centroids.max(), centroids.min()
    stiffness = float(np.asarray(C).reshape(grid.cells.num, -1)[0, 0])

    z = np.asarray(grid.nodes.coords, dtype=float)[:, dim - 1]
    analytic = analytic_displacement(z, start=start, top=top,
                                     stiffness=stiffness,
                                     face_force=face_force,
                                     gravity_load=gravity_load)
    divergence = (vem_div(grid) @ u.reshape(-1)) / grid.cells.volumes
    cell_z = np.asarray(grid.cells.centroids, dtype=float)[:, dim - 1]
    analytic_div = analytic_divergence(cell_z, top=top, stiffness=stiffness,
                                       face_force=face_force,
                                       gravity_load=gravity_load)

    scale = max(float(np.abs(analytic).max()), np.finfo(float).tiny)
    error = float(np.abs(u[:, dim - 1] - analytic).max()) / scale
    return CompactionResult(grid=grid, u=u, analytic=analytic,
                            divergence=divergence,
                            analytic_divergence=analytic_div,
                            relative_error=error)


if __name__ == "__main__":
    result = run_compaction_test_2d()
    dim = result.grid.griddim
    print(f"2D compaction on {result.grid.cells.num} cells "
          f"({result.grid.nodes.num} nodes)")
    print(f"  vertical displacement: {result.u[:, dim - 1].min():+.6e} .. "
          f"{result.u[:, dim - 1].max():+.6e}")
    print(f"  analytic             : {result.analytic.min():+.6e} .. "
          f"{result.analytic.max():+.6e}")
    print(f"  max relative error   : {result.relative_error:.3e}")
    print(f"  divergence           : {result.divergence.min():+.4e} .. "
          f"{result.divergence.max():+.4e}")
    assert np.all(np.isfinite(result.u))
    print("\nExample ran.")
