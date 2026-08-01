"""3-D compaction against its analytic solution (MRST `compactionTest3D`).

The 3-D counterpart of `compaction_test_2d`, on the `box` case from
`complex_3d_grid`: a Cartesian block whose top and bottom faces are
triangulated, so cells are general polyhedra. Gravity and an overburden load
act together, and the same closed-form column profile applies.

MRST's `sbed` and `norne` cases need ECLIPSE decks it downloads separately;
`complex_3d_grid` refuses those with an explanation rather than pretending.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.vemmech.examples.compaction_test_2d import (
    analytic_displacement,
    analytic_divergence,
)
from pymrst.modules.vemmech.examples.utils.make_compaction_test import (
    CompactionOptions,
    make_compaction_test,
)
from pymrst.modules.vemmech.utils.complex_3d_grid import (
    ComplexGridOptions,
    complex_3d_grid,
)
from pymrst.modules.vemmech.utils.vem_div import vem_div
from pymrst.modules.vemmech.vem import create_augmented_grid, enu2c
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast


@dataclass
class Compaction3DResult:
    grid: object
    u: np.ndarray
    analytic: np.ndarray
    divergence: np.ndarray
    relative_error: float


def run_compaction_test_3d(
    *,
    E: float = 0.3e9,
    nu: float = 0.4,
    cart_dims: Sequence[int] = (3, 3, 10),
    L: Sequence[float] = (15.0, 15.0, 3.0),
    grid_case: str = "box",
    triangulate: bool = False,
    disturb: float = 0.0,
    gravity_load: bool = True,
    top_load: bool = True,
    free_top: bool = True,
    hanging: bool = False,
) -> Compaction3DResult:
    """Solve the 3-D compaction problem and compare against the analytic field."""
    options = ComplexGridOptions(cart_dims=list(cart_dims), L=list(L),
                                 disturb=disturb, triangulate=triangulate)
    grid, _ = complex_3d_grid(options, grid_case)
    grid = create_augmented_grid(compute_geometry(grid))

    C = enu2c(np.full(grid.cells.num, E), np.full(grid.cells.num, nu))
    el_bc, load = make_compaction_test(
        grid, CompactionOptions(gravity_load=gravity_load, top_load=top_load,
                                free_top=free_top, hanging=hanging))
    u, _ = vem_lin_elast(grid, C, el_bc, load)

    dim = grid.griddim
    face_force = abs(float(el_bc.force[0, dim - 1])) if el_bc.force.size else 0.0
    centroids = np.asarray(grid.faces.centroids, dtype=float)[:, dim - 1]
    start, top = centroids.max(), centroids.min()
    stiffness = float(np.asarray(C).reshape(grid.cells.num, -1)[0, 0])

    z = np.asarray(grid.nodes.coords, dtype=float)[:, dim - 1]
    analytic = analytic_displacement(z, start=start, top=top,
                                     stiffness=stiffness,
                                     face_force=face_force,
                                     gravity_load=gravity_load)
    divergence = (vem_div(grid) @ u.reshape(-1)) / grid.cells.volumes
    scale = max(float(np.abs(analytic).max()), np.finfo(float).tiny)
    error = float(np.abs(u[:, dim - 1] - analytic).max()) / scale
    return Compaction3DResult(grid=grid, u=u, analytic=analytic,
                              divergence=divergence, relative_error=error)


if __name__ == "__main__":
    result = run_compaction_test_3d()
    print(f"3D compaction on {result.grid.cells.num} cells "
          f"({result.grid.nodes.num} nodes)")
    print(f"  vertical displacement: {result.u[:, 2].min():+.6e} .. "
          f"{result.u[:, 2].max():+.6e}")
    print(f"  analytic             : {result.analytic.min():+.6e} .. "
          f"{result.analytic.max():+.6e}")
    print(f"  max relative error   : {result.relative_error:.3e}")
    assert np.all(np.isfinite(result.u))

    triangulated = run_compaction_test_3d(triangulate=True)
    print(f"  with triangulated top/bottom faces "
          f"({triangulated.grid.cells.num} cells): "
          f"relative error {triangulated.relative_error:.3e}")
    print("\nExample ran.")
