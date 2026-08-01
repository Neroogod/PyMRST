"""Linear elasticity on an irregular 2-D grid (MRST `example_2D`).

Solves the `original_2D_test` case on a `mixed4` grid - Cartesian, PEBI and
triangular blocks glued side by side and then distorted. The point is that VEM
handles all of them with one assembly, which a scheme requiring a single cell
shape could not.

Reports the displacement field and its divergence, which is the quantity the
MRST script plots.
"""

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.vemmech.utils.square_test import square_test
from pymrst.modules.vemmech.utils.vem2d_div import vem2d_div
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast


@dataclass
class Example2DResult:
    grid: object
    u: np.ndarray
    divergence: np.ndarray
    case_name: str


def run_example_2d(
    *,
    E: float = 1.0,
    nu: float = 0.3,
    L=(1.0, 1.0),
    dims=(16, 16),
    disturb: float = 0.03,
    grid_type: str = "mixed4",
    test_name: str = "original_2D_test",
) -> Example2DResult:
    """Solve the named 2-D case and return displacement and divergence."""
    grid, _, cases = square_test(
        E=E, nu=nu, cart_dims=list(dims), L=list(L), grid_type=grid_type,
        disturb=disturb, make_testcases=False, test_name=test_name,
    )
    grid = compute_geometry(grid)
    case = cases[0]
    u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)

    divergence = (vem2d_div(grid) @ u.reshape(-1)) / grid.cells.volumes
    return Example2DResult(grid=grid, u=u, divergence=divergence,
                           case_name=case.name)


if __name__ == "__main__":
    result = run_example_2d()
    print(f"case '{result.case_name}' on a {result.grid.cells.num}-cell grid")
    print(f"  x displacement: {result.u[:, 0].min():+.4e} .. {result.u[:, 0].max():+.4e}")
    print(f"  y displacement: {result.u[:, 1].min():+.4e} .. {result.u[:, 1].max():+.4e}")
    print(f"  divergence    : {result.divergence.min():+.4e} .. "
          f"{result.divergence.max():+.4e}")
    assert np.all(np.isfinite(result.u))
    print("\nExample ran.")
