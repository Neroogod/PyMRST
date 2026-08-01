"""Hanging rod through three material layers (MRST `example_2D_layers`).

Three horizontal layers with a ten-fold stiffness contrast in the middle one,
on the same irregular `mixed4` grid as `example_2d`. The rod hangs from its
left and right sides under its own weight, so the stiff layer carries the load
differently from the soft ones and the divergence field shows the jump.
"""

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.vemmech.utils.square_layers_test import (
    layer_index,
    square_layers_test,
)
from pymrst.modules.vemmech.utils.vem2d_div import vem2d_div
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast


@dataclass
class Example2DLayersResult:
    grid: object
    u: np.ndarray
    divergence: np.ndarray
    layer: np.ndarray
    case_name: str


def run_example_2d_layers(
    *,
    E=(1.0, 10.0, 1.0),
    nu=(0.3, 0.3, 0.3),
    zlayers=(0.3, 0.3, 0.3),
    L=(1.0, 1.0),
    dims=(4, 4),
    disturb: float = 0.042,
    grid_type: str = "mixed4",
    test_name: str = "hanging_rod_2D",
) -> Example2DLayersResult:
    """Solve the layered hanging-rod case."""
    zlayers = np.asarray(zlayers, dtype=float)
    zlayers = zlayers / zlayers.sum()                    # MRST normalises these

    grid, _, cases = square_layers_test(
        E=list(E), nu=list(nu), zlayers=list(zlayers), cart_dims=list(dims),
        L=list(L), grid_type=grid_type, disturb=disturb,
        make_testcases=False, test_name=test_name,
    )
    grid = compute_geometry(grid)
    case = cases[0]
    u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)

    divergence = (vem2d_div(grid) @ u.reshape(-1)) / grid.cells.volumes
    return Example2DLayersResult(
        grid=grid, u=u, divergence=divergence,
        layer=layer_index(grid, zlayers, list(L)), case_name=case.name,
    )


if __name__ == "__main__":
    result = run_example_2d_layers()
    print(f"case '{result.case_name}' on a {result.grid.cells.num}-cell grid")
    print(f"  layers            : {np.bincount(result.layer).tolist()} cells")
    print(f"  y displacement    : {result.u[:, 1].min():+.4e} .. "
          f"{result.u[:, 1].max():+.4e}")
    for index in range(result.layer.max() + 1):
        inside = result.layer == index
        print(f"  layer {index} divergence: {result.divergence[inside].mean():+.4e} "
              f"(mean over {int(inside.sum())} cells)")
    assert np.all(np.isfinite(result.u))
    print("\nExample ran.")
