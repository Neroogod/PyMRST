"""Uniform Cartesian grid constructor, ported from MRST `cartGrid`."""

from typing import Optional, Sequence

import numpy as np

from pymrst.gridprocessing.grid_structure import Grid
from pymrst.gridprocessing.tensor_grid import cart_grid as _cart_grid


def cart_grid(
    celldim: Sequence[int],
    physdims: Optional[Sequence[float]] = None,
    depthz: Optional[np.ndarray] = None,
) -> Grid:
    """Construct a uniform Cartesian grid."""
    return _cart_grid(celldim, physdims=physdims, depthz=depthz)


if __name__ == "__main__":

    def test_cart_grid_counts() -> None:
        grid = cart_grid([3, 4, 5], [300.0, 200.0, 100.0])
        assert grid.cells.num == 60
        assert grid.nodes.num == 4 * 5 * 6
        assert grid.faces.num == 4 * 4 * 5 + 3 * 5 * 5 + 3 * 4 * 6
        assert grid.type == ["tensorGrid", "cartGrid"]
        grid.check()
        print("ok test_cart_grid_counts")

    test_cart_grid_counts()
    print("\nAll tests passed!")

