"""Empty grid shell, ported from MRST private `emptyGrid`."""

from typing import Optional

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes


def empty_grid(
    num_cells: int = 0,
    coords: Optional[np.ndarray] = None,
    griddim: int = 2,
) -> Grid:
    """Return a grid shell with faceless cells and the given nodes."""
    if coords is None:
        coords = np.zeros((0, 3))
    coords = np.asarray(coords, dtype=np.float64)
    return Grid(
        cells=Cells(
            num=num_cells,
            facePos=np.zeros(num_cells + 1, dtype=np.int64),
            faces=np.zeros(0, dtype=np.int64),
            indexMap=np.arange(num_cells, dtype=np.int64),
        ),
        faces=Faces(
            num=0,
            nodePos=np.zeros(1, dtype=np.int64),
            nodes=np.zeros(0, dtype=np.int64),
            neighbors=np.zeros((0, 2), dtype=np.int64),
        ),
        nodes=Nodes(num=coords.shape[0], coords=coords.copy()),
        griddim=griddim,
        type=["emptyGrid"],
    )


if __name__ == "__main__":

    def test_empty_grid() -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        grid = empty_grid(1, coords)
        assert grid.cells.num == 1
        assert grid.faces.num == 0
        assert grid.nodes.num == 2
        assert np.array_equal(grid.cells.facePos, [0, 0])
        print("ok test_empty_grid")

    test_empty_grid()
    print("\nAll tests passed!")

