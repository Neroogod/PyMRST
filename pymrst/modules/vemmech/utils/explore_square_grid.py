"""Build every `square_grid` variant (MRST `exploreSquareGrid`).

MRST's version is a script that plots each grid type in its own figure. The
useful part is the enumeration, so this returns the grids and leaves plotting
to the caller - which also makes it usable as a smoke test that every declared
grid type still builds.
"""

from typing import Dict, Sequence

from pymrst.modules.vemmech.utils.square_grid import GRID_TYPES, square_grid

# MRST plots the Cartesian case in 3D and the rest in 2D
CARTESIAN_DIMS = (4, 4, 3)
CARTESIAN_SIZE = (3.0, 3.0, 2.0)
PLANAR_DIMS = (4, 4)
PLANAR_SIZE = (3.0, 3.0)


def explore_square_grid(disturb: float = 0.1,
                        grid_types: Sequence[str] = GRID_TYPES) -> Dict[str, object]:
    """Return ``{grid_type: grid}`` for each requested type."""
    grids = {}
    for name in grid_types:
        if name == "cartgrid":
            grids[name] = square_grid(list(CARTESIAN_DIMS), list(CARTESIAN_SIZE),
                                      grid_type=name, disturb=disturb)
        else:
            grids[name] = square_grid(list(PLANAR_DIMS), list(PLANAR_SIZE),
                                      grid_type=name, disturb=disturb)
    return grids


exploreSquareGrid = explore_square_grid


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing.compute_geometry import compute_geometry

    def test_every_declared_grid_type_builds() -> None:
        grids = explore_square_grid()
        assert set(grids) == set(GRID_TYPES)
        for name, grid in grids.items():
            assert grid.cells.num > 0, name
        print(f"ok test_every_declared_grid_type_builds ({len(grids)} types)")

    def test_every_grid_has_positive_volume_cells() -> None:
        for name, grid in explore_square_grid().items():
            geom = compute_geometry(grid)
            assert np.all(geom.cells.volumes > 0), name
        print("ok test_every_grid_has_positive_volume_cells")

    def test_a_subset_can_be_requested() -> None:
        grids = explore_square_grid(grid_types=["cartgrid", "triangle"])
        assert set(grids) == {"cartgrid", "triangle"}
        assert grids["cartgrid"].griddim == 3
        assert grids["triangle"].griddim == 2
        print("ok test_a_subset_can_be_requested")

    test_every_declared_grid_type_builds()
    test_every_grid_has_positive_volume_cells()
    test_a_subset_can_be_requested()
    print("\nAll tests passed!")
