"""Pressure side boundary helper, ported from MRST `pside`."""

from typing import Optional, Union

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.wells_and_bc.add_bc import add_bc
from pymrst.params.wells_and_bc.types import BoundaryCondition

Scalar = Union[float, np.ndarray]


def pside(
    bc: Optional[BoundaryCondition],
    g: Grid,
    side: str,
    pressure: Scalar,
    sat: Scalar = 1.0,
    i1: Optional[Union[int, list[int]]] = None,
    i2: Optional[Union[int, list[int]]] = None,
    search_range: Optional[Union[int, list[int]]] = None,
) -> Optional[BoundaryCondition]:
    """Apply a pressure boundary condition on one logical side."""
    from pymrst.utils.gridtools import boundary_face_indices

    faces = boundary_face_indices(g, side, i1, i2, search_range)
    return add_bc(bc, faces, "pressure", pressure, sat=sat)


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.units import BARSA

    def test_pside_picks_the_right_faces() -> None:
        g = compute_geometry(cart_grid([5, 4, 3], [50.0, 40.0, 30.0]))
        bc = pside(None, g, "xmin", 200 * BARSA)
        bc = pside(bc, g, "xmax", 100 * BARSA)
        assert bc.face.size == 2 * 4 * 3
        assert set(bc.type) == {"pressure"}
        assert np.all(g.faces.neighbors[bc.face].min(axis=1) < 0)
        print("ok test_pside_picks_the_right_faces")

    def test_pside_restricts_to_an_index_range() -> None:
        g = compute_geometry(cart_grid([6, 5, 4], [60.0, 50.0, 40.0]))
        full = pside(None, g, "xmin", 1e5)
        part = pside(None, g, "xmin", 1e5, i1=[0, 1], i2=[2, 3])
        assert full.face.size == 5 * 4
        assert part.face.size == 2 * 2
        assert set(part.face.tolist()) <= set(full.face.tolist())
        print("ok test_pside_restricts_to_an_index_range")

    test_pside_picks_the_right_faces()
    test_pside_restricts_to_an_index_range()
    print("\nAll tests passed!")
