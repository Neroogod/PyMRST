"""Flux side boundary helper, ported from MRST `fluxside`."""

from typing import Optional, Union

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.wells_and_bc.add_bc import add_bc
from pymrst.params.wells_and_bc.types import BoundaryCondition

Scalar = Union[float, np.ndarray]


def fluxside(
    bc: Optional[BoundaryCondition],
    g: Grid,
    side: str,
    flux: Scalar,
    sat: Scalar = 1.0,
    i1: Optional[Union[int, list[int]]] = None,
    i2: Optional[Union[int, list[int]]] = None,
    search_range: Optional[Union[int, list[int]]] = None,
) -> Optional[BoundaryCondition]:
    """Apply a total flux across one logical side of the grid."""
    from pymrst.utils.gridtools import boundary_face_indices

    faces = boundary_face_indices(g, side, i1, i2, search_range)
    total = np.asarray(flux, dtype=np.float64).ravel()
    if faces.size == 0:
        return add_bc(bc, faces, "flux", total, sat=sat)
    if total.size not in (1, faces.size):
        raise ValueError("flux must be scalar or contain one value per selected face")
    if g.faces.areas is None:
        raise ValueError("run compute_geometry before applying a flux side")
    areas = g.faces.areas[faces]
    values = total*areas/areas.sum()
    return add_bc(bc, faces, "flux", values, sat=sat)


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_fluxside_splits_the_rate_by_area() -> None:
        for dims in ([4, 3, 2], [8, 6, 4]):
            g = compute_geometry(cart_grid(dims, [40.0, 30.0, 20.0]))
            bc = fluxside(None, g, "xmin", 2.0)
            assert np.isclose(bc.value.sum(), 2.0)
            assert np.allclose(bc.value, bc.value[0])
        print("ok test_fluxside_splits_the_rate_by_area")

    test_fluxside_splits_the_rate_by_area()
    print("\nAll tests passed!")
