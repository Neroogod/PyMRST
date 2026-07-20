"""Boundary-condition insertion, ported from MRST `addBC`."""

from typing import Optional, Union

import numpy as np

from pymrst.params.wells_and_bc.types import BoundaryCondition

Scalar = Union[float, np.ndarray]


def add_bc(
    bc: Optional[BoundaryCondition],
    faces: np.ndarray,
    bctype: str,
    values: Scalar,
    sat: Scalar = 1.0,
) -> BoundaryCondition:
    """Append boundary conditions of one type on boundary faces."""
    if bc is None:
        bc = BoundaryCondition()
    faces = np.asarray(faces, dtype=np.int64).ravel()
    assert faces.size > 0, "empty list of boundary faces"
    bctype = bctype.lower()
    assert bctype in ("pressure", "flux"), "type must be 'pressure' or 'flux'"
    assert not np.isin(faces, bc.face).any(), (
        "new boundary condition overlaps with existing conditions"
    )

    values_arr = _broadcast_values(values, faces.size)
    sat_arr = _broadcast_saturation(sat, faces.size)
    bc.face = np.concatenate([bc.face, faces])
    bc.type = bc.type + [bctype] * faces.size
    bc.value = np.concatenate([bc.value, values_arr])
    bc.sat = sat_arr if bc.sat.size == 0 else np.vstack([bc.sat, sat_arr])
    return bc


def _broadcast_values(values: Scalar, count: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 1:
        return np.full(count, arr[0])
    assert arr.size == count, "one value per face or one scalar value required"
    return arr


def _broadcast_saturation(sat: Scalar, count: int) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(sat, dtype=np.float64))
    if arr.shape[0] == 1:
        arr = np.repeat(arr, count, axis=0)
    return arr


if __name__ == "__main__":
    from pymrst.utils.units import BARSA

    def test_add_bc_accumulates_and_rejects_overlap() -> None:
        bc = add_bc(None, np.array([0, 1]), "pressure", 200 * BARSA)
        bc = add_bc(bc, np.array([5, 6]), "flux", 1e-3)
        assert bc.face.size == 4
        assert bc.type == ["pressure"] * 2 + ["flux"] * 2
        try:
            add_bc(bc, np.array([1]), "pressure", 0.0)
        except AssertionError:
            print("ok test_add_bc_accumulates_and_rejects_overlap")
            return
        raise AssertionError("expected overlap rejection")

    test_add_bc_accumulates_and_rejects_overlap()
    print("\nAll tests passed!")

