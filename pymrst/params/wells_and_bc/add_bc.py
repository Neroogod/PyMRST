"""Boundary-condition insertion, ported from MRST `addBC`."""

from typing import Optional, Sequence, Union
import warnings

import numpy as np

from pymrst.params.wells_and_bc.types import BoundaryCondition

Scalar = Union[float, np.ndarray]


def add_bc(
    bc: Optional[BoundaryCondition],
    faces: np.ndarray,
    bctype: Union[str, Sequence[str]],
    values: Scalar,
    sat: Scalar = 1.0,
) -> Optional[BoundaryCondition]:
    """Append boundary conditions of one type on boundary faces."""
    faces = np.asarray(faces, dtype=np.int64).ravel()
    if faces.size == 0:
        warnings.warn("Empty list of boundary faces", RuntimeWarning, stacklevel=2)
        return bc
    if bc is None:
        bc = BoundaryCondition()
    types = _types(bctype, faces.size)

    values_arr = _broadcast_values(values, faces.size)
    sat_arr = _broadcast_saturation(sat, faces.size)
    if bc.sat.size and bc.sat.shape[1] != sat_arr.shape[1]:
        raise ValueError(
            "number of saturation columns does not match existing boundary conditions"
        )
    if np.isin(faces, bc.face).any():
        raise ValueError("new boundary condition overlaps with existing conditions")
    bc.face = np.concatenate([bc.face, faces])
    bc.type = bc.type + types
    bc.value = np.concatenate([bc.value, values_arr])
    bc.sat = sat_arr if bc.sat.size == 0 else np.vstack([bc.sat, sat_arr])
    return bc


def _types(value, count):
    if isinstance(value, str):
        values = [value]*count
    else:
        values = list(value)
        if len(values) != count or not all(isinstance(item, str) for item in values):
            raise ValueError("type must be a string or one string per face")
    values = [item.lower() for item in values]
    if any(item not in ("pressure", "flux") for item in values):
        raise ValueError("type must be 'pressure' or 'flux'")
    return values


def _broadcast_values(values: Scalar, count: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 1:
        return np.full(count, arr[0])
    if arr.size != count:
        raise ValueError("one value per face or one scalar value required")
    return arr


def _broadcast_saturation(sat: Scalar, count: int) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(sat, dtype=np.float64))
    if arr.shape[0] == 1:
        arr = np.repeat(arr, count, axis=0)
    if arr.shape[0] not in (0, count):
        raise ValueError("saturation must have one row or one row per face")
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
