"""Run-length encoding, ported from MRST `rlencode`."""

from typing import Tuple

import numpy as np


def rlencode(a: np.ndarray, dim: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Compress equal adjacent layers along ``dim``.

    Python dimensions are zero-based, so MRST's ``dim=1`` is ``dim=0`` here.
    NaNs in corresponding positions compare equal, matching MRST.
    """
    arr = np.asarray(a)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if dim < 0:
        dim += arr.ndim
    if not 0 <= dim < arr.ndim:
        raise np.AxisError(dim, ndim=arr.ndim)
    if arr.shape[dim] == 0:
        return arr, np.zeros(0, dtype=np.int64)

    moved = np.moveaxis(arr, dim, 0)
    top, bottom = moved[:-1], moved[1:]
    equal = (top == bottom) | (np.isnan(top) & np.isnan(bottom)) \
        if np.issubdtype(arr.dtype, np.number) else top == bottom
    if moved.ndim == 1:
        change = ~equal
    else:
        change = ~np.all(equal.reshape(equal.shape[0], -1), axis=1)
    starts = np.flatnonzero(np.concatenate(([True], change)))
    lengths = np.diff(np.concatenate((starts, [moved.shape[0]])))
    return np.moveaxis(moved[starts], 0, dim), lengths


if __name__ == "__main__":
    from pymrst.utils.rldecode import rldecode

    def test_rlencode() -> None:
        arr = np.array([5, 5, 7, 7, 7, 5])
        values, lengths = rlencode(arr)
        assert np.array_equal(values, [5, 7, 5])
        assert np.array_equal(lengths, [2, 3, 1])
        assert np.array_equal(rldecode(values, lengths), arr)
        print("ok test_rlencode")

    def test_rlencode_empty() -> None:
        values, lengths = rlencode(np.zeros(0))
        assert values.size == 0
        assert lengths.size == 0
        print("ok test_rlencode_empty")

    test_rlencode()
    test_rlencode_empty()
    print("\nAll tests passed!")
