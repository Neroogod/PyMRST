"""Run-length encoding, ported from MRST `rlencode`."""

from typing import Tuple

import numpy as np


def rlencode(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compress runs of identical elements or rows."""
    arr = np.asarray(a)
    if arr.shape[0] == 0:
        return arr, np.zeros(0, dtype=np.int64)
    if arr.ndim == 1:
        change = arr[1:] != arr[:-1]
    else:
        change = np.any(arr[1:] != arr[:-1], axis=1)
    starts = np.flatnonzero(np.concatenate(([True], change)))
    lengths = np.diff(np.concatenate((starts, [arr.shape[0]])))
    return arr[starts], lengths


if __name__ == "__main__":
    from pymrst.utils.gridtools.rldecode import rldecode

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

