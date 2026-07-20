"""Run-length decoding, ported from MRST `rldecode`."""

import numpy as np


def rldecode(a: np.ndarray, n: np.ndarray, axis: int = 0) -> np.ndarray:
    """Repeat each element or row of `a` by the matching run length in `n`."""
    return np.repeat(np.asarray(a), np.asarray(n), axis=axis)


if __name__ == "__main__":

    def test_rldecode() -> None:
        out = rldecode(np.array([10, 20, 30]), np.array([2, 0, 3]))
        assert np.array_equal(out, [10, 10, 30, 30, 30])
        rows = rldecode(np.array([[1, 2], [3, 4]]), np.array([1, 2]))
        assert np.array_equal(rows, [[1, 2], [3, 4], [3, 4]])
        print("ok test_rldecode")

    test_rldecode()
    print("\nAll tests passed!")

