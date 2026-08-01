"""Multi-range index helper, ported from MRST `mcolon`."""

import numpy as np


def mcolon(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Concatenate ranges `[lo[i], hi[i])` into one index array."""
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    lengths = hi - lo
    keep = lengths > 0
    lo, lengths = lo[keep], lengths[keep]
    if lengths.size == 0:
        return np.zeros(0, dtype=np.int64)
    csum = np.cumsum(lengths)
    run_starts = csum - lengths
    offset = np.arange(csum[-1]) - np.repeat(run_starts, lengths)
    return np.repeat(lo, lengths) + offset


if __name__ == "__main__":

    def test_mcolon() -> None:
        out = mcolon(np.array([0, 5]), np.array([3, 7]))
        assert np.array_equal(out, [0, 1, 2, 5, 6])
        out = mcolon(np.array([2, 4, 9]), np.array([2, 6, 9]))
        assert np.array_equal(out, [4, 5])
        assert mcolon(np.array([3]), np.array([3])).size == 0
        print("ok test_mcolon")

    def test_mcolon_facepos_pattern() -> None:
        pos = np.array([0, 6, 12, 18])
        assert np.array_equal(mcolon(pos[:-1], pos[1:]), np.arange(18))
        print("ok test_mcolon_facepos_pattern")

    test_mcolon()
    test_mcolon_facepos_pattern()
    print("\nAll tests passed!")

