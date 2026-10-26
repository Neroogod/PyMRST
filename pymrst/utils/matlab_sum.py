"""MATLAB's `sum` of a double vector, in MATLAB's order of additions.

Floating-point sums depend on the order of the additions, and where code
compares a sum (a remainder `time - sum(dt)` against zero, say), the order
decides the outcome. MATLAB (R2021b, as run for the MRST references here)
adds up to `SEQUENTIAL_MAX` elements one after another, and longer vectors in
`LANES` interleaved partial sums, element `i` going to lane `i mod 4`, which
are then added in order. This was fitted to MATLAB's results on random
vectors of 1 to 20,000 elements; much longer vectors are summed by several
threads, and the order there is not reproduced. numpy's `sum` is pairwise and
agrees with MATLAB only by chance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SEQUENTIAL_MAX = 43
LANES = 4


def matlab_sum(x: Any) -> float:
    """MATLAB `sum(x)` of a vector, added in MATLAB's order."""
    values = np.asarray(x, dtype=float).ravel()
    if values.size <= SEQUENTIAL_MAX:
        total = 0.0
        for v in values.tolist():
            total += v
        return total
    n = values.size - values.size % LANES
    acc = values[:n].reshape(-1, LANES)
    partial = np.cumsum(acc, axis=0)[-1] if n else np.zeros(LANES)
    partial = partial.tolist()
    for i, v in enumerate(values[n:].tolist()):
        partial[i] += v
    total = 0.0
    for p in partial:
        total += p
    return total


__all__ = ["matlab_sum"]


if __name__ == "__main__":
    def test_short_vectors_add_in_order():
        x = [1e16, 1.0, -1e16, 1.0]
        assert matlab_sum(x) == ((1e16 + 1.0) - 1e16) + 1.0 == 1.0
        assert matlab_sum([]) == 0.0
        print("ok test_short_vectors_add_in_order")

    def test_long_vectors_use_four_lanes():
        dt = 10*365.2425*86400/100
        # MATLAB's sum(repmat(dt, 99, 1)), printed with %.17g.
        assert matlab_sum([dt]*99) == 312413824.80000013
        assert float(np.sum([dt]*99)) != 312413824.80000013
        print("ok test_long_vectors_use_four_lanes")

    def test_tail_goes_to_the_first_lanes():
        x = np.arange(1.0, 47.0)
        lanes = [sum(x[i::4]) for i in range(4)]
        assert matlab_sum(x) == ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3]
        print("ok test_tail_goes_to_the_first_lanes")

    test_short_vectors_add_in_order()
    test_long_vectors_use_four_lanes()
    test_tail_goes_to_the_first_lanes()
    print("\nAll tests passed!")
