"""MRST ``sequential/utils/multiphaseUpwindIndices.m``.

Phase-potential upwinding for a fixed total flux (Brenier & Jaffre): per
face, the phases are sorted by their interface pressure difference `G`, and
each phase's "total potential" `theta_l = vT + T sum_{j != l} (G_l - G_j) K_j`
is formed with the mobilities `K_j` of the cell upstream for that pair (the
left cell for phases sorted after `l`, the right cell otherwise). The phases
from the last one with `theta <= 0` onwards flow from the left cell; the
rest from the right. Returns `(up, theta, r)`: the upwind flags in the
original phase order, the potentials (sorted order) and the split index
(MRST's, 1-based; 0 means every phase flows from the left).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import numpy as np

from pymrst.utils.adi import value


def multiphase_upwind_indices(G: Sequence[Any], vT: Any, T: Any, K: Sequence[Any],
                              upstr: Callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MRST `[up, theta, r] = multiphaseUpwindIndices(G, vT, T, K, upstr)`."""
    nPh = len(G)
    T = np.asarray(value(T), dtype=float).ravel()
    nF = T.size
    r = np.zeros(nF, dtype=np.int64)
    theta = np.tile(np.asarray(value(vT), dtype=float).reshape(-1, 1), (1, nPh))
    if nF == 0:
        return np.ones((nF, nPh), dtype=bool), theta, r
    G = np.column_stack([np.asarray(value(g), dtype=float).ravel() for g in G])
    K = np.column_stack([np.asarray(value(k), dtype=float).ravel() for k in K])
    sortInd = np.argsort(G, axis=1, kind="stable")
    G = np.take_along_axis(G, sortInd, axis=1)
    flag = np.ones(nF, dtype=bool)
    left_upwind = np.column_stack([np.asarray(upstr(flag, K[:, k])) for k in range(nPh)])
    right_upwind = np.column_stack([np.asarray(upstr(~flag, K[:, k])) for k in range(nPh)])
    left_upwind = np.take_along_axis(left_upwind, sortInd, axis=1)
    right_upwind = np.take_along_axis(right_upwind, sortInd, axis=1)
    for l in range(nPh):
        for j in range(nPh):
            if j == l:
                continue
            kj = left_upwind[:, j] if j > l else right_upwind[:, j]
            theta[:, l] = theta[:, l] + T*(G[:, l] - G[:, j])*kj
    for l in range(nPh):
        r[theta[:, l] <= 0] = l + 1
    up_sorted = np.zeros((nF, nPh), dtype=bool)
    for l in range(nPh):
        up_sorted[:, l] = (l + 1) > r
    up = np.zeros((nF, nPh), dtype=bool)
    np.put_along_axis(up, sortInd, up_sorted, axis=1)        # MRST's `up(subs) = up`
    return up, theta, r


multiphaseUpwindIndices = multiphase_upwind_indices

__all__ = ["multiphaseUpwindIndices", "multiphase_upwind_indices"]


if __name__ == "__main__":
    def _upstr(flag, x):
        """Two cells, one face: left cell 0, right cell 1."""
        return np.where(flag, x[0], x[1])

    def test_without_gravity_the_total_flux_decides():
        """G = 0: every phase follows vT."""
        for vT, expected in ((1.0, [True, True]), (-1.0, [False, False])):
            up, _, _ = multiphase_upwind_indices([np.zeros(1), np.zeros(1)], np.array([vT]),
                                                 np.array([1.0]), [np.ones(2), np.ones(2)], _upstr)
            assert up[0].tolist() == expected, (vT, up)
        print("ok test_without_gravity_the_total_flux_decides")

    def test_counter_current_gravity_segregation():
        """No net flux, the heavy phase pushed left-to-right by gravity
        (G larger) flows from the left, the light one from the right."""
        up, _, _ = multiphase_upwind_indices([np.array([1.0]), np.array([-1.0])], np.zeros(1),
                                             np.array([1.0]), [np.ones(2), np.ones(2)], _upstr)
        assert up[0].tolist() == [True, False], up
        print("ok test_counter_current_gravity_segregation")

    test_without_gravity_the_total_flux_decides()
    test_counter_current_gravity_segregation()
    print("\nAll tests passed!")
