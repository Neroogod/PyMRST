"""Reservoir-state initialization, ported from MRST `initResSol`."""

from typing import Union

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.solvers.state_types import State

Scalar = Union[float, np.ndarray]


def init_res_sol(g: Grid, p0: Scalar, s0: Scalar = 1.0) -> State:
    """Initialize reservoir pressure, face fluxes, and saturations."""
    nc = g.cells.num
    pressure = _cell_vector(p0, nc)
    saturation = np.atleast_2d(np.asarray(s0, dtype=np.float64))
    if saturation.shape[0] == 1:
        saturation = np.repeat(saturation, nc, axis=0)
    assert saturation.shape[0] == nc, "saturation must be scalar or one row per cell"
    return State(pressure=pressure, flux=np.zeros(g.faces.num), s=saturation)


def _cell_vector(value: Scalar, count: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 1:
        return np.full(count, arr[0])
    assert arr.size == count, "pressure must be scalar or one value per cell"
    return arr.copy()


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid

    def test_init_res_sol() -> None:
        g = cart_grid([2, 2, 1])
        state = init_res_sol(g, 1e7, 1.0)
        assert state.pressure.shape == (4,)
        assert np.allclose(state.pressure, 1e7)
        assert state.s.shape == (4, 1)
        assert state.flux.shape == (g.faces.num,)
        print("ok test_init_res_sol")

    test_init_res_sol()
    print("\nAll tests passed!")

