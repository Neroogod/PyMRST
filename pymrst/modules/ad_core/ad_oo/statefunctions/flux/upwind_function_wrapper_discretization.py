"""MRST ``ad-core/statefunctions/flux/upwind/UpwindFunctionWrapperDiscretization.m``.

Single-point upwinding through a function handle - the model operators'
`faceUpstr` (PyMRST: `upw`) by default.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.ad_oo.statefunctions.flux.upwind_discretization import (
    UpwindDiscretization,
)
from pymrst.utils.adi import ad_matmul


class UpwindFunctionWrapperDiscretization(UpwindDiscretization):
    """Upwind with `function_handle(flag, cellvalue)`."""

    def __init__(self, model: Any):
        super().__init__(model)
        if model.operators is None:
            from pymrst.gridprocessing import get_neighbourship
            N = np.asarray(get_neighbourship(model.G), dtype=np.int64)
            nc = model.G.cells.num
            up = lambda flag, x: _face_upstr(flag, x, N, nc)
        else:
            up = model.operators.upw
        assert callable(up)
        self.function_handle: Callable = up

    def face_upstream(self, model: Any, state: Any, flag: Any,
                      cellvalue: Any) -> Any:
        return self.function_handle(flag, cellvalue)

    def set_function_handle(self, up: Callable):
        self.function_handle = up
        return self

    faceUpstream = face_upstream
    setFunctionHandle = set_function_handle


def _face_upstr(flag: Any, x: Any, N: np.ndarray, nc: int) -> Any:
    """MRST `faceUpstr`: `x(N(:,1))` where the flag is set, else `x(N(:,2))`."""
    nf = N.shape[0]
    cells = np.where(np.asarray(flag, dtype=bool), N[:, 0], N[:, 1])
    pick = sp.csr_array((np.ones(nf), (np.arange(nf), cells)), shape=(nf, nc))
    return ad_matmul(pick, x)


__all__ = ["UpwindFunctionWrapperDiscretization"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model

    def test_the_default_handle_is_the_operators_upwind():
        model = toy_model("W", nx=3)
        up = UpwindFunctionWrapperDiscretization(model)
        v = up.face_upstream(model, None, np.array([True, False]),
                             np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(v, [1.0, 3.0])
        print("ok test_the_default_handle_is_the_operators_upwind")

    def test_the_grid_fallback_picks_the_same_cells():
        N = np.array([[0, 1], [1, 2]])
        v = _face_upstr(np.array([True, False]), np.array([1.0, 2.0, 3.0]), N, 3)
        np.testing.assert_allclose(v, [1.0, 3.0])
        print("ok test_the_grid_fallback_picks_the_same_cells")

    test_the_default_handle_is_the_operators_upwind()
    test_the_grid_fallback_picks_the_same_cells()
    print("\nAll tests passed!")
