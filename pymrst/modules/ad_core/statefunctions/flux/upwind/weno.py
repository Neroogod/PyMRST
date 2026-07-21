"""WENO upwind discretization (MRST ad-core/statefunctions/flux/upwind).

Ports the one-dimensional path of MRST's `WENOUpwindDiscretization.m` - the
weighted essentially-non-oscillatory reconstruction of a cell field onto the
interior faces. Where single-point upwind takes the raw upstream cell value,
WENO reconstructs a higher-order face value from two linear stencils (one biased
to each side of the upstream cell) and blends them by a smoothness indicator:
in smooth regions both stencils contribute and the scheme is second order; near
a shock the oscillatory stencil is down-weighted, keeping the front monotone.

For each interior face with upstream cell ``mid`` (chosen by the upwind flag)
and its ordered left/right neighbours:

    q_L, B_L = linear reconstruction from (left, mid)   evaluated at the face
    q_R, B_R = linear reconstruction from (mid, right)  evaluated at the face
    w_L = B_L/(B_L+B_R),  w_R = B_R/(B_L+B_R)
    q_face = w_L q_L + w_R q_R

with the smoothness weight ``B = (dx * sigma^2 + E)^(-L)`` (E=1e-7, L=2), where
``sigma`` is the stencil slope. This matches MRST's `getFaceValue1D` /
`interpolate1D`, validated against a MATLAB dump in `validate_phase348.py`.

Only the 1D scheme is ported (the case `wenoExampleAD` and the Buckley-Leverett
front exercise). MRST's 2D/3D WENO uses a triangulation-based reconstruction
(`getFaceValueTri`) that is a separate, larger port.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from pymrst.modules.ad_core.statefunctions.flux.upwind.upwind_discretization import (  # noqa: E501
    UpwindDiscretization,
)

_E = 1e-7
_L = 2


class WENOUpwindDiscretization(UpwindDiscretization):
    """One-dimensional WENO face reconstruction."""

    def __init__(self, model: Any = None, dim: int = 1):
        super().__init__(model)
        assert dim == 1, "only the 1D WENO scheme is ported"
        self.dim = dim
        self._x: Optional[np.ndarray] = None
        self._xf: Optional[np.ndarray] = None
        self._neighbors: Optional[np.ndarray] = None   # [left, right] per cell
        if model is not None:
            self._setup_1d(model)

    def _setup_1d(self, model: Any) -> None:
        g = model.G
        x = np.asarray(g.cells.centroids[:, 0], dtype=float)
        n = model.operators.N
        internal = np.all(g.faces.neighbors >= 0, axis=1)
        xf = np.asarray(g.faces.centroids[internal, 0], dtype=float)

        # cellNeighbors[cell, {0:left, 1:right}] - the neighbour on each side,
        # ordered by x. -1 marks "no neighbour" (a boundary). MRST uses 0 as
        # that marker because its cell indices are 1-based; in PyMRST's 0-based
        # scheme cell 0 is a real cell, so a distinct sentinel is required.
        neighbors = np.full((g.cells.num, 2), -1, dtype=int)
        left, right = n[:, 0], n[:, 1]
        # for the right cell R of each face, its neighbour L is on the smaller
        # side if the face lies below R's centre
        smaller = xf < x[right]
        neighbors[right[smaller], 0] = left[smaller]
        neighbors[right[~smaller], 1] = left[~smaller]
        smaller = xf < x[left]
        neighbors[left[smaller], 0] = right[smaller]
        neighbors[left[~smaller], 1] = right[~smaller]

        self._x, self._xf, self._neighbors = x, xf, neighbors

    def _interpolate_1d(self, l: np.ndarray, r: np.ndarray, v: Any,
                        xq: np.ndarray):
        """Linear reconstruction from cells (l, r) at face point xq, and the
        smoothness weight. Mirrors MRST's `interpolate1D`."""
        l = l.copy()
        r = r.copy()
        # constant extrapolation at boundaries (a -1 index means "no cell")
        l[l == -1] = r[l == -1]
        r[r == -1] = l[r == -1]

        x = self._x
        x_l, x_r = x[l], x[r]
        dx = x_r - x_l
        bad = dx == 0
        dx = np.where(bad, 1.0, dx)

        v_l = v[l]
        v_r = v[r]
        sigma = (v_r - v_l) / dx
        if np.any(bad):
            sigma = sigma * (~bad)                     # AD-safe zero on bad
        vq = v_l + sigma * (xq - x_l)

        sigma_v = np.asarray(_value(sigma), dtype=float)
        smoothness = (dx * sigma_v ** 2 + _E) ** (-_L)
        return vq, smoothness

    def face_upstream(self, model: Any, state: Any, flag: Any,
                      cell_values: Any) -> Any:
        n = model.operators.N
        flag = np.asarray(flag, dtype=bool)
        mid = np.where(flag, n[:, 0], n[:, 1])

        left = self._neighbors[mid, 0]
        right = self._neighbors[mid, 1]
        xq = self._xf

        q_l, b_l = self._interpolate_1d(left, mid, cell_values, xq)
        q_r, b_r = self._interpolate_1d(mid, right, cell_values, xq)

        b_t = b_l + b_r
        w_l = b_l / b_t
        w_r = b_r / b_t
        return w_l * q_l + w_r * q_r


def _value(x: Any) -> Any:
    return x.val if hasattr(x, "val") else x


__all__ = ["WENOUpwindDiscretization"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi import init_variables_adi, value

    def _model(nx=20):
        g = compute_geometry(cart_grid([nx, 1, 1], [float(nx), 1.0, 1.0]))
        rock = make_rock(g, 1e-13, 1.0)
        ops = setup_operators_tpfa(g, rock)
        return type("M", (), {"G": g, "operators": ops})(), g

    def test_smooth_field_is_second_order_exact():
        """A linear field is reconstructed exactly on interior faces, where
        both stencils are full (the two boundary-adjacent faces have a
        one-sided stencil and are handled by the smoothness weighting, exactly
        as MRST does - see the MATLAB cross-check in validate_phase348)."""
        model, g = _model()
        weno = WENOUpwindDiscretization(model, 1)
        xc = g.cells.centroids[:, 0]
        v = 3.0 + 2.0 * xc                         # linear
        n_if = model.operators.T.size
        out = np.asarray(weno.face_upstream(model, None,
                                            np.ones(n_if, bool), v))
        xf = g.faces.centroids[np.all(g.faces.neighbors >= 0, axis=1), 0]
        interior = slice(1, n_if)                  # exclude the boundary face
        assert np.allclose(out[interior], (3.0 + 2.0 * xf)[interior])
        print("ok test_smooth_field_is_second_order_exact")

    def test_weno_is_bounded_near_a_step():
        """Near a monotone step WENO does not overshoot outside [min,max]."""
        model, g = _model()
        weno = WENOUpwindDiscretization(model, 1)
        xc = g.cells.centroids[:, 0]
        v = (xc > 10).astype(float)
        n_if = model.operators.T.size
        out = np.asarray(weno.face_upstream(model, None,
                                            np.ones(n_if, bool), v))
        assert out.min() >= -1e-9 and out.max() <= 1.0 + 1e-9
        print("ok test_weno_is_bounded_near_a_step")

    def test_carries_ad_jacobian():
        """WENO of an AD field propagates a Jacobian (it is a linear combo)."""
        model, g = _model(8)
        weno = WENOUpwindDiscretization(model, 1)
        (v,) = init_variables_adi(np.linspace(0.0, 1.0, g.cells.num))
        n_if = model.operators.T.size
        out = weno.face_upstream(model, None, np.ones(n_if, bool), v)
        assert hasattr(out, "jac")
        assert out.jac[0].shape == (n_if, g.cells.num)
        print("ok test_carries_ad_jacobian")

    test_smooth_field_is_second_order_exact()
    test_weno_is_bounded_near_a_step()
    test_carries_ad_jacobian()
    print("\nAll tests passed!")
