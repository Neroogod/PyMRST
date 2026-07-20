"""Upwind discretizations (MRST ad-core/statefunctions/flux/upwind).

Ports MRST's `UpwindDiscretization.m` (abstract base) and
`UpwindFunctionWrapperDiscretization.m` (the classical function-handle wrapper).

Upwinding chooses, for each interior face, which of the two neighbouring cell
values to carry across it - the one on the *upstream* side of the flow. It is
the hyperbolic (advective) half of a conservation law, and getting the side
right is what keeps a saturation front sharp and monotone.

PyMRST already has the single-point upwind as `model.operators.upw(flag, x)`,
validated bit-for-bit against MATLAB. This module wraps that operator behind
MRST's discretization interface so a state function can be handed *an upwind
object* instead of reaching for the operator directly - which is what makes the
scheme swappable (single-point today, WENO or a higher-order reconstruction
later) without touching the state functions that consume it.
"""

from __future__ import annotations

from typing import Any


class UpwindDiscretization:
    """Abstract upwind scheme: pick the upstream cell value on each face.

    A subclass implements ``face_upstream(model, state, flag, cell_values)``,
    returning the interior-face values taken from the upstream side, where
    ``flag[f]`` is True when the first neighbour of face ``f`` is upstream.
    """

    def __init__(self, model: Any = None):
        pass

    def face_upstream(self, model: Any, state: Any, flag: Any,
                      cell_values: Any) -> Any:
        raise NotImplementedError(
            "UpwindDiscretization is abstract; use a concrete scheme such as "
            "UpwindFunctionWrapperDiscretization"
        )


class UpwindFunctionWrapperDiscretization(UpwindDiscretization):
    """Wrap a face-upwind function handle (the classical MRST style).

    Defaults to `model.operators.upw` - PyMRST's validated single-point
    upwind. The wrapper exists so the same call site works whether the scheme
    is this thin wrapper or a higher-order reconstruction.
    """

    def __init__(self, model: Any = None, function_handle: Any = None):
        super().__init__(model)
        if function_handle is None:
            assert model is not None and getattr(model, "operators", None) is not None, (
                "UpwindFunctionWrapperDiscretization needs a model with "
                "operators, or an explicit function_handle"
            )
            function_handle = model.operators.upw
        assert callable(function_handle), "function_handle must be callable"
        self._function_handle = function_handle

    def set_function_handle(self, up: Any) -> "UpwindFunctionWrapperDiscretization":
        self._function_handle = up
        return self

    def face_upstream(self, model: Any, state: Any, flag: Any,
                      cell_values: Any) -> Any:
        return self._function_handle(flag, cell_values)


__all__ = [
    "UpwindDiscretization",
    "UpwindFunctionWrapperDiscretization",
]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi import init_variables_adi, value

    def _ops():
        g = compute_geometry(cart_grid([5, 4, 2], [50.0, 40.0, 20.0]))
        rock = make_rock(g, 1e-13, 0.2)
        return setup_operators_tpfa(g, rock), g

    def test_wrapper_matches_operator_numeric():
        """The wrapper reproduces model.operators.upw for a numeric field."""
        ops, g = _ops()
        model = type("M", (), {"operators": ops})()
        up = UpwindFunctionWrapperDiscretization(model)
        n_if = ops.T.size
        rng = np.random.default_rng(0)
        flag = rng.random(n_if) < 0.5
        x = rng.random(g.cells.num)
        assert np.allclose(up.face_upstream(model, None, flag, x),
                           ops.upw(flag, x))
        print("ok test_wrapper_matches_operator_numeric")

    def test_wrapper_matches_operator_ad():
        """The wrapper carries the Jacobian, exactly as the operator does."""
        ops, g = _ops()
        model = type("M", (), {"operators": ops})()
        up = UpwindFunctionWrapperDiscretization(model)
        n_if = ops.T.size
        flag = (np.arange(n_if) % 2 == 0)
        (x,) = init_variables_adi(np.linspace(1.0, 2.0, g.cells.num))
        a = up.face_upstream(model, None, flag, x)
        b = ops.upw(flag, x)
        assert np.allclose(value(a), value(b))
        assert np.allclose((a - b).jac[0].toarray(), 0.0)
        print("ok test_wrapper_matches_operator_ad")

    def test_wrapper_picks_upstream_side():
        """flag True takes the first neighbour, False the second."""
        ops, g = _ops()
        model = type("M", (), {"operators": ops})()
        up = UpwindFunctionWrapperDiscretization(model)
        n = ops.N
        x = np.arange(g.cells.num, dtype=float)
        n_if = n.shape[0]
        allT = up.face_upstream(model, None, np.ones(n_if, bool), x)
        allF = up.face_upstream(model, None, np.zeros(n_if, bool), x)
        assert np.allclose(allT, x[n[:, 0]])
        assert np.allclose(allF, x[n[:, 1]])
        print("ok test_wrapper_picks_upstream_side")

    def test_explicit_handle_and_base_abstract():
        """A wrapper takes an explicit handle; the base cannot be evaluated."""
        up = UpwindFunctionWrapperDiscretization(
            function_handle=lambda flag, x: x * 0 + 7.0)
        assert np.allclose(up.face_upstream(None, None, None, np.zeros(3)), 7.0)
        try:
            UpwindDiscretization().face_upstream(None, None, None, None)
            raise SystemExit("base should be abstract")
        except NotImplementedError:
            pass
        print("ok test_explicit_handle_and_base_abstract")

    test_wrapper_matches_operator_numeric()
    test_wrapper_matches_operator_ad()
    test_wrapper_picks_upstream_side()
    test_explicit_handle_and_base_abstract()
    print("\nAll tests passed!")
