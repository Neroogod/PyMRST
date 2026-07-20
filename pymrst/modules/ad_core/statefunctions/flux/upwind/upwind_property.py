"""UpwindProperty mixin (MRST ad-core/statefunctions/flux/upwind).

Ports MRST's `UpwindProperty.m`: a mixin that gives a state function a
`face_upstream(...)` method by holding an `UpwindDiscretization` and delegating
to it. `FaceComponentMobility` and `FaceMobility` inherit from it in MRST, so
they upwind through whatever scheme the model was configured with rather than
reaching for a fixed operator.

In Python this is multiple inheritance: a state function subclasses both
`StateFunction` and `UpwindProperty`. The mixin only adds behaviour (the
`face_upstream` delegate and the held discretization); it carries no state of
its own beyond that object, so co-operative construction is simple.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.statefunctions.flux.upwind.upwind_discretization import (  # noqa: E501
    UpwindDiscretization,
    UpwindFunctionWrapperDiscretization,
)


class UpwindProperty:
    """Adds `face_upstream` to a state function via a held discretization."""

    def __init__(self, upstream: UpwindDiscretization):
        assert isinstance(upstream, UpwindDiscretization), (
            "UpwindProperty needs an UpwindDiscretization instance"
        )
        self.upwind_discretization = upstream

    def face_upstream(self, model: Any, state: Any, flag: Any,
                      cell_value: Any) -> Any:
        return self.upwind_discretization.face_upstream(
            model, state, flag, cell_value)


__all__ = ["UpwindProperty"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.statefunctions.state_function import (
        StateFunction,
    )
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi import init_variables_adi, value

    def _model():
        g = compute_geometry(cart_grid([4, 4, 1], [40.0, 40.0, 10.0]))
        rock = make_rock(g, 1e-13, 0.2)
        ops = setup_operators_tpfa(g, rock)
        return type("M", (), {"operators": ops})(), g

    def test_mixin_delegates_to_discretization():
        """face_upstream on the mixin equals the discretization's own."""
        model, g = _model()
        disc = UpwindFunctionWrapperDiscretization(model)
        prop = UpwindProperty(disc)
        n_if = model.operators.T.size
        flag = np.arange(n_if) % 2 == 0
        x = np.linspace(0.0, 1.0, g.cells.num)
        assert np.allclose(prop.face_upstream(model, None, flag, x),
                           disc.face_upstream(model, None, flag, x))
        print("ok test_mixin_delegates_to_discretization")

    def test_statefunction_can_mix_it_in():
        """A StateFunction + UpwindProperty upwinds a cell field to faces."""
        model, g = _model()

        class _UpwindedField(StateFunction, UpwindProperty):
            def __init__(self, model):
                StateFunction.__init__(self, model)
                UpwindProperty.__init__(
                    self, UpwindFunctionWrapperDiscretization(model))

            def evaluate_on_domain(self, model, state):
                flag = np.ones(model.operators.T.size, bool)
                return self.face_upstream(model, state, flag, state["c"])

        sf = _UpwindedField(model)
        (c,) = init_variables_adi(np.arange(g.cells.num, dtype=float))
        out = sf.evaluate_on_domain(model, {"c": c})
        assert np.allclose(value(out), c.val[model.operators.N[:, 0]])
        assert out.jac[0].shape == (model.operators.N.shape[0], g.cells.num)
        print("ok test_statefunction_can_mix_it_in")

    def test_requires_discretization_instance():
        """The mixin rejects a non-discretization argument."""
        try:
            UpwindProperty(lambda flag, x: x)
            raise SystemExit("should reject a bare callable")
        except AssertionError:
            pass
        print("ok test_requires_discretization_instance")

    test_mixin_delegates_to_discretization()
    test_statefunction_can_mix_it_in()
    test_requires_discretization_instance()
    print("\nAll tests passed!")
