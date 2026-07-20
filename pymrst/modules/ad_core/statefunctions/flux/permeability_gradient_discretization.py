"""Permeability-gradient discretizations (MRST ad-core/statefunctions/flux).

Ports MRST's `PermeabilityGradientDiscretization.m` (the abstract base) and
`TwoPointFluxApproximation.m` (the concrete two-point scheme).

The point of the abstraction: the flux graph should not care *how* `K grad p`
is discretized on a face. `PermeabilityPotentialGradient` (a state function)
holds a discretization object and asks it, per phase, for the face value of
`K` times a potential difference `dp`. TPFA answers `T .* dp`; an MPFA or
mimetic scheme would answer with its own stencil. Swapping the discretization
is then a one-line change, not a rewrite of the flux group - which is exactly
why MRST factors it out this way, and why it must be a separate object rather
than the hard-coded `T * pot` PyMRST's `PhaseFlux` uses today.
"""

from __future__ import annotations

from typing import Any


class PermeabilityGradientDiscretization:
    """Abstract discretization of ``K grad(p)`` on interior faces.

    A subclass turns a face potential difference ``dp`` (defined on interior
    faces) into the discretized ``K grad`` face value. The base class defines
    only the interface; it cannot be evaluated.
    """

    def get_permeability_gradient(self, model: Any, state: Any, dp: Any) -> Any:
        """Discretized ``K grad`` on interior faces for potential diff ``dp``."""
        raise NotImplementedError(
            "PermeabilityGradientDiscretization is abstract; use a concrete "
            "scheme such as TwoPointFluxApproximation"
        )


class TwoPointFluxApproximation(PermeabilityGradientDiscretization):
    """The two-point flux approximation: ``K grad -> T .* dp``.

    Mirrors MRST's `TwoPointFluxApproximation`. The transmissibility ``T`` is
    read from the flow-discretization grouping (`model.FlowDiscretization`),
    not captured at construction, so a model that recomputes ``T`` (a changed
    rock, a multiplier) is honoured on the next evaluation.
    """

    def __init__(self, model: Any = None):
        if model is not None:
            ops = getattr(model, "operators", None)
            assert ops is not None and getattr(ops, "T", None) is not None, (
                "Transmissibility must be present in model.operators for the "
                "TPFA discretization"
            )

    def get_permeability_gradient(self, model: Any, state: Any, dp: Any) -> Any:
        t = model.FlowDiscretization.get(model, state, "Transmissibility")
        return t * dp


__all__ = [
    "PermeabilityGradientDiscretization",
    "TwoPointFluxApproximation",
]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.statefunctions.flow_discretization import (
        Transmissibility,
    )
    from pymrst.modules.ad_core.statefunctions.state_function import (
        State,
        StateFunctionGrouping,
    )
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi import init_variables_adi

    class _Ops:
        """Minimal operators carrier: just interior transmissibility."""

        def __init__(self, g, rock):
            self.T = setup_operators_tpfa(g, rock).T

    class _Model:
        def __init__(self, g, rock):
            self.operators = _Ops(g, rock)
            self.FlowDiscretization = StateFunctionGrouping("FluxDisc")
            self.FlowDiscretization.set_state_function(
                "Transmissibility", Transmissibility(self))

    def _build():
        g = compute_geometry(cart_grid([4, 3, 1], [40.0, 30.0, 10.0]))
        rock = make_rock(g, 1e-13, 0.2)
        return _Model(g, rock), g

    def test_tpfa_requires_transmissibility():
        """Constructing TPFA on a model without T is refused, not silent."""
        class _Bare:
            operators = type("O", (), {"T": None})()
        try:
            TwoPointFluxApproximation(_Bare())
            raise SystemExit("should have refused a model without T")
        except AssertionError:
            pass
        print("ok test_tpfa_requires_transmissibility")

    def test_tpfa_returns_T_times_dp_numeric():
        """K grad = T .* dp for a plain numeric potential difference."""
        m, g = _build()
        n_if = m.operators.T.size
        dp = np.linspace(-2.0, 3.0, n_if)
        tpfa = TwoPointFluxApproximation(m)
        v = tpfa.get_permeability_gradient(m, State(), dp)
        assert np.allclose(v, m.operators.T * dp)
        print("ok test_tpfa_returns_T_times_dp_numeric")

    def test_tpfa_carries_ad_jacobian():
        """T .* dp propagates the Jacobian of an AD potential difference."""
        m, g = _build()
        n_if = m.operators.T.size
        (dp,) = init_variables_adi(np.linspace(-1.0, 1.0, n_if))
        tpfa = TwoPointFluxApproximation(m)
        v = tpfa.get_permeability_gradient(m, State(), dp)
        assert np.allclose(v.val, m.operators.T * dp.val)
        # d(T*dp)/d(dp) = diag(T)
        assert np.allclose(v.jac[0].toarray(), np.diag(m.operators.T))
        print("ok test_tpfa_carries_ad_jacobian")

    def test_base_class_is_abstract():
        """The base discretization cannot be evaluated."""
        try:
            PermeabilityGradientDiscretization().get_permeability_gradient(
                None, None, None)
            raise SystemExit("base class should be abstract")
        except NotImplementedError:
            pass
        print("ok test_base_class_is_abstract")

    test_tpfa_requires_transmissibility()
    test_tpfa_returns_T_times_dp_numeric()
    test_tpfa_carries_ad_jacobian()
    test_base_class_is_abstract()
    print("\nAll tests passed!")
