"""MRST ``ad-core/statefunctions/flux/TwoPointFluxApproximation.m``.

TPFA: the face flux driver is the face transmissibility times the potential
difference across the face.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.ad_oo.statefunctions.flux.permeability_gradient_discretization import (
    PermeabilityGradientDiscretization,
)


class TwoPointFluxApproximation(PermeabilityGradientDiscretization):
    """`T_f * dTheta`."""

    def __init__(self, model: Any):
        assert getattr(model.operators, "T", None) is not None, (
            "Transmissibility must be present in model.operators for TPFA "
            "discretization.")

    def get_permeability_gradient(self, model: Any, state: Any, dp: Any) -> Any:
        T = model.FlowDiscretization.get(model, state, "Transmissibility")
        return T*dp

    getPermeabilityGradient = get_permeability_gradient


__all__ = ["TwoPointFluxApproximation"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_flux_driver_is_transmissibility_times_difference():
        model = toy_model("W", nx=3)
        tpfa = TwoPointFluxApproximation(model)
        v = tpfa.get_permeability_gradient(model, toy_state(model),
                                           np.array([1.0, -2.0]))
        np.testing.assert_allclose(v, model.operators.T*np.array([1.0, -2.0]))
        print("ok test_flux_driver_is_transmissibility_times_difference")

    test_flux_driver_is_transmissibility_times_difference()
    print("\nAll tests passed!")
