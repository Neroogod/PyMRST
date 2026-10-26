"""MRST ``sequential/statefunctions/flux/PhaseInterfacePressureDifferences.m``.

Each phase's pressure difference across the faces apart from the common
pressure: minus the gravity potential difference, minus the gradient of the
phase's capillary pressure where it has one.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class PhaseInterfacePressureDifferences(StateFunction):
    """MRST `PhaseInterfacePressureDifferences`."""

    def __init__(self, *args: Any, **options: Any):
        super().__init__(*args, **options)
        self.depends_on("CapillaryPressure", "FlowPropertyFunctions")
        self.depends_on(["GravityPotentialDifference"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        pot = self.get_evaluated_dependencies(state, "GravityPotentialDifference")
        pc = model.get_prop(state, "CapillaryPressure")
        G = list(pot)
        for i in range(len(G)):
            G[i] = -pot[i]
            if pc[i] is not None and (hasattr(pc[i], "jac") or np.size(pc[i])):
                G[i] = G[i] - model.operators.grad(pc[i])
        return G

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PhaseInterfacePressureDifferences"]


if __name__ == "__main__":
    import warnings

    import numpy as np

    from pymrst.modules.sequential.testing import toy_case, toy_transport_state
    from pymrst.utils.adi import value

    warnings.simplefilter("ignore")
    T = toy_transport_state(toy_case())

    def test_without_gravity_or_capillarity_they_vanish():
        G = T.parent.get_prop(T.ad_state, "PhaseInterfacePressureDifferences")
        assert len(G) == 2
        for g in G:
            np.testing.assert_allclose(value(g), 0.0, atol=1e-12)
        print("ok test_without_gravity_or_capillarity_they_vanish")

    test_without_gravity_or_capillarity_they_vanish()
    print("\nAll tests passed!")
