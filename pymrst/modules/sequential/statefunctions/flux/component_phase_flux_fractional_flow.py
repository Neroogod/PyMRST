"""MRST ``sequential/statefunctions/flux/ComponentPhaseFluxFractionalFlow.m``.

Component fluxes in each phase for a fixed total flux: the face component
mobility times `(vT + T sum_{j != ph} mob_j (G_ph - G_j))/mobT`.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.utils.numel_value import numel_value


class ComponentPhaseFluxFractionalFlow(StateFunction):
    """MRST `ComponentPhaseFluxFractionalFlow`."""

    def __init__(self, backend: Any, upwinding: Any = None):
        super().__init__(backend)
        self.depends_on(["FaceComponentMobility", "TotalFlux", "PhaseInterfacePressureDifferences",
                         "Transmissibility", "FaceMobility", "FaceTotalMobility"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[List[Any]]:
        ncomp = model.get_number_of_components()
        nph = model.get_number_of_phases()
        compMob, vT, G, T, fmob, mobT = self.get_evaluated_dependencies(
            state, "FaceComponentMobility", "TotalFlux", "PhaseInterfacePressureDifferences",
            "Transmissibility", "FaceMobility", "FaceTotalMobility")
        w = 1/mobT
        nf = numel_value(vT)
        kgrad = [None]*nph
        for i in range(nph):
            mobG = np.zeros(nf)
            for j in range(nph):
                if i != j:
                    mobG = mobG + fmob[j]*(G[i] - G[j])
            kgrad[i] = w*(vT + T*mobG)
        v = [[None]*nph for _ in range(ncomp)]
        for c in range(ncomp):
            for ph in range(nph):
                mob = compMob[c][ph]
                if mob is not None and (hasattr(mob, "jac") or np.size(mob)):
                    v[c][ph] = mob*kgrad[ph]
        return v

    evaluateOnDomain = evaluate_on_domain


__all__ = ["ComponentPhaseFluxFractionalFlow"]


if __name__ == "__main__":
    import warnings

    import numpy as np

    from pymrst.modules.sequential.testing import toy_case, toy_transport_state
    from pymrst.utils.adi import value

    warnings.simplefilter("ignore")
    T = toy_transport_state(toy_case())

    def test_each_component_flows_with_its_phase():
        """Immiscible: water only in the water phase, oil only in oil, both
        moving with the flood."""
        v = T.parent.get_prop(T.ad_state, "ComponentPhaseFlux")
        assert v[0][1] is None and v[1][0] is None
        assert np.all(value(v[0][0]) > 0) and np.all(value(v[1][1]) > 0)
        print("ok test_each_component_flows_with_its_phase")

    test_each_component_flows_with_its_phase()
    print("\nAll tests passed!")
