"""MRST ``sequential/statefunctions/flux/PhaseFluxFixedTotalVelocity.m``.

Phase fluxes for a fixed total flux `vT`: the fractional-flow form
`v_i = (mob_i/mobT) (vT + T sum_{j != i} mob_j (G_i - G_j))`, with face
mobilities and the phases' interface pressure differences `G`.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class PhaseFluxFixedTotalVelocity(StateFunction):
    """MRST `PhaseFluxFixedTotalVelocity`."""

    def __init__(self, *args: Any, **options: Any):
        super().__init__(*args, **options)
        self.depends_on(["TotalFlux", "PhaseInterfacePressureDifferences", "Transmissibility",
                         "FaceMobility", "FaceTotalMobility"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        vT, G, T, fmob, mobT = self.get_evaluated_dependencies(
            state, "TotalFlux", "PhaseInterfacePressureDifferences", "Transmissibility",
            "FaceMobility", "FaceTotalMobility")
        nph = len(fmob)
        f = [None]*nph
        for i in range(nph):
            mobG = 0
            for j in range(nph):
                if i != j:
                    mobG = mobG + fmob[j]*(G[i] - G[j])
            f[i] = (fmob[i]/mobT)*(vT + T*mobG)
        return f

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PhaseFluxFixedTotalVelocity"]


if __name__ == "__main__":
    import warnings

    import numpy as np

    from pymrst.modules.sequential.testing import toy_case, toy_transport_state
    from pymrst.utils.adi import value

    warnings.simplefilter("ignore")
    T = toy_transport_state(toy_case())

    def test_the_phase_fluxes_add_up_to_the_total():
        v = T.parent.get_prop(T.ad_state, "PhaseFlux")
        vT = T.parent.get_prop(T.ad_state, "TotalFlux")
        np.testing.assert_allclose(sum(value(x) for x in v), vT, rtol=1e-12)
        print("ok test_the_phase_fluxes_add_up_to_the_total")

    test_the_phase_fluxes_add_up_to_the_total()
    print("\nAll tests passed!")
