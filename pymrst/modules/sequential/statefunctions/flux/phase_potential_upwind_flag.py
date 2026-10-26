"""MRST ``sequential/statefunctions/flux/PhasePotentialUpwindFlag.m``.

Upwind flags for a fixed total flux: `multiphaseUpwindIndices` of the phases'
interface pressure differences, the total flux (or zero, without
`includeTotalVelocity`), transmissibilities and cell mobilities.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import STATE, StateFunction
from pymrst.utils.adi import value


class PhasePotentialUpwindFlag(StateFunction):
    """MRST `PhasePotentialUpwindFlag`."""

    def __init__(self, *args: Any, **options: Any):
        super().__init__(*args)
        self.includeTotalVelocity = options.pop("includeTotalVelocity", True)
        self.depends_on("flux", STATE)
        self.depends_on("Mobility", "FlowPropertyFunctions")
        self.depends_on(["PhaseInterfacePressureDifferences", "Transmissibility", "TotalFlux"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[np.ndarray]:
        from pymrst.modules.sequential.utils.multiphase_upwind_indices import (
            multiphase_upwind_indices,
        )
        G, T, vT = self.get_evaluated_dependencies(
            state, "PhaseInterfacePressureDifferences", "Transmissibility", "TotalFlux")
        mob = model.get_prop(state, "Mobility")
        v = vT if self.includeTotalVelocity else np.zeros(np.size(value(vT)))
        flag_array, _, _ = multiphase_upwind_indices(G, v, T, mob, model.operators.upw)
        return [flag_array[:, i] for i in range(len(mob))]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PhasePotentialUpwindFlag"]


if __name__ == "__main__":
    import warnings

    import numpy as np

    from pymrst.modules.sequential.testing import toy_case, toy_transport_state
    from pymrst.utils.adi import value

    warnings.simplefilter("ignore")
    T = toy_transport_state(toy_case())

    def test_cocurrent_flow_is_upwinded_from_the_left():
        flags = T.parent.get_prop(T.ad_state, "PhaseUpwindFlag")
        assert all(np.all(f) for f in flags)
        print("ok test_cocurrent_flow_is_upwinded_from_the_left")

    test_cocurrent_flow_is_upwinded_from_the_left()
    print("\nAll tests passed!")
