"""MRST ``sequential/statefunctions/flux/ComponentPhaseMassTotalSaturation.m``.

The component phase masses of `ComponentPhaseMass`, each scaled by the total saturation
`sT` (the transport model's total-saturation formulation).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.component_phase_mass import ComponentPhaseMass


class ComponentPhaseMassTotalSaturation(ComponentPhaseMass):
    """MRST `ComponentPhaseMassTotalSaturation`: `ComponentPhaseMass` times `TotalSaturation`."""

    def __init__(self, model: Any, *args: Any, **options: Any):
        super().__init__(model, *args, **options)
        self.depends_on("TotalSaturation")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[List[Any]]:
        v = super().evaluate_on_domain(model, state)
        sT = self.get_evaluated_dependencies(state, "TotalSaturation")
        return [[x if _empty(x) else x*sT for x in row] for row in v]

    evaluateOnDomain = evaluate_on_domain


def _empty(x: Any) -> bool:
    return x is None or (not hasattr(x, "jac") and np.size(x) == 0)


__all__ = ["ComponentPhaseMassTotalSaturation"]


if __name__ == "__main__":
    import warnings

    import numpy as np

    from pymrst.modules.sequential.testing import toy_case, toy_transport_state
    from pymrst.utils.adi import value

    warnings.simplefilter("ignore")
    T = toy_transport_state(toy_case())

    def test_it_is_the_base_mass_times_the_total_saturation():
        """With `sT = 0.5` everywhere each entry is half the base value."""
        from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops import (
            component_phase_mass as base_module,
        )
        from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
        state = copy_state(T.state)
        state["sT"] = np.full(np.shape(T.state["s"])[0], 0.5)
        state = T.model.get_state_ad(state)[0]
        scaled = T.parent.get_prop(state, "ComponentPhaseMass")
        plain = base_module.ComponentPhaseMass(T.parent).evaluate_on_domain(T.parent, state)
        for row_s, row_p in zip(scaled, plain):
            for a, b in zip(row_s, row_p):
                if b is not None:
                    np.testing.assert_allclose(value(a), 0.5*value(b), rtol=1e-14)
        print("ok test_it_is_the_base_mass_times_the_total_saturation")

    test_it_is_the_base_mass_times_the_total_saturation()
    print("\nAll tests passed!")
