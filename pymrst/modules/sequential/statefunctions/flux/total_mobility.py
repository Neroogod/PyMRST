"""MRST ``sequential/statefunctions/flux/TotalMobility.m``.

The total mobility in each cell: the sum of the phase mobilities.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class TotalMobility(StateFunction):
    """MRST `TotalMobility`."""

    def __init__(self, *args: Any, **options: Any):
        super().__init__(*args, **options)
        self.depends_on("Mobility")

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        mob = self.get_evaluated_dependencies(state, "Mobility")
        mobT = 0
        for m in mob:
            mobT = mobT + m
        return mobT

    evaluateOnDomain = evaluate_on_domain


__all__ = ["TotalMobility"]


if __name__ == "__main__":
    import warnings

    import numpy as np

    from pymrst.modules.sequential.testing import toy_case, toy_transport_state
    from pymrst.utils.adi import value

    warnings.simplefilter("ignore")
    T = toy_transport_state(toy_case())

    def test_it_sums_the_phase_mobilities():
        fn = TotalMobility(T.parent)
        mob = T.parent.get_prop(T.ad_state, "Mobility")
        fn.get_evaluated_dependencies = lambda state, name: mob
        np.testing.assert_allclose(value(fn.evaluate_on_domain(T.parent, T.ad_state)),
                                   sum(value(m) for m in mob))
        print("ok test_it_sums_the_phase_mobilities")

    test_it_sums_the_phase_mobilities()
    print("\nAll tests passed!")
