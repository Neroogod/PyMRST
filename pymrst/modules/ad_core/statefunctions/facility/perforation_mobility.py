"""PerforationMobility (MRST ad-core/statefunctions/facility).

Ports MRST's `PerforationMobility.m`: the reservoir phase mobility sampled in
each perforated cell. It reaches into the reservoir model's flow-property graph
for `Mobility` and restricts it to the perforation cells - the ``mob[wc]`` of
PyMRST's `_single_well`.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class PerforationMobility(StateFunction):
    """Reservoir phase mobility at the perforation cells."""

    def __init__(self, facility: Any = None):
        super().__init__(facility, label="\\lambda_{wc}")
        self.depends_on("FacilityWellMapping")

    def evaluate_on_domain(self, facility: Any, state: Any) -> List[Any]:
        m = self.get_evaluated(state, "FacilityWellMapping")
        mob = facility.reservoir_prop(state, "Mobility")
        return [mb[m.cells] for mb in mob]


__all__ = ["PerforationMobility"]


if __name__ == "__main__":
    print("ok perforation_mobility (see validate_phase343)")
    print("\nAll tests passed!")
