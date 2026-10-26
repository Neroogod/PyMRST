"""MRST ``ad-core/statefunctions/facility/WellIndex.m``.

Each open connection's well index (Peaceman connection factor); closed
connections get zero.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class WellIndex(StateFunction):
    """`WI * cstatus` over the active wells' perforations."""

    def __init__(self, model: Any):
        super().__init__(model, label="W_I")
        self.AutoDiffBackend = getattr(model, "AutoDiffBackend", None)
        self.depends_on("FacilityWellMapping")

    def evaluate_on_domain(self, model: Any, state: Any) -> Optional[np.ndarray]:
        mapping = self.get_evaluated_dependencies(state, "FacilityWellMapping")
        W = model.get_well_struct(mapping.active)
        if not W:
            return None
        cstatus = np.concatenate([np.asarray(state["wellSol"][i].cstatus,
                                             dtype=bool).ravel()
                                  for i in mapping.active])
        wi = np.concatenate([np.asarray(w.WI, dtype=float).ravel() for w in W])
        return wi*cstatus

    evaluateOnDomain = evaluate_on_domain


__all__ = ["WellIndex"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_closed_connections_have_no_well_index():
        W = [SimpleNamespace(WI=np.array([2.0, 3.0]))]
        facility = SimpleNamespace(get_well_struct=lambda subs: [W[i] for i in subs])
        prop = WellIndex(facility)
        prop.get_evaluated_dependencies = lambda st, n: SimpleNamespace(active=[0])
        state = {"wellSol": [SimpleNamespace(cstatus=np.array([True, False]))]}
        np.testing.assert_allclose(prop.evaluate_on_domain(facility, state),
                                   [2.0, 0.0])
        print("ok test_closed_connections_have_no_well_index")

    test_closed_connections_have_no_well_index()
    print("\nAll tests passed!")
