"""MRST ``ad-core/statefunctions/facility/FacilityWellMapping.m``.

The bookkeeping every facility property shares: which wells are active, which
cells they perforate, which well each perforation belongs to, which wells
inject, and a sparse sum from perforations to wells. Indices are 0-based.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.modules.ad_core.utils.get_perforation_to_well_mapping import (
    get_perforation_to_well_mapping,
)


class FacilityWellMapping(StateFunction):
    """`W -> c`: the perforation/well maps for the active wells."""

    def __init__(self, model: Any):
        super().__init__(model, label="W\\rightarrow c")
        self.AutoDiffBackend = getattr(model, "AutoDiffBackend", None)

    def evaluate_on_domain(self, model: Any, state: Any) -> SimpleNamespace:
        well_sol = state["wellSol"]
        active = model.get_indices_of_active_wells(well_sol)
        wc, _ = model.get_active_well_cells(well_sol)
        W = model.get_well_struct(active)
        p2w = (np.zeros(0, dtype=np.int64) if not W else
               np.asarray(get_perforation_to_well_mapping(W)[0], dtype=np.int64))
        is_inj = (np.zeros(0, dtype=bool) if not W else
                  np.array([w.sign > 0 for w in W]))
        n = p2w.size
        wsum = sp.csr_matrix((np.ones(n), (p2w, np.arange(n))),
                             shape=(len(W), n))
        return SimpleNamespace(active=active, cells=wc, perf2well=p2w,
                               isInjector=is_inj, perforationSum=wsum, W=W)

    evaluateOnDomain = evaluate_on_domain


__all__ = ["FacilityWellMapping"]


if __name__ == "__main__":
    def _facility():
        W = [SimpleNamespace(cells=np.array([0, 1]), sign=1),
             SimpleNamespace(cells=np.array([4]), sign=-1)]
        return SimpleNamespace(
            get_indices_of_active_wells=lambda ws: np.array([0, 1]),
            get_active_well_cells=lambda ws: (np.array([0, 1, 4]), None),
            get_well_struct=lambda subs: [W[i] for i in subs])

    def test_perforations_map_onto_their_wells():
        m = FacilityWellMapping(_facility()).evaluate_on_domain(
            _facility(), {"wellSol": [None, None]})
        np.testing.assert_array_equal(m.perf2well, [0, 0, 1])
        np.testing.assert_array_equal(m.isInjector, [True, False])
        np.testing.assert_allclose(m.perforationSum @ np.array([1.0, 2.0, 3.0]),
                                   [3.0, 3.0])
        print("ok test_perforations_map_onto_their_wells")

    test_perforations_map_onto_their_wells()
    print("\nAll tests passed!")
