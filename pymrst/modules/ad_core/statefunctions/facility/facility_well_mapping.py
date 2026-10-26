"""FacilityWellMapping (MRST ad-core/statefunctions/facility).

Ports MRST's `FacilityWellMapping.m`: the per-perforation bookkeeping every
other facility state function reads - which cells the active wells perforate,
which well each perforation belongs to, which wells are injectors, and the
sparse matrix that sums perforation quantities back to wells.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.statefunctions.facility.facility_flow_discretization import (  # noqa: E501
    build_well_mapping,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class FacilityWellMapping(StateFunction):
    """Perforation<->well maps for the active wells."""

    def __init__(self, facility: Any = None):
        super().__init__(facility, label="W\\rightarrow c")

    def evaluate_on_domain(self, facility: Any, state: Any):
        return build_well_mapping(facility)


__all__ = ["FacilityWellMapping"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
    )
    from pymrst.modules.ad_core.statefunctions.facility.facility_flow_discretization import (  # noqa: E501
        FacilityContext,
    )
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _facility():
        g = compute_geometry(cart_grid([5, 1, 4], [500.0, 100.0, 40.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        # a 4-perforation vertical injector and a 4-perforation producer
        inj_cells = np.array([0, 5, 10, 15])
        prod_cells = np.array([4, 9, 14, 19])
        w = add_well(None, g, rock, inj_cells, well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, prod_cells, well_type="bhp", val=150 * BARSA,
                     compi=np.array([0.0, 1.0]), sign=-1)
        model = GenericBlackOilModelAD(g, rock, fluid, w)
        return FacilityContext(model, w), inj_cells, prod_cells

    def test_mapping_lists_all_perforations():
        fac, inj_c, prod_c = _facility()
        m = fac.get_props(State(), "FacilityWellMapping")
        assert np.array_equal(m.cells, np.concatenate([inj_c, prod_c]))
        assert np.array_equal(m.perf2well, [0, 0, 0, 0, 1, 1, 1, 1])
        assert list(m.is_injector) == [True, False]
        print("ok test_mapping_lists_all_perforations")

    def test_perforation_sum_aggregates_to_wells():
        """perforation_sum @ per-perf vector gives the per-well totals."""
        fac, _, _ = _facility()
        m = fac.get_props(State(), "FacilityWellMapping")
        per_perf = np.arange(1.0, 9.0)          # 8 perforations
        per_well = m.perforation_sum @ per_perf
        assert np.allclose(per_well, [1 + 2 + 3 + 4, 5 + 6 + 7 + 8])
        print("ok test_perforation_sum_aggregates_to_wells")

    test_mapping_lists_all_perforations()
    test_perforation_sum_aggregates_to_wells()
    print("\nAll tests passed!")
