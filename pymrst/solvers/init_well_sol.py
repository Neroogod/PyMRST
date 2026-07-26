"""Initialise a well-solution array (MRST `initWellSol`).

Creates the per-well solution structure a flow solver fills in: a zero connection
flux for each of the well's perforations and a starting bottom-hole pressure of
`p0`. It is the well-side companion to `initResSol` - the blank slate the first
solve writes into.
"""

from typing import List, Sequence

import numpy as np

from pymrst.solvers.state_types import WellSol


def init_well_sol(wells: Sequence, p0: float) -> List[WellSol]:
    """Return one `WellSol` per well: zero perforation fluxes, pressure `p0`."""
    return [WellSol(flux=np.zeros(len(np.atleast_1d(w.cells))), pressure=float(p0))
            for w in wells]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA

    def _wells():
        g = compute_geometry(cart_grid([4, 4, 1], [4.0, 4.0, 1.0]))
        rock = make_rock(g, 1e-13, 0.2)
        w = add_well(None, g, rock, np.array([0, 1, 2]), well_type="bhp",
                     val=200 * BARSA, compi=np.array([1.0, 0.0]), name="I")
        w = add_well(w, g, rock, np.array([15]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([1.0, 0.0]), name="P")
        return w

    def test_one_wellsol_per_well_with_p0():
        ws = init_well_sol(_wells(), 175 * BARSA)
        assert len(ws) == 2
        assert all(w.pressure == 175 * BARSA for w in ws)
        print("ok test_one_wellsol_per_well_with_p0")

    def test_flux_is_zero_per_perforation():
        wells = _wells()
        ws = init_well_sol(wells, 200 * BARSA)
        assert ws[0].flux.shape == (3,) and np.all(ws[0].flux == 0.0)
        assert ws[1].flux.shape == (1,)
        print("ok test_flux_is_zero_per_perforation")

    def test_empty_well_list():
        assert init_well_sol([], 1e5) == []
        print("ok test_empty_well_list")

    test_one_wellsol_per_well_with_p0()
    test_flux_is_zero_per_perforation()
    test_empty_well_list()
    print("\nAll tests passed!")
