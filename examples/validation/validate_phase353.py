"""Phase 353 (AC16): equilibration regions.

Adds the region layer over PyMRST's single-region equilibration (Phase 42):
`get_initialization_regions_base` (build a region with MRST's defaults),
`get_min_max_phase_saturations`, and `init_state_equil_regions` (equilibrate a
multi-region field). Proved against the validated single-region path:

1. **Factory == hand-built.** A region from the factory equilibrates to exactly
   the same state as a hand-built `EquilibriumRegion` (the factory only supplies
   MRST defaults, it does not change the physics).
2. **Partition invariance.** Splitting the grid into two regions with the *same*
   contact reproduces the single-region whole-field equilibrium to round-off -
   so the multi-region assembly is a faithful partition of the validated path.
3. **Regions are independent.** Two regions with different contacts give
   genuinely different fluid columns (one all-oil, one all-water), which is why
   the region layer exists.

No `.mat`: everything is checked against the single-region equilibration that
already carries the Phase 42 validation.
"""

import numpy as np

from pymrst.gridprocessing import compute_geometry, tensor_grid
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.initialization import (
    EquilibriumRegion,
    get_initialization_regions_base,
    get_min_max_phase_saturations,
    init_state_equil,
    init_state_equil_regions,
)
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

G_ACC = 9.80665


def _model(nz=24, top=1000.0, thick=120.0):
    z = np.linspace(top, top + thick, nz + 1)
    g = compute_geometry(tensor_grid(np.array([0.0, 100.0]),
                                     np.array([0.0, 100.0]), z))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[0.0, 0.0], p_ref=200 * BARSA)
    return g, GenericBlackOilModel(g, rock, fluid, [], gravity=G_ACC)


RHO = [1000.0, 800.0]


def main():
    print("Phase 353 (AC16): equilibration regions")
    g, model = _model()
    nc = g.cells.num
    woc, datum, dp = 1055.0, 1000.0, 200 * BARSA

    # (1) factory vs hand-built
    reg = get_initialization_regions_base(model, RHO, [woc], datum_depth=datum,
                                          datum_pressure=dp)
    hand = EquilibriumRegion(datum_depth=datum, datum_pressure=dp, contacts=[woc])
    hand.rho = RHO
    a = init_state_equil(model, reg, G_ACC)
    b = init_state_equil(model, hand, G_ACC)
    wp = np.max(np.abs(a["pressure"] - b["pressure"]))
    ws = np.max(np.abs(a["sW"] - b["sW"]))
    print(f"  factory vs hand-built region: p {wp:.2e}, sW {ws:.2e}")
    assert wp == 0.0 and ws == 0.0

    # (2) partition invariance
    full = init_state_equil(model, reg, G_ACC)
    half = nc // 2
    r1 = get_initialization_regions_base(model, RHO, [woc], datum_depth=datum,
                                         datum_pressure=dp, cells=np.arange(half))
    r2 = get_initialization_regions_base(model, RHO, [woc], datum_depth=datum,
                                         datum_pressure=dp,
                                         cells=np.arange(half, nc))
    multi = init_state_equil_regions(model, [r1, r2], G_ACC)
    dp_max = np.max(np.abs(multi["pressure"] - full["pressure"]))
    ds_max = np.max(np.abs(multi["sW"] - full["sW"]))
    print(f"  two regions (same contact) vs single: p {dp_max:.2e}, "
          f"sW {ds_max:.2e}")
    assert dp_max < 1e-9 and ds_max < 1e-12

    # (3) independent regions
    ro = get_initialization_regions_base(model, RHO, [1200.0], datum_depth=datum,
                                         datum_pressure=dp, cells=np.arange(half))
    rw = get_initialization_regions_base(model, RHO, [900.0], datum_depth=datum,
                                         datum_pressure=dp,
                                         cells=np.arange(half, nc))
    st = init_state_equil_regions(model, [ro, rw], G_ACC)
    print(f"  independent regions: oil-region sW={st['sW'][:half].mean():.2f}, "
          f"water-region sW={st['sW'][half:].mean():.2f}")
    assert st["sW"][:half].mean() < 0.05 and st["sW"][half:].mean() > 0.95

    s_min, s_max = get_min_max_phase_saturations(model)
    assert np.all(s_min <= s_max)
    print("Phase 353 passed - the region layer reproduces the validated "
          "single-region equilibrium and partitions the field faithfully.")


if __name__ == "__main__":
    main()
