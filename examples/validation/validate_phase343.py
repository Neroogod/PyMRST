"""Phase 343 (AC7): facility well-flux state functions.

Machinery - proved against the validated procedural well model. PyMRST's
`FacilityModel._single_well` (a port of MRST's
`computeWellContributionsSingleWell`) computes the reservoir volumetric
perforation flux ``cq_vol``; it is exercised by the multi-perforation Phase 49
cross-check and every black-oil well case. AC7 re-expresses that flux as MRST's
facility state-function graph (`FacilityWellMapping` -> `WellIndex` /
`PerforationPressureGradient` / `PerforationMobility` -> `WellPhaseFlux`).

The proof: on two five-perforation wells (one injector, one producer), the
graph's `WellPhaseFlux` must reproduce ``cq_vol`` perforation by perforation,
phase by phase, to round-off - the injector branch (total mobility times
injection composition) and the producer branch (reservoir mobility) both. If it
does, the well-flux graph inherits the validation of the procedural model.

No `.mat`: two Python paths - one procedural, one graph - that must agree.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.statefunctions.facility import FacilityContext
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def build():
    """Two five-perforation vertical wells in a 5x1x5 slab."""
    g = compute_geometry(cart_grid([5, 1, 5], [500.0, 100.0, 50.0]))
    rock = make_rock(g, 150 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
    inj_cells = np.array([0, 5, 10, 15, 20])          # left column
    prod_cells = np.array([4, 9, 14, 19, 24])         # right column
    w = add_well(None, g, rock, inj_cells, well_type="rate", val=3e-3,
                 compi=np.array([1.0, 0.0]), sign=1)
    w = add_well(w, g, rock, prod_cells, well_type="bhp", val=150 * BARSA,
                 compi=np.array([0.0, 1.0]), sign=-1)
    model = GenericBlackOilModel(g, rock, fluid, w, gravity=9.80665)
    # a state where the injector really injects and the producer produces
    st = model.init_state_ad(State(
        pressure=np.full(g.cells.num, 200 * BARSA),
        sW=np.full(g.cells.num, 0.5),
        qWs=np.array([3e-3, 0.0]), qOs=np.array([0.0, -3e-3]),
        bhp=np.array([260 * BARSA, 150 * BARSA])))     # inj high, prod low
    return model, st, w


def procedural_cq_vol(model, st, wells):
    """cq_vol per well from the procedural _single_well, plus the cdp used."""
    fac = model.facility
    mob = model.get_props(st, "Mobility")
    b = model.get_props(st, "ShrinkageFactors")
    rho = model.get_props(st, "Density")
    p = st["pressure"]
    nph = model.n_phases
    q_s = [st["qWs"], st["qOs"]]
    bhp = st["bhp"]
    cq_vol_all, cdp_all = [], []
    for k, w in enumerate(wells):
        cdp = fac.connection_pressure_drop(model, w, value(p), mob, None, nph,
                                           rho=rho, flux=None)
        _, _, _, cq_vol = fac._single_well(
            w, p, mob, b, None, nph, [q[k:k + 1] for q in q_s],
            bhp[k:k + 1], cdp)
        cq_vol_all.append(cq_vol)
        cdp_all.append(np.asarray(cdp, float).ravel())
    return cq_vol_all, np.concatenate(cdp_all)


def graph_well_phase_flux(model, st, wells, cdp_concat):
    fac = FacilityContext(model, wells)
    fstate = State(reservoir_state=st,
                   bhp=value(st["bhp"]),
                   cdp=cdp_concat)
    return fac.get_props(fstate, "WellPhaseFlux")


def main():
    print("Phase 343 (AC7): facility well-flux state functions")
    model, st, wells = build()
    cq_vol_all, cdp = procedural_cq_vol(model, st, wells)
    q_graph = graph_well_phase_flux(model, st, wells, cdp)

    nph = model.n_phases
    # concatenate procedural cq_vol across wells to match the graph's perf order
    worst = scale = 0.0
    offset = 0
    for k, cq_vol in enumerate(cq_vol_all):
        n_perf = np.asarray(wells[k].cells).size
        for i in range(nph):
            proc = np.asarray(value(cq_vol[i]), float).ravel()
            graph = np.asarray(value(q_graph[i]), float).ravel()[offset:offset + n_perf]
            worst = max(worst, np.max(np.abs(proc - graph)))
            scale = max(scale, np.max(np.abs(proc)))
        offset += n_perf
    print(f"  WellPhaseFlux vs procedural cq_vol: {worst:.2e} "
          f"(rate scale {scale:.2e})")
    assert worst < 1e-12 * scale
    print("Phase 343 passed - the facility well-flux graph reproduces the "
          "validated procedural perforation fluxes (injector and producer).")


if __name__ == "__main__":
    main()
