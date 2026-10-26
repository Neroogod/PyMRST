"""MRST ``ad-core/initialization/initStateBlackOilAD.m``.

A black-oil state from equilibration regions: in each region's cells the
phase pressures are equilibrated (`initializeEquilibriumPressures`, or
taken from the `pressure` option) and the saturations follow from them
(`initializeEquilibriumSaturations`). The state's pressure is the reference
phase's, except in cells held mostly by gas or water where the reference
phase cannot flow (relative permeability below 1e-8): there it is that
phase's pressure minus its capillary pressure. With dissolved gas `rs` is
the region's `rs(p_o, z)` capped at saturation (saturated where there is
gas); vaporised oil likewise with `rv`. Returns `(state, pressures)`; a
region overlap is an error and incomplete coverage a warning. MRST's
vaporised-oil branch reads the oil pressure set by the dissolved-gas branch
(so a model with `vapoil` but not `disgas` fails there); that is kept.

`init_state_black_oil_ad_ad`, below, is PyMRST's earlier version, which
returns only the state. Its notes follow.


`getInitializationRegionsBlackOil` says *where* the contacts are and what the
phase densities do with depth; this integrates the hydrostatic column against
them and assembles the state. The two are always used together, which is why
`simulatorWorkflowExample` writes::

    region = getInitializationRegionsBlackOil(model, 5, 'datum_pressure', p_ref);
    state0 = initStateBlackOilAD(model, region);

The name was previously aliased to `init_state_deck_ad`, which is a different
function taking a *deck* rather than regions. Calling it as MRST calls it -
with a region - either raised or, worse, found an `EQUIL` attribute on the
region and equilibrated against the wrong thing. The two initialisation paths
are genuinely distinct: `init_state_deck_ad` reads a deck's `EQUIL` record, and
this takes regions a caller has already built, which is what lets an example
place a contact without writing a deck.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.initialization.regions import (
    init_state_equil_regions,
)

#: MRST's `gravity reset on`.
NORM_GRAVITY = 9.80665


def init_state_black_oil_ad(model: Any, regions: Any, *, pressure: Any = None):
    """MRST `[state, pressures] = initStateBlackOilAD(model, regions, ...)`."""
    import warnings

    from pymrst.modules.ad_core.ad_oo.physical_model import State
    from pymrst.modules.ad_core.initialization.initialize_equilibrium_pressures import (
        initialize_equilibrium_pressures,
    )
    from pymrst.modules.ad_core.initialization.initialize_equilibrium_saturations import (
        initialize_equilibrium_saturations,
    )
    from pymrst.utils.adi import value
    from pymrst.modules.ad_core.ad_oo.wrapper_model import value_copy_model
    model = value_copy_model(model).validate_model()
    if not isinstance(regions, (list, tuple)):
        regions = [regions]
    vapoil = bool(getattr(model, "vapoil", False))
    disgas = bool(getattr(model, "disgas", False))
    if hasattr(model, "EOSModel"):
        raise NotImplementedError("compositional initialisation is not ported")
    G = model.G
    nc = G.cells.num
    nph = int(np.sum(model.get_active_phases()))
    state = {"pressure": np.zeros(nc), "s": np.zeros((nc, nph))}
    if disgas:
        state["rs"] = np.zeros(nc)
    if vapoil:
        state["rv"] = np.zeros(nc)
    watIx = model.get_phase_index("W") if model.water else None
    oilIx = model.get_phase_index("O") if model.oil else None
    gasIx = model.get_phase_index("G") if model.gas else None
    refIx = oilIx if model.oil else (gasIx if model.gas else 0)
    pressures = np.zeros((nc, nph))
    touched = np.zeros(nc, dtype=bool)
    for region in regions:
        if isinstance(region.cells, str):
            assert len(regions) == 1
            region.cells = np.arange(nc)
        cells = np.asarray(region.cells, dtype=np.int64)
        assert not np.any(touched[cells]), "Multiple regions defined in same cells."
        touched[cells] = True
        if pressure is None:
            p = initialize_equilibrium_pressures(model, region)
        else:
            p = np.atleast_2d(np.asarray(pressure, dtype=float))[cells, :]
        z = np.asarray(G.cells.centroids, dtype=float)[cells, 2]
        s = initialize_equilibrium_saturations(model, region, p)
        state["s"][cells, :] = s
        pressures[cells, :] = p
        pc = [region.pc_sign[i]*np.asarray(region.pc_functions[i](state["s"][cells, i]))
              for i in range(nph)]
        from copy import copy
        m = copy(model)
        m.FlowPropertyFunctions = model.FlowPropertyFunctions.subset(cells)
        tmp_state = State({"s": s, "pressure": p[:, 0]})
        tmp_state = model.init_state_function_containers(tmp_state)
        try:
            kr = np.column_stack([np.asarray(value(k), dtype=float)
                                  for k in m.get_prop(tmp_state, "RelativePermeability")])
        except Exception:
            warnings.warn("Unable to evaluate relative permeability during initialization. "
                          "The phase pressures might be off where saturations do not "
                          "correspond to mobile phases")
            kr = s
        maxSat = s.max(axis=1)
        reference_immobile = kr[:, refIx] < 1e-8
        to_reference = np.ones(p.shape[0], dtype=bool)
        if model.gas:
            gas_majority = s[:, gasIx] == maxSat
            only_gas = gas_majority & reference_immobile
            to_reference[only_gas] = False
            state["pressure"][cells[only_gas]] = p[only_gas, gasIx] - pc[gasIx][only_gas]
            if disgas:
                po = p[:, oilIx]
                rsSat = model.fluid["rsSat"] if isinstance(model.fluid, dict) else model.fluid.rsSat
                if isinstance(rsSat, (list, tuple)):
                    rsSat = rsSat[region.pvt_region - 1]
                rsMax = np.asarray(rsSat(po), dtype=float)
                rs = np.asarray(region.rs(po, z), dtype=float)*np.ones(po.shape)
                rs[rs > rsMax] = rsMax[rs > rsMax]
                sg = s[:, gasIx]
                rs[sg > 0] = rsMax[sg > 0]
                state["rs"][cells] = rs
        if model.oil and vapoil:
            pg = p[:, gasIx]
            rvSat = model.fluid["rvSat"] if isinstance(model.fluid, dict) else model.fluid.rvSat
            if isinstance(rvSat, (list, tuple)):
                rvSat = rvSat[region.pvt_region - 1]
            rvMax = np.asarray(rvSat(po), dtype=float)        # MRST: `po` from the disgas branch
            rv = np.asarray(region.rv(pg, z), dtype=float)*np.ones(pg.shape)
            rv[rv > rvMax] = rvMax[rv > rvMax]
            so = s[:, oilIx]
            rv[so > 0] = rvMax[so > 0]
            state["rv"][cells] = rv
        if model.water:
            wat_majority = s[:, watIx] == maxSat
            only_wat = wat_majority & reference_immobile
            to_reference[only_wat] = False
            state["pressure"][cells[only_wat]] = p[only_wat, watIx] - pc[watIx][only_wat]
        state["pressure"][cells[to_reference]] = p[to_reference, refIx]
    if not np.all(touched):
        warnings.warn("Regions did not cover all cells. Model only partially initialized.")
    if hasattr(model, "polymer"):
        state["cp"] = np.zeros(nc)
    if hasattr(model, "surfactant"):
        state["cs"] = np.zeros(nc)
    return state, pressures


initStateBlackOilAD = init_state_black_oil_ad


def init_state_black_oil_ad_ad(model: Any, regions: Any, *,
                            gravity: Optional[float] = None,
                            pressure=None):
    """Equilibrate ``model`` over ``regions`` and return the state.

    ``regions`` may be a single region or a sequence of them, as in MRST -
    where a lone struct and a one-element cell array mean the same thing.
    """
    if not isinstance(regions, (list, tuple)):
        regions = [regions]
    if gravity is None:
        gravity = float(getattr(model, "gravity_magnitude",
                                getattr(model, "gravity", NORM_GRAVITY))
                        or NORM_GRAVITY)

    state = init_state_equil_regions(model, list(regions), float(gravity))
    if pressure is not None:
        # MRST's 'pressure' option overrides the equilibrated column outright.
        state["pressure"] = np.asarray(pressure, dtype=float).ravel().copy()
    return state


__all__ = ["NORM_GRAVITY", "initStateBlackOilAD", "init_state_black_oil_ad",
           "init_state_black_oil_ad_ad"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
    from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
        get_initialization_regions_black_oil_ad,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _model(nz=10, height=100.0):
        g = compute_geometry(cart_grid([1, 1, nz], [10.0, 10.0, height]))
        rock = make_rock(g, 100*MILLI*DARCY, 0.3)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0])*CENTI*POISE, rho=[1000.0, 700.0],
            n=[2, 2], c=[1e-6/BARSA, 1e-6/BARSA], p_ref=300*BARSA)
        # `GenericBlackOilModelAD`, not `TwoPhaseOilWaterModelAD`: the latter
        # takes no `gravity`, and without it there is no column to equilibrate.
        return GenericBlackOilModelAD(g, rock, fluid, [], gravity=NORM_GRAVITY)

    def test_it_takes_regions_not_a_deck():
        """The alias this file corrects pointed at `init_state_deck_ad`, whose
        argument is a deck. A region has no `EQUIL` record."""
        model = _model()
        region = get_initialization_regions_black_oil_ad(
            model, [50.0], datum_pressure=300*BARSA)
        state = init_state_black_oil_ad_ad(model, region)
        assert state["pressure"].size == model.G.cells.num
        assert state["sW"].size == model.G.cells.num
        print("ok test_it_takes_regions_not_a_deck")

    def test_a_lone_region_and_a_one_element_list_agree():
        """MRST wraps a bare struct in a cell array; both must work."""
        model = _model()
        region = get_initialization_regions_black_oil_ad(
            model, [50.0], datum_pressure=300*BARSA)
        one = init_state_black_oil_ad_ad(model, region)
        many = init_state_black_oil_ad_ad(model, [region])
        assert np.allclose(one["pressure"], many["pressure"])
        print("ok test_a_lone_region_and_a_one_element_list_agree")

    def test_the_contact_separates_the_phases():
        """The whole point of equilibrating: water below the contact, oil
        above. A state that ignores the contact is uniformly saturated and
        the run starts from the wrong fluid distribution."""
        model = _model()
        region = get_initialization_regions_black_oil_ad(
            model, [50.0], datum_pressure=300*BARSA)
        state = init_state_black_oil_ad_ad(model, region)
        z = np.asarray(model.G.cells.centroids)[:, 2]
        below, above = z > 50.0, z < 50.0
        assert state["sW"][below].mean() > state["sW"][above].mean(), (
            state["sW"])
        print(f"ok test_the_contact_separates_the_phases "
              f"(sW {state['sW'][above].mean():.2f} above, "
              f"{state['sW'][below].mean():.2f} below)")

    def test_pressure_increases_with_depth():
        model = _model()
        region = get_initialization_regions_black_oil_ad(
            model, [50.0], datum_pressure=300*BARSA)
        p = np.asarray(init_state_black_oil_ad_ad(model, region)["pressure"])
        z = np.asarray(model.G.cells.centroids)[:, 2]
        assert np.all(np.diff(p[np.argsort(z)]) > 0.0)
        print("ok test_pressure_increases_with_depth")

    def test_an_explicit_pressure_overrides_the_column():
        model = _model()
        region = get_initialization_regions_black_oil_ad(
            model, [50.0], datum_pressure=300*BARSA)
        forced = np.full(model.G.cells.num, 250*BARSA)
        state = init_state_black_oil_ad_ad(model, region, pressure=forced)
        assert np.allclose(state["pressure"], 250*BARSA)
        print("ok test_an_explicit_pressure_overrides_the_column")

    def test_the_mrst_spelling_resolves_to_this_and_not_to_the_deck_reader():
        from pymrst.modules.ad_core.initialization.deck import init_state_deck_ad
        assert initStateBlackOilAD is init_state_black_oil_ad
        assert initStateBlackOilAD is not init_state_deck_ad
        print("ok test_the_mrst_spelling_resolves_to_this_and_not_to_the_deck_reader")

    test_it_takes_regions_not_a_deck()
    test_a_lone_region_and_a_one_element_list_agree()
    test_the_contact_separates_the_phases()
    test_pressure_increases_with_depth()
    test_an_explicit_pressure_overrides_the_column()
    test_the_mrst_spelling_resolves_to_this_and_not_to_the_deck_reader()
    print("\nAll tests passed!")
