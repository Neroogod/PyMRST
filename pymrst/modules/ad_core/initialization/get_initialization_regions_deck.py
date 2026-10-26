"""MRST ``ad-core/initialization/getInitializationRegionsDeck.m``.

The equilibration regions of a deck: one per EQUIL record (its cells from
EQLNUM, or all), split further by the combinations of SATNUM and PVTNUM
present. Each takes the record's datum depth and pressure, its water-oil
and gas-oil contacts and capillary pressures there (only those of active
phase pairs; for a water-gas model the first), and, with dissolved gas, an
`rs(p, z)`: the saturated value at the datum pressure when the record's RS
method is 0 or less, else interpolated in depth from RSVD (or from PBVD
through `rsSat`), linearly and constant beyond the table. `rv(p, z)` comes
likewise from the gas-oil contact pressure or RVVD. The regions are
`getInitializationRegionsBlackOil`'s; compositional models (with an EOS) are
not ported and say so.

`get_initialization_regions_deck_ad`, below, is PyMRST's earlier version.
"""

from __future__ import annotations

import numpy as np

from pymrst.utils.matlab_interp1 import matlab_interp1

from pymrst.modules.ad_core.initialization.regions import (
    get_initialization_regions_base_ad,
)


def get_initialization_regions_deck(model, deck):
    """MRST `regions = getInitializationRegionsDeck(model, deck)`."""
    solution = deck["SOLUTION"]
    deck_regions = deck.get("REGIONS", {})
    equil = np.atleast_2d(np.asarray(solution["EQUIL"], dtype=float))
    n = equil.shape[0]
    from pymrst.modules.ad_core.ad_oo.wrapper_model import value_copy_model
    model = value_copy_model(model).validate_model()
    imap = np.asarray(model.G.cells.indexMap, dtype=np.int64).ravel()
    regions = []
    for region_ix in range(1, n + 1):
        if "EQLNUM" in deck_regions:
            cells = np.flatnonzero(np.asarray(deck_regions["EQLNUM"]).ravel()[imap] == region_ix)
        else:
            cells = np.arange(model.G.cells.num)
        eql = equil[region_ix - 1, :]
        sat = _region_numbers(model, deck_regions, "SATNUM", cells)
        pvt = _region_numbers(model, deck_regions, "PVTNUM", cells)
        pairs = np.unique(np.column_stack([sat, pvt]), axis=0)
        for satnum, pvtnum in pairs:
            subcells = cells[(sat == satnum) & (pvt == pvtnum)]
            regions.append(_get_region(model, deck, eql, subcells, region_ix, int(satnum),
                                       int(pvtnum)))
    return regions


getInitializationRegionsDeck = get_initialization_regions_deck


def _region_numbers(model, deck_regions, name, cells):
    if name in deck_regions:
        imap = np.asarray(model.G.cells.indexMap, dtype=np.int64).ravel()
        return np.asarray(deck_regions[name], dtype=np.int64).ravel()[imap[cells]]
    return np.ones(cells.size, dtype=np.int64)


def _get_region(model, deck, eql, cells, region_ix, satnum, pvtnum):
    from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
        get_initialization_regions_black_oil,
    )
    rs_method, rv_method = eql[6], eql[7]
    contacts = eql[[2, 4]]
    contacts_pc = eql[[3, 5]]
    p_datum = eql[1]
    if model.water and model.gas and not model.oil:
        act = np.array([True, False])
    else:
        act = np.array([model.water and model.oil, model.oil and model.gas])
    has_vapoil = bool(getattr(model, "vapoil", False))
    has_disgas = bool(getattr(model, "disgas", False))
    rs = rv = None
    fluid = model.fluid
    solution = deck["SOLUTION"]
    if has_disgas:
        rsSat = _regional_function(fluid, "rsSat", pvtnum)
        if rs_method <= 0:
            rs_datum = float(np.ravel(rsSat(np.array([p_datum])))[0])
            rs = lambda p, z, v=rs_datum: 0*np.asarray(p, dtype=float) + v
        else:
            if "RSVD" in solution:
                rsvd = np.asarray(solution["RSVD"][region_ix - 1], dtype=float)
            else:
                assert "PBVD" in solution
                pbvd = np.asarray(solution["PBVD"][region_ix - 1], dtype=float)
                rsvd = np.column_stack([pbvd[:, 0], np.ravel(rsSat(pbvd[:, 1]))])
            rs = lambda p, z, t=rsvd: _gridded_linear_nearest(t[:, 0], t[:, 1], z)
    if has_vapoil:
        rvSat = _regional_function(fluid, "rvSat", pvtnum)
        if rv_method <= 0:
            pg_goc = p_datum + eql[5]
            rv_goc = float(np.ravel(rvSat(np.array([pg_goc])))[0])
            rv = lambda p, z, v=rv_goc: 0*np.asarray(p, dtype=float) + v
        else:
            assert "RVVD" in solution
            rvvd = np.asarray(solution["RVVD"][region_ix - 1], dtype=float)
            rv = lambda p, z, t=rvvd: _gridded_linear_nearest(t[:, 0], t[:, 1], z)
    if hasattr(model, "EOSModel"):
        raise NotImplementedError(
            "getInitializationRegionsCompositional (EOS models) is not ported")
    region = get_initialization_regions_black_oil(
        model, contacts[act], cells=cells, datum_pressure=p_datum, datum_depth=eql[0],
        contacts_pc=contacts_pc[act], rs=rs, rv=rv)
    if has_disgas:
        region.rs = rs
    if has_vapoil:
        region.rv = rv
    return region


def _regional_function(fluid, name, reg):
    fn = fluid.get(name) if isinstance(fluid, dict) else getattr(fluid, name, None)
    return fn[reg - 1] if isinstance(fn, (list, tuple)) else fn


def _gridded_linear_nearest(x, v, xq):
    """`griddedInterpolant(x, v, 'linear', 'nearest')` at `xq`."""
    xq = np.asarray(xq, dtype=float)
    return matlab_interp1(x, v, xq, "nearest")


def _section(mapping, name):
    for key in (name, name.upper(), name.lower()):
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
    return {}


def _flag(model, name):
    return bool(getattr(model, name, False))


def _phase_densities(model, pvt_region):
    rho = np.asarray(
        getattr(model.fluid, "rho_s", getattr(model.fluid, "rhoS", [])),
        dtype=float,
    ).ravel()
    if rho.size == 0:
        raise ValueError("model fluid must expose surface densities")
    return [float(value) for value in rho[: model.n_phases]]


def get_initialization_regions_deck_ad(model, deck):
    """Build one equilibrium subregion per EQLNUM/SATNUM/PVTNUM combination."""
    solution = _section(deck, "SOLUTION")
    regions_data = _section(deck, "REGIONS")
    equil = np.atleast_2d(np.asarray(_section(solution, "EQUIL"), dtype=float))
    if equil.size == 0:
        raise ValueError("deck SOLUTION.EQUIL is required")
    nc = model.G.cells.num
    index_map = np.asarray(
        getattr(model.G.cells, "indexMap", np.arange(nc)), dtype=np.int64
    ).ravel()

    def region_array(name, default):
        raw = _section(regions_data, name)
        if isinstance(raw, dict) and not raw:
            return np.full(nc, default, dtype=np.int64)
        values = np.asarray(raw, dtype=np.int64).ravel()
        return values[index_map]

    eqlnum = region_array("EQLNUM", 1)
    satnum = region_array("SATNUM", 1)
    pvtnum = region_array("PVTNUM", 1)
    output = []
    for eql_index, record in enumerate(equil, 1):
        cells = np.flatnonzero(eqlnum == eql_index)
        if not cells.size:
            continue
        pairs = np.unique(np.column_stack([satnum[cells], pvtnum[cells]]), axis=0)
        for sat_region, pvt_region in pairs:
            subcells = cells[
                (satnum[cells] == sat_region) & (pvtnum[cells] == pvt_region)
            ]
            owc = float(record[2])
            goc = float(record[4]) if record.size > 4 else owc
            pc_ow = float(record[3]) if record.size > 3 else 0.0
            pc_go = float(record[5]) if record.size > 5 else 0.0
            # **Phase order, not depth order.** `initialize_equilibrium_pressures_ad`
            # walks the phases by index - water, then gas - and takes the next
            # contact for each, so the list is `[water-oil, gas-oil]`. That is
            # also what MRST passes: `getInitializationRegionsBlackOil(model,
            # [70, 30])` in `blackoilTutorialMexAcceleration` gives the deeper
            # water contact first.
            #
            # Supplying them shallowest-first anchors *water at the gas-oil
            # contact and gas at the water-oil contact*. Each phase then floods
            # its whole range and the oil is squeezed out completely - Norne
            # equilibrated with zero oil in all 44431 cells, and the run failed
            # on a singular Jacobian because the undersaturated cells' `rs`
            # multiplied a zero oil saturation.
            contacts = [owc, goc] if model.n_phases > 2 else [owc]
            contacts_pc = [pc_ow, pc_go] if model.n_phases > 2 else [pc_ow]
            # The deck's capillary curves, for *this* saturation region.
            # Without them the fill places a sharp contact and every cell above
            # the water-oil contact gets connate water - where a real SWOF's
            # `Pcow` column implies a transition zone that can be tens of
            # metres thick.
            from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
                capillary_functions,
            )
            fluid = getattr(model, "raw_fluid", None) or getattr(
                model, "fluid", None)
            phase_names = getattr(model, "get_phase_names", None)
            # A model that cannot name its phases cannot say which curve goes
            # where, so the fill keeps its sharp contact rather than guessing.
            names = ([str(n).upper() for n in phase_names()]
                     if callable(phase_names) else [])
            region = get_initialization_regions_base_ad(
                model,
                _phase_densities(model, int(pvt_region)),
                contacts,
                cells=subcells,
                datum_depth=float(record[0]),
                datum_pressure=float(record[1]),
                contacts_pc=contacts_pc,
                pc_functions=(capillary_functions(fluid, names, model,
                                                  int(sat_region))
                              if names and fluid is not None else None),
            )
            region.satnum = int(sat_region)
            region.pvtnum = int(pvt_region)
            region.eqlnum = eql_index
            output.append(region)
    return output



__all__ = ["getInitializationRegionsDeck", "get_initialization_regions_deck",
           "get_initialization_regions_deck_ad"]


if __name__ == "__main__":
    def _model():
        from pymrst.gridprocessing import cart_grid, compute_geometry
        from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
        from pymrst.modules.ad_props import init_simple_adi_fluid
        from pymrst.params.rock import make_rock
        G = compute_geometry(cart_grid([6, 1]))
        fluid = init_simple_adi_fluid(phases="WO", n=[2, 2], mu=[1, 1], rho=[1000, 800])
        return GenericBlackOilModel(G, make_rock(G, 1, 0.2), fluid, gas=False).validate_model()

    def test_one_region_per_record_and_saturation_table():
        deck = {"SOLUTION": {"EQUIL": np.array([[1000, 1e7, 1100, 5, 900, 0, 0, 0, 0, 0],
                                               [1200, 1.2e7, 1300, 7, 1000, 0, 0, 0, 0, 0]])},
                "REGIONS": {"EQLNUM": np.array([1, 1, 1, 2, 2, 2]),
                            "SATNUM": np.array([1, 1, 2, 1, 1, 1])}}
        regions = get_initialization_regions_deck(_model(), deck)
        assert len(regions) == 3
        np.testing.assert_array_equal(np.sort(np.concatenate([r.cells for r in regions])),
                                      np.arange(6))
        assert regions[-1].datum_pressure == 1.2e7 and regions[0].datum_depth == 1000
        print("ok test_one_region_per_record_and_saturation_table")

    def test_depth_tables_hold_their_end_values():
        out = _gridded_linear_nearest(np.array([0.0, 10.0]), np.array([1.0, 3.0]),
                                      np.array([-5.0, 5.0, 20.0]))
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])
        print("ok test_depth_tables_hold_their_end_values")

    test_one_region_per_record_and_saturation_table()
    test_depth_tables_hold_their_end_values()
    print("\nAll tests passed!")
