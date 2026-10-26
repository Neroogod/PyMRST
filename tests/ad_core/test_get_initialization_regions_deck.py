"""EQLNUM/SATNUM/PVTNUM equilibrium regions from a deck (F0.4 risk list).

This function partitions the grid, and a partition has two properties worth
more than any tolerance: every cell lands in exactly one region, and no cell
lands in none. A cell in two regions is equilibrated twice and keeps whichever
result was scattered last; a cell in none keeps whatever the state was
initialised to. Neither raises, and both produce a field that looks
initialised.

The second trap is the active-cell one, the same shape as the ECLIPSE reader's:
`REGIONS` arrays are written over the *Cartesian* box while the model runs on
active cells, so they must be mapped through `indexMap`. A grid with no
inactive cells cannot exercise that, so the mapping is tested on one that has
them.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry, process_grdecl
from pymrst.modules.ad_core.initialization import get_initialization_regions_deck_ad

# datum depth, datum pressure, OWC, Pc_ow, GOC, Pc_go
EQUIL_ONE = np.array([[2000.0, 200e5, 2050.0, 0.11, 2020.0, 0.22]])
EQUIL_TWO = np.array([[2000.0, 200e5, 2050.0, 0.0, 2020.0, 0.0],
                      [2000.0, 210e5, 2060.0, 0.0, 2030.0, 0.0]])


def _model(grid, n_phases=2, densities=(1000.0, 800.0, 1.2)):
    return SimpleNamespace(
        G=grid, n_phases=n_phases,
        fluid=SimpleNamespace(rho_s=np.array(densities[:n_phases]), swcon=0.1),
        raw_fluid=SimpleNamespace(swcon=0.1))


def _grid(dims=(2, 2, 4)):
    return compute_geometry(cart_grid(list(dims), [100.0, 100.0, 40.0]))


def _grid_with_inactive_cells():
    """2x2x2 with two cells deactivated, so indexMap is not the identity."""
    coord = []
    for j in range(3):
        for i in range(3):
            coord += [float(i), float(j), 0.0, float(i), float(j), 10.0]
    zcorn = []
    for k in range(2):
        for top in (0, 1):
            for _ in range(2):
                for _ in range(2):
                    for _ in range(2):
                        for _ in range(2):
                            zcorn.append(float(k + top))
    actnum = np.ones(8, dtype=np.int64)
    actnum[[1, 5]] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_geometry(process_grdecl(
            {"cartDims": np.array([2, 2, 2]), "COORD": np.array(coord, dtype=float),
             "ZCORN": np.array(zcorn, dtype=float), "ACTNUM": actnum},
            split_disconnected=False))


def _deck(equil, **regions):
    return {"SOLUTION": {"EQUIL": equil}, "REGIONS": dict(regions)}


def _partition_of(regions):
    return np.concatenate([np.asarray(r.cells).ravel() for r in regions])


def test_the_regions_partition_every_cell_exactly_once():
    """The property that makes the result an initialisation rather than a mess.

    All three keywords vary independently, and that is the point rather than
    thoroughness for its own sake: with PVTNUM constant, dropping it from the
    subregion filter produces exactly the same partition, so the test would
    pass against a function that ignores it entirely. Confirmed by making that
    change and watching an earlier version of this test stay green.
    """
    grid = _grid()
    cells = grid.cells.num
    eqlnum = np.ones(cells, dtype=np.int64)
    eqlnum[cells//2:] = 2
    satnum = np.ones(cells, dtype=np.int64)
    satnum[::3] = 2
    pvtnum = np.ones(cells, dtype=np.int64)
    pvtnum[::2] = 2

    regions = get_initialization_regions_deck_ad(
        _model(grid), _deck(EQUIL_TWO, EQLNUM=eqlnum, SATNUM=satnum,
                            PVTNUM=pvtnum))
    covered = _partition_of(regions)
    assert sorted(covered.tolist()) == list(range(cells)), "cells missed or doubled"
    assert len(set(covered.tolist())) == covered.size, "a cell is in two regions"


def test_one_region_per_distinct_combination():
    grid = _grid()
    cells = grid.cells.num
    eqlnum = np.ones(cells, dtype=np.int64)
    eqlnum[cells//2:] = 2
    satnum = np.ones(cells, dtype=np.int64)
    satnum[::2] = 2
    pvtnum = np.ones(cells, dtype=np.int64)
    pvtnum[:4] = 2

    regions = get_initialization_regions_deck_ad(
        _model(grid), _deck(EQUIL_TWO, EQLNUM=eqlnum, SATNUM=satnum, PVTNUM=pvtnum))
    tags = {(r.eqlnum, r.satnum, r.pvtnum) for r in regions}
    expected = {(int(e), int(s), int(p))
                for e, s, p in zip(eqlnum, satnum, pvtnum)}
    assert tags == expected
    assert len(regions) == len(tags), "a combination produced two regions"


def test_region_arrays_are_mapped_through_the_active_cell_index():
    """The active-cell trap. REGIONS is written over the Cartesian box.

    A reader that indexes it directly puts every region boundary in the wrong
    place on any real model, and on a grid with all cells active - which is what
    a Cartesian test grid is - the bug is invisible. So this uses a grid with
    inactive cells and checks the assignment cell by cell.
    """
    grid = _grid_with_inactive_cells()
    index_map = np.asarray(grid.cells.indexMap).ravel()
    assert grid.cells.num < 8 and not np.array_equal(index_map, np.arange(8))

    cartesian_eqlnum = np.array([1, 1, 2, 2, 1, 1, 2, 2], dtype=np.int64)
    regions = get_initialization_regions_deck_ad(
        _model(grid), _deck(EQUIL_TWO, EQLNUM=cartesian_eqlnum))

    expected = cartesian_eqlnum[index_map]
    for region in regions:
        for cell in np.asarray(region.cells).ravel():
            assert expected[cell] == region.eqlnum, (
                f"active cell {cell} (cartesian {index_map[cell]}) went to "
                f"region {region.eqlnum}, deck says {expected[cell]}")
    assert sorted(_partition_of(regions).tolist()) == list(range(grid.cells.num))


def test_an_equil_record_with_no_cells_produces_no_region():
    """An empty region would be equilibrated over an empty cell set and
    scattered back as nothing, which is harmless only until something counts
    the regions.

    This pins the outcome, not the line that produces it. The explicit
    `continue` on an empty cell set turns out to be redundant - an empty set
    also yields an empty array of (SATNUM, PVTNUM) pairs, so the inner loop
    never runs - and removing it leaves this test green. Said plainly here so
    the test is not mistaken for protection of that guard.
    """
    grid = _grid()
    eqlnum = np.ones(grid.cells.num, dtype=np.int64)      # nothing in region 2
    regions = get_initialization_regions_deck_ad(
        _model(grid), _deck(EQUIL_TWO, EQLNUM=eqlnum))
    assert len(regions) == 1
    assert regions[0].eqlnum == 1


def test_missing_regions_keywords_default_to_a_single_region():
    grid = _grid()
    regions = get_initialization_regions_deck_ad(_model(grid), _deck(EQUIL_ONE))
    assert len(regions) == 1
    assert (regions[0].eqlnum, regions[0].satnum, regions[0].pvtnum) == (1, 1, 1)
    assert sorted(np.asarray(regions[0].cells).ravel().tolist()) == \
        list(range(grid.cells.num))


def test_contacts_are_ordered_water_oil_then_gas_oil():
    """**Phase order, not depth order** - and this test used to assert the
    opposite, which is how the bug survived.

    `initialize_equilibrium_pressures_ad` walks the phases by index (water, then
    gas) and takes the next contact for each, so the list is
    `[water-oil, gas-oil]`. MRST passes the same:
    `getInitializationRegionsBlackOil(model, [70, 30])` gives the deeper water
    contact first.

    Shallowest-first anchors water at the gas-oil contact and gas at the
    water-oil contact. Each then floods its whole range and the oil is
    squeezed out entirely - Norne equilibrated with zero oil in all 44431
    cells. The previous version of this test asserted `[goc, owc]` while its
    own docstring described that outcome as "simply wrong", which is why the
    order below is checked against the *pressures it produces* and not only
    against the array.
    """
    grid = _grid()
    owc, pc_ow, goc, pc_go = (EQUIL_ONE[0, 2], EQUIL_ONE[0, 3],
                              EQUIL_ONE[0, 4], EQUIL_ONE[0, 5])

    two = get_initialization_regions_deck_ad(_model(grid, 2), _deck(EQUIL_ONE))[0]
    np.testing.assert_allclose(np.asarray(two.contacts).ravel(), [owc])
    np.testing.assert_allclose(np.asarray(two.contacts_pc).ravel(), [pc_ow])

    three = get_initialization_regions_deck_ad(_model(grid, 3), _deck(EQUIL_ONE))[0]
    np.testing.assert_allclose(np.asarray(three.contacts).ravel(), [owc, goc])
    np.testing.assert_allclose(np.asarray(three.contacts_pc).ravel(),
                               [pc_ow, pc_go])


def test_each_phase_is_anchored_at_its_own_contact():
    """The consequence, measured on a real model rather than assumed.

    Water's phase pressure must cross the oil's *at the water-oil contact*,
    and gas's at the gas-oil contact. That is what "anchored" means, and it is
    the property the ordering exists to produce - so a future reordering fails
    here even if someone updates the array assertion above to match it.

    Swapping them is what left Norne with no oil in any of its 44431 cells.
    """
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
    from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
        get_initialization_regions_black_oil_ad,
    )
    from pymrst.modules.ad_core.initialization.equilibrium import (
        initialize_equilibrium_pressures_ad,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, POISE

    owc, goc = 70.0, 30.0                      # deeper first: water, then gas
    g = compute_geometry(cart_grid([1, 1, 20], [10.0, 10.0, 100.0]))
    rock = make_rock(g, 0.1 * DARCY, 0.3)
    fluid = init_simple_adi_fluid(
        n=[2, 3, 1], rho=[1000.0, 700.0, 10.0],
        mu=np.array([1.0, 5.0, 0.1]) * CENTI * POISE,
        c=np.array([0.0, 1e-6, 1e-4]) / BARSA)
    model = GenericBlackOilModelAD(g, rock, fluid, [], gravity=9.80665)
    region = get_initialization_regions_black_oil_ad(
        model, [owc, goc], datum_pressure=200 * BARSA)

    p = initialize_equilibrium_pressures_ad(model, region, 9.80665)
    z = np.asarray(g.cells.centroids)[:, 2]
    order = np.argsort(z)

    for phase, contact in ((0, owc), (2, goc)):
        # `p_phase - p_oil` rises with depth for water and *falls* for gas,
        # so the crossing is located by the sign change rather than by
        # `np.interp`, which needs an increasing abscissa and silently
        # returns a plausible wrong answer for the gas otherwise.
        difference = (p[:, phase] - p[:, 1])[order]
        depths = z[order]
        sign_change = np.flatnonzero(np.diff(np.sign(difference)) != 0)
        assert sign_change.size == 1, (phase, sign_change)
        k = int(sign_change[0])
        lo, hi = difference[k], difference[k + 1]
        crossing = depths[k] + (depths[k + 1] - depths[k])*(-lo)/(hi - lo)
        assert abs(crossing - contact) < 3.0, (phase, crossing, contact)


def test_the_datum_is_taken_from_the_matching_equil_record():
    grid = _grid()
    cells = grid.cells.num
    eqlnum = np.ones(cells, dtype=np.int64)
    eqlnum[cells//2:] = 2
    regions = get_initialization_regions_deck_ad(
        _model(grid), _deck(EQUIL_TWO, EQLNUM=eqlnum))
    by_tag = {r.eqlnum: r for r in regions}
    for tag, record in ((1, EQUIL_TWO[0]), (2, EQUIL_TWO[1])):
        assert float(by_tag[tag].datum_depth) == pytest.approx(record[0])
        assert float(by_tag[tag].datum_pressure) == pytest.approx(record[1])


def test_a_deck_without_equil_is_refused():
    """Refused either way, but by two different routes - worth pinning both.

    An *empty* EQUIL reaches the module's own guard and gets its message. An
    *absent* one does not: `_section` returns `{}`, and `np.asarray({},
    dtype=float)` raises `TypeError` on the line before, so the intended
    "deck SOLUTION.EQUIL is required" never fires for the case a user is most
    likely to hit. That is a message defect rather than a correctness one - the
    deck is still rejected - and changing the exception type would change
    behaviour callers can already see, so it is recorded rather than repaired.
    """
    grid = _grid()
    for empty in ([], np.zeros((0, 6))):
        with pytest.raises(ValueError, match="EQUIL"):
            get_initialization_regions_deck_ad(
                _model(grid), {"SOLUTION": {"EQUIL": empty}})

    for absent in ({"SOLUTION": {}}, {}):
        with pytest.raises((TypeError, ValueError)):
            get_initialization_regions_deck_ad(_model(grid), absent)


def test_a_fluid_without_surface_densities_is_refused():
    grid = _grid()
    model = _model(grid)
    model.fluid = SimpleNamespace(swcon=0.1)
    with pytest.raises(ValueError, match="surface densities"):
        get_initialization_regions_deck_ad(model, _deck(EQUIL_ONE))


def test_surface_densities_do_not_vary_by_pvt_region():
    """A limitation, pinned so it is documented rather than discovered.

    `_phase_densities(model, pvt_region)` accepts the region and never reads
    it, so a deck whose PVT regions carry different DENSITY records initialises
    every region from the same surface densities. Whether MRST looks these up
    per region is not verifiable here - there is no MATLAB source in this tree -
    so this asserts the behaviour that exists rather than a behaviour claimed
    for MRST, and the gap is recorded in the ledger for an owner to resolve.
    """
    grid = _grid()
    cells = grid.cells.num
    pvtnum = np.ones(cells, dtype=np.int64)
    pvtnum[cells//2:] = 2

    regions = get_initialization_regions_deck_ad(
        _model(grid), _deck(EQUIL_ONE, PVTNUM=pvtnum))
    assert {r.pvtnum for r in regions} == {1, 2}
    densities = [tuple(np.asarray(r.rho, dtype=float).ravel()) for r in regions]
    assert densities[0] == densities[1], (
        "per-PVT-region densities appear to have been implemented - update this "
        "test and the ledger entry rather than deleting the check")
