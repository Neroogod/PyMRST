"""CO2 trapping inventory (MRST co2lab-ve/utils/inventory).

The headline output of a storage assessment. Injected CO2 does not simply
sit where it was put: it spreads under the caprock, some of it is caught in
structural traps, some is left behind as a residual saturation the water
cannot displace, some dissolves into the brine, and the rest is still free
to migrate - and can eventually leak. How the mass divides between those
categories, and how that division evolves over centuries, is what decides
whether a site is safe.

The seven categories (MRST's massTrappingDistributionVEADI), in increasing
order of risk:

  1. dissolved     in the brine. Gone for good.
  2. residual in a structural trap
  3. residual      outside a trap, left behind by a plume that moved on
  4. residual in the plume (it will be left behind when the plume moves)
  5. free in a structural trap - mobile, but it cannot escape the trap
  6. sub-scale trapped - caught by structure below the grid's resolution
  7. free and mobile - the part that can still migrate, and leak

Categories 2-7 must sum to the free-phase mass in the model; that identity
is checked, not assumed.

The geometry that drives it all is `zt`, the depth of the spill point above
each cell: CO2 lying within `zt` of the caprock is inside a trap, and CO2
above it is not.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from pymrst.modules.co2lab_spillpoint.trapping import TrapAnalysis
from pymrst.modules.co2lab_ve.ve import TopSurfaceGrid, VEFluid, VEState
from pymrst.params.rock import Rock
from pymrst.utils.adi import value

CATEGORIES = (
    "dissolved",
    "residual in trap",
    "residual",
    "residual in plume",
    "free in trap",
    "sub-scale trapped",
    "free and mobile",
)


@dataclass
class Inventory:
    """The mass in each category, in kg."""

    masses: np.ndarray               # (7,) in the order of CATEGORIES
    per_cell: Optional[np.ndarray] = None    # (7, nc), if asked for

    @property
    def total(self) -> float:
        return float(self.masses.sum())

    @property
    def free_phase(self) -> float:
        """Everything not dissolved - i.e. still a separate CO2 phase."""
        return float(self.masses[1:].sum())

    @property
    def mobile(self) -> float:
        """The part that can still migrate, and so can still leak."""
        return float(self.masses[6])

    def as_dict(self) -> dict:
        return dict(zip(CATEGORIES, self.masses))


def trap_depth(ts: TopSurfaceGrid, res: TrapAnalysis) -> np.ndarray:
    """`zt`: how deep the spill point sits below the caprock, per cell.

    Zero outside any trap's catchment, and never more than the column
    thickness. CO2 within zt of the caprock is structurally trapped.
    """
    tr_z = np.concatenate([[0.0], np.asarray(res.trap_z, float).ravel()])
    regions = np.asarray(res.trap_regions, dtype=np.int64)
    # trap_regions is 0 outside any catchment, and tr_z[0] = 0 makes zt = 0
    # there, so the lookup needs no special case
    zt = np.maximum(tr_z[regions] - ts.z, 0.0)
    return np.minimum(zt, ts.H)


def mass_trapping_distribution(
    ts: TopSurfaceGrid,
    state: VEState,
    rock: Rock,
    fluid: VEFluid,
    res: TrapAnalysis,
    dh: Optional[np.ndarray] = None,
    rs: float = 0.0,
    per_cell: bool = False,
) -> Inventory:
    """Split the CO2 in `state` into the seven categories.

    `dh` is the sub-scale trapping height (structure the grid cannot see);
    `rs` is the dissolved CO2 per unit brine, as MRST takes it - a number
    supplied to the inventory, not computed by it.
    """
    z, H, area = ts.z, ts.H, ts.area
    poro = np.asarray(rock.poro, float).ravel()

    sg = np.asarray(value(state.sG), float).ravel()
    sw = 1.0 - sg
    p = np.asarray(value(state.pressure), float).ravel()

    h, h_max = fluid.heights(sg, state.sGmax, H)
    h = np.asarray(value(h), float).ravel()
    h_max = np.asarray(value(h_max), float).ravel()

    srg, srw = fluid.srg, fluid.srw

    # pore AREA, not pore volume: every term below multiplies it by a height
    pv = poro * area
    rho = fluid.rhoGS * np.asarray(value(fluid.b_g(p)), float).ravel()

    zt = trap_depth(ts, res)

    # sub-scale trapping takes its cut off the top of both heights
    if dh is None:
        h_sub = np.zeros_like(h)
        hm_sub = np.zeros_like(h)
    else:
        dh = np.asarray(dh, float).ravel()
        h_sub = np.minimum(dh, h)
        hm_sub = np.minimum(dh, h_max)
    h_eff = h - h_sub
    hm_eff = h_max - hm_sub

    # the part of (h_max - h) that lies inside a trap: residual CO2 the
    # plume left behind but which is caught by structure anyway
    hdift = np.maximum(np.minimum(zt, hm_eff) - np.minimum(zt, h_eff), 0.0)

    struc = np.minimum(zt, h_eff) * pv * rho          # in a trap, still flowing
    plume = rho * h_eff * pv - struc                  # outside a trap, flowing

    res_struc = (struc + hdift * rho * pv) * srg
    free_struc = struc * (1.0 - srg - srw)
    free_res = plume * srg
    free_mov = plume * (1.0 - srw - srg)
    res_trap = np.maximum(hm_eff - np.maximum(zt, h_eff), 0.0) * rho * pv * srg
    res_dis = fluid.rhoGS * pv * (rs * np.asarray(
        value(fluid.b_w(p)), float).ravel() * (sw * H))
    subtrap = (hm_sub * srg + h_sub * (1.0 - srg - srw)) * pv * rho

    cells = np.vstack([res_dis, res_struc, res_trap, free_res,
                       free_struc, subtrap, free_mov])
    cells = np.maximum(cells, 0.0)
    masses = cells.sum(axis=1)

    # the free-phase categories must add up to the CO2 actually in the model
    gas_phase = float(np.sum(pv * rho * (sg * H)))
    got = float(masses[1:].sum())
    if gas_phase > 0 and abs(got - gas_phase) > 1e-3 * gas_phase:
        raise AssertionError(
            f"the inventory does not close: the categories hold {got:.4e} kg "
            f"of free CO2 but the model has {gas_phase:.4e} kg "
            f"(off by {abs(got - gas_phase) / gas_phase:.2%})")

    return Inventory(masses=masses, per_cell=cells if per_cell else None)


def inventory_history(
    ts: TopSurfaceGrid,
    states: Sequence[VEState],
    rock: Rock,
    fluid: VEFluid,
    res: TrapAnalysis,
    dh: Optional[np.ndarray] = None,
    rs: float = 0.0,
) -> np.ndarray:
    """The inventory at every step: (n_steps, 7), in kg."""
    return np.array([
        mass_trapping_distribution(ts, s, rock, fluid, res, dh=dh, rs=rs).masses
        for s in states])


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.co2lab_spillpoint.trapping import trap_analysis
    from pymrst.modules.co2lab_ve.ve import (
        VEModel,
        average_rock,
        top_surface_grid,
    )
    from pymrst.params.gravity_const import NORM_GRAVITY
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import BARSA, DARCY, DAY, MILLI, YEAR

    def _dome_case():
        """A sloping aquifer with two domes, so there is something to trap."""
        nx, ny, nz = 60, 24, 5
        g = cart_grid([nx, ny, nz], [2500.0, 1000.0, 25.0])
        x, y = g.nodes.coords[:, 0], g.nodes.coords[:, 1]
        g.nodes.coords[:, 2] += (
            1000.0 - 0.02 * x
            - 14.0 * np.exp(-(((x - 900.0) / 140.0) ** 2
                              + ((y - 500.0) / 260.0) ** 2))
            - 18.0 * np.exp(-(((x - 1800.0) / 150.0) ** 2
                              + ((y - 500.0) / 260.0) ** 2)))
        g = compute_geometry(g)
        rock = make_rock(g, 300 * MILLI * DARCY, 0.22)
        ts = top_surface_grid(g)
        rock2 = average_rock(rock, ts)

        fluid = VEFluid(rhoWS=1020.0, rhoGS=700.0, mu_w=8e-4, mu_g=6e-5,
                        srw=0.2, srg=0.2, krW_max=1.0, krG_max=1.0,
                        c_water=4.3e-10, c_gas=1e-9, p_ref=1.2e7)
        return ts, rock2, fluid, trap_analysis(ts)

    def _run(ts, rock2, fluid, n_inj=12, n_mig=30):
        """Inject downdip, then shut in and let the plume migrate updip."""
        inj = int(np.argmin(ts.g.cells.centroids[:, 0]
                            + 1e3 * np.abs(ts.g.cells.centroids[:, 1] - 500.0)))
        rate = 0.05e9 / YEAR / fluid.rhoGS      # 0.05 Mt CO2 / year
        model = VEModel(ts, rock2, fluid, well_cells=np.array([inj]),
                        well_rate=rate, well_WI=np.array([2e-11]))

        p0 = 1e5 + fluid.rhoWS * NORM_GRAVITY * ts.z
        st = VEState(pressure=p0, sG=np.zeros(ts.g.cells.num),
                     sGmax=np.zeros(ts.g.cells.num),
                     qGs=np.array([rate]), bhp=np.array([p0[inj] + 5e5]))

        states = [st]
        for k in range(n_inj):
            st, ok, _ = model.step(st, 60.0 * DAY)
            assert ok, f"injection step {k} failed"
            states.append(st)

        quiet = VEModel(ts, rock2, fluid)
        st = VEState(pressure=st.pressure, sG=st.sG, sGmax=st.sGmax,
                     qGs=np.zeros(0), bhp=np.zeros(0))
        # 1-year steps: the VE model's Newton has no timestep cutting, and
        # 2-year steps do not converge from this state
        for k in range(n_mig):
            st, ok, _ = quiet.step(st, 1.0 * YEAR)
            assert ok, f"migration step {k} failed"
            states.append(st)
        return states, n_inj

    def test_inventory_closes_and_totals_the_injected_mass():
        """The categories must account for every kilogram in the model."""
        ts, rock2, fluid, res = _dome_case()
        assert res.n_traps >= 1, "the dome should give at least one trap"
        states, _ = _run(ts, rock2, fluid)

        inv = mass_trapping_distribution(ts, states[-1], rock2, fluid, res)

        # mass_trapping_distribution raises if the categories do not close, so
        # reaching here already proves that. Check it against the model's own
        # accounting too, independently.
        #
        # Not to round-off, though. The seven categories sum algebraically to
        # h (1 - srg - srw) + h_max srg, which is exactly H sG - *unless*
        # h_max is clipped at the column thickness, as it is in the fully
        # saturated columns around the injector. That clipping breaks the
        # identity by a little, which is why MRST's own closure check allows
        # a 1e-3 mismatch rather than demanding zero.
        pv_area = np.asarray(rock2.poro, float).ravel() * ts.area
        rho = fluid.rhoGS * fluid.b_g(states[-1].pressure)
        in_model = float(np.sum(pv_area * rho * states[-1].sG * ts.H))
        gap = abs(inv.free_phase - in_model) / in_model
        assert gap < 1e-5, f"the inventory is off by {gap:.2e}"
        assert inv.masses[0] == 0.0, "no dissolution was asked for"
        print(f"ok test_inventory_closes_and_totals_the_injected_mass "
              f"({inv.free_phase / 1e9:.3f} Mt of free-phase CO2, "
              f"closing to {gap:.1e})")

    def test_migration_moves_mass_from_mobile_to_trapped():
        """The point of the whole exercise: CO2 gets safer as it migrates.

        Right after injection most of the plume is free and mobile. As it
        runs updip it leaves residual CO2 behind and fills the dome, so the
        mobile fraction must fall and the trapped fractions must rise.
        """
        ts, rock2, fluid, res = _dome_case()
        states, n_inj = _run(ts, rock2, fluid)

        end_inj = mass_trapping_distribution(ts, states[n_inj], rock2, fluid, res)
        final = mass_trapping_distribution(ts, states[-1], rock2, fluid, res)

        mobile_before = end_inj.mobile / end_inj.free_phase
        mobile_after = final.mobile / final.free_phase
        assert mobile_after < mobile_before, (
            f"migration should immobilise CO2 "
            f"({mobile_before:.1%} -> {mobile_after:.1%})")

        # residual trapping outside the traps must have grown: the plume has
        # swept new ground and left CO2 behind in it
        assert final.masses[2] > end_inj.masses[2], "no residual was left behind"
        # and the dome must have caught something
        assert (final.masses[1] + final.masses[4]) > 0.0, "the trap caught nothing"

        print(f"ok test_migration_moves_mass_from_mobile_to_trapped "
              f"(mobile {mobile_before:.0%} of the plume at the end of "
              f"injection, {mobile_after:.0%} after 30 years)")

    def test_categories_respond_to_the_physics_that_drives_them():
        """Each knob must move the category it is supposed to move."""
        ts, rock2, fluid, res = _dome_case()
        states, _ = _run(ts, rock2, fluid)
        st = states[-1]

        base = mass_trapping_distribution(ts, st, rock2, fluid, res)

        # no residual saturation -> nothing can be residually trapped
        dry = VEFluid(**{**fluid.__dict__, "srg": 0.0})
        no_res = mass_trapping_distribution(ts, st, rock2, dry, res)
        assert np.allclose(no_res.masses[[1, 2, 3]], 0.0), \
            "with srg = 0 there can be no residual trapping"
        assert no_res.mobile > base.mobile, \
            "removing residual trapping must leave more CO2 mobile"

        # no traps -> nothing can be structurally trapped
        flat = TrapAnalysis(traps=np.zeros(ts.g.cells.num, dtype=np.int64),
                            trap_z=np.zeros(0),
                            trap_regions=np.zeros(ts.g.cells.num, dtype=np.int64),
                            trap_adj=np.zeros((0, 0)),
                            top=np.zeros(ts.g.cells.num, dtype=np.int64),
                            fill_z=ts.z.copy())
        untrapped = mass_trapping_distribution(ts, st, rock2, fluid, flat)
        assert np.allclose(untrapped.masses[[1, 4]], 0.0), \
            "with no traps there can be no structural trapping"
        assert untrapped.mobile > base.mobile

        # dissolution adds to category 1 and to nothing else
        wet = mass_trapping_distribution(ts, st, rock2, fluid, res, rs=0.05)
        assert wet.masses[0] > 0.0
        assert np.allclose(wet.masses[1:], base.masses[1:]), \
            "dissolution must not disturb the free-phase split"
        print(f"ok test_categories_respond_to_the_physics_that_drives_them "
              f"(srg = 0 leaves {no_res.mobile / no_res.free_phase:.0%} mobile "
              f"vs {base.mobile / base.free_phase:.0%})")

    def test_subscale_trapping_takes_from_the_mobile_part():
        """Sub-grid structure catches CO2 the grid cannot see."""
        ts, rock2, fluid, res = _dome_case()
        states, _ = _run(ts, rock2, fluid)
        st = states[-1]

        base = mass_trapping_distribution(ts, st, rock2, fluid, res)
        rough = mass_trapping_distribution(
            ts, st, rock2, fluid, res,
            dh=np.full(ts.g.cells.num, 1.0))       # 1 m of unseen relief

        assert rough.masses[5] > 0.0, "sub-scale trapping caught nothing"
        assert base.masses[5] == 0.0
        assert rough.mobile < base.mobile, \
            "sub-scale trapping must come out of the mobile CO2"
        assert np.isclose(rough.free_phase, base.free_phase, rtol=1e-10), \
            "it moves mass between categories, it does not create it"
        print(f"ok test_subscale_trapping_takes_from_the_mobile_part "
              f"({rough.masses[5] / rough.free_phase:.1%} of the plume caught "
              f"by 1 m of unresolved relief)")

    def test_inventory_history_is_monotone_in_safety():
        """Over the whole run, the CO2 only ever gets safer."""
        ts, rock2, fluid, res = _dome_case()
        states, n_inj = _run(ts, rock2, fluid)
        hist = inventory_history(ts, states, rock2, fluid, res)

        assert hist.shape == (len(states), 7)
        free = hist[n_inj:, 1:].sum(axis=1)

        # The sealed model neither gains nor loses CO2 after shut-in - but it
        # does drift by a few parts per million over 30 Newton solves, because
        # the CO2 is compressible and the solver stops at a finite tolerance.
        # That drift belongs to the VE model, not to the inventory: check the
        # inventory against the model's own mass (which it tracks to 1e-7), and
        # the model's conservation separately.
        pv = np.asarray(rock2.poro, float).ravel() * ts.area
        model_mass = np.array([
            float(np.sum(pv * fluid.rhoGS * fluid.b_g(s.pressure)
                         * s.sG * ts.H))
            for s in states[n_inj:]])
        assert np.abs(free - model_mass).max() / model_mass[0] < 1e-5, \
            "the inventory does not track the model's own mass"
        drift = (model_mass.max() - model_mass.min()) / model_mass[0]
        assert drift < 1e-4, f"the sealed model lost {drift:.1e} of its CO2"

        # the mobile fraction only ever falls: the CO2 gets safer, never riskier
        mobile = hist[n_inj:, 6] / free
        assert np.all(np.diff(mobile) <= 1e-9), \
            f"the mobile fraction should never rise during migration: {mobile}"
        print(f"ok test_inventory_history_is_monotone_in_safety "
              f"(mobile fraction {mobile[0]:.0%} -> {mobile[-1]:.0%} over "
              f"{len(mobile)} years; the model conserves mass to {drift:.1e})")

    test_inventory_closes_and_totals_the_injected_mass()
    test_migration_moves_mass_from_mobile_to_trapped()
    test_categories_respond_to_the_physics_that_drives_them()
    test_subscale_trapping_takes_from_the_mobile_part()
    test_inventory_history_is_monotone_in_safety()
    print("\nAll tests passed!")
