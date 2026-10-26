"""Equilibrium initialisation (MRST ad-core/initialization).

Every initial state in this port was set by hand: a uniform pressure and a
uniform saturation, which is not a reservoir. A real one sits in **gravity-
capillary equilibrium** - the fluids have had geological time to sort
themselves by density, so water lies under oil lies under gas, separated by
contacts, with each phase's pressure following its own hydrostatic gradient.
Start a simulation anywhere else and the first thing it does is fall towards
this state, which is not the answer you asked for.

The algorithm (MRST's `initializeEquilibriumPressures` /
`initializeEquilibriumSaturations`):

1. **Pressures.** For the reference phase (oil, when there is oil), integrate
   `dp/dz = g rho(p)` from the datum, both upward and downward. The density
   depends on the pressure, which is what is being solved for, so it is an ODE
   and not a formula - `rho` is only constant if the fluid is incompressible.
2. Anchor every other phase at its **contact**: at the water-oil contact the
   two phases differ by the capillary pressure there (zero, by default), and
   from that anchor the phase follows *its own* density.
3. **Saturations.** In a cell, the phases' pressures differ by
   `p_i - p_ref`, and capillarity says that difference *is* `pc(s)`. So invert
   the capillary curve. With no capillary pressure the inversion degenerates to
   a step - a sharp contact - which is exactly right.

That last point is the one worth understanding: a sharp fluid contact is not a
simplification bolted on, it is what a capillary transition zone becomes when
the capillary pressure goes to zero.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_blackoil.utils.get_cell_status_vo import (
    get_cell_status_vo,
)
from pymrst.utils.adi import value


@dataclass
class EquilibriumRegion:
    """One equilibrium region (MRST's `region` struct).

    `contacts` are the depths separating the phases, shallowest first: for
    water-oil-gas that is (gas-oil contact, water-oil contact) ordered as MRST
    orders them - one contact per phase boundary, `n_phases - 1` of them.
    """

    datum_depth: float
    datum_pressure: float
    contacts: Sequence[float]
    cells: Optional[np.ndarray] = None
    # pc[i](s) for each non-reference phase; None means a sharp contact.
    pc_functions: Optional[List[Optional[Callable]]] = None
    pc_sign: Optional[np.ndarray] = None
    contacts_pc: Optional[np.ndarray] = None
    s_min: Optional[np.ndarray] = None
    s_max: Optional[np.ndarray] = None
    reference_index: Optional[int] = None
    pc_scale: Optional[np.ndarray] = None
    # Dissolved-gas / vaporised-oil specification for the density integration
    # (MRST's `region.rs` / `region.rv`). A scalar constant, a callable of
    # pressure, or None. None means "saturated" - the oil rides at `rsSat(p)`
    # and the gas at `rvSat(p)`, which is the default when no RSVD table pins
    # the composition against depth. MRST *requires* rs for a disgas model;
    # PyMRST supplies the saturated default rather than erroring.
    rs: Optional[Callable] = None
    rv: Optional[Callable] = None


def _reference_index(model, region: Optional[EquilibriumRegion] = None) -> int:
    """Oil is the reference phase when there is oil; otherwise the first."""
    if region is not None and region.reference_index is not None:
        return int(region.reference_index)
    if hasattr(model, "getPhaseNames"):
        names = tuple(str(name).lower() for name in model.getPhaseNames())
        return names.index("o") if "o" in names else 0
    names = model.fluid.phases
    return names.index("oil") if "oil" in names else 0


def _as_composition(spec, p, sat_fn) -> Optional[np.ndarray]:
    """The dissolved-gas (or vaporised-oil) amount at pressure `p`.

    `None` -> the saturated value `rsSat(p)`; a callable -> `spec(p)`; a scalar
    -> that constant. MRST requires this to be given for a live fluid and caps
    it at saturation; the None default supplies the saturated value.
    """
    if sat_fn is None:
        return None
    sat = np.asarray(value(sat_fn(p)), dtype=float)
    if spec is None:
        return sat
    given = (np.asarray(value(spec(p)), dtype=float)
             if callable(spec) else np.full(p.shape, float(spec)))
    return given


def _phase_densities(model, region: "EquilibriumRegion | None" = None
                     ) -> List[Callable[[np.ndarray], np.ndarray]]:
    r"""`rho_i(p)` for the hydrostatic ODE (MRST's `getOilDensity`/`getGasDensity`).

    For a live-oil fluid the oil column carries its dissolved gas, so its
    density is `rho_o = b_o(p, rs) (rhoS_o + rs rhoS_g)`, and `rs` is a *given*
    composition (a constant, or an RSVD function of depth) **capped at
    saturation**: `rs = min(rs_given, rsSat(p))`, with the oil counted saturated
    exactly where `rs_given >= rsSat(p)`. Absent an `rs` (no RSVD table) the oil
    rides fully saturated, `rs = rsSat(p)`. Gas with vaporised oil is the mirror.

    This is MRST's `getInitializationRegionsBlackOil` density, formula for
    formula - including that passing `rs = None` for a live fluid would send
    `b_o` into its undersaturated branch and return NaN, which is why the
    saturated default exists. A dead-oil / analytic fluid ignores `rs` and this
    reduces to `b_i(p) rhoS_i`, leaving the black-oil-free path unchanged.
    """
    rho_s = np.asarray(model.fluid.rho_s, dtype=float)
    nph = model.n_phases
    disgas = bool(getattr(model.fluid, "disgas", False))
    vapoil = bool(getattr(model.fluid, "vapoil", False))
    rs_sat = getattr(model.fluid, "rs_sat", None) if disgas else None
    rv_sat = getattr(model.fluid, "rv_sat", None) if vapoil else None
    rs_spec = getattr(region, "rs", None) if region is not None else None
    rv_spec = getattr(region, "rv", None) if region is not None else None
    oil_ix, gas_ix = 1, 2                    # black-oil phase order W, O, G

    def make(i):
        def rho(p):
            p = np.atleast_1d(np.asarray(p, dtype=float))
            rs_given = _as_composition(rs_spec, p, rs_sat)
            rv_given = _as_composition(rv_spec, p, rv_sat)
            if i == oil_ix and rs_given is not None:
                rs_max = np.asarray(value(rs_sat(p)), float)
                rs_used = np.minimum(rs_given, rs_max)
                flag = rs_given >= rs_max
                b = model.fluid.b(p, rs_used, flag, None, None)
                bi = np.asarray(value(b[oil_ix]), float)
                return bi * (rho_s[oil_ix] + rs_used * rho_s[gas_ix])
            if i == gas_ix and rv_given is not None:
                rv_max = np.asarray(value(rv_sat(p)), float)
                rv_used = np.minimum(rv_given, rv_max)
                flag = rv_given >= rv_max
                b = model.fluid.b(p, None, None, rv_used, flag)
                bi = np.asarray(value(b[gas_ix]), float)
                return bi * (rho_s[gas_ix] + rv_used * rho_s[oil_ix])
            # water, or a phase with no dissolution
            sat = np.ones(p.shape, bool) if disgas else None
            sat_g = np.ones(p.shape, bool) if vapoil else None
            b = model.fluid.b(p, rs_given, sat, rv_given, sat_g)
            return np.asarray(value(b[i]), float) * rho_s[i]
        return rho

    return [make(i) for i in range(nph)]


def _density_at(rho: Callable, p, z):
    """Evaluate an MRST density closure, accepting ``rho(p)`` or ``rho(p,z)``."""
    try:
        return rho(p, z)
    except TypeError:
        return rho(p)


def _integrate_pressure(rho: Callable, p0: float, z0: float, g: float,
                        z_eval: np.ndarray, z_min: float, z_max: float):
    """Solve dp/dz = g rho(p) from (z0, p0), and return p(z) as a callable.

    MRST uses `ode45`; this uses the same explicit Runge-Kutta (RK45) with
    comparable tolerances.
    """
    def dp(z, p):
        return g * _density_at(rho, p, z)

    pieces = []
    if z_min < z0:
        up = solve_ivp(dp, (z0, z_min), [p0], method="RK45", dense_output=True,
                       rtol=5e-8, atol=1e-10)
        pieces.append((z_min, z0, up))
    if z_max > z0:
        down = solve_ivp(dp, (z0, z_max), [p0], method="RK45",
                         dense_output=True, rtol=5e-8, atol=1e-10)
        pieces.append((z0, z_max, down))

    def evaluate(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        out = np.full(z.shape, float(p0))
        for lo, hi, sol in pieces:
            m = (z >= lo) & (z <= hi) & (z != z0)
            if np.any(m):
                out[m] = sol.sol(z[m])[0]
        return out

    return evaluate


def initialize_equilibrium_pressures_ad(model, region: EquilibriumRegion,
                                     gravity: float) -> np.ndarray:
    """Each phase's own pressure in every cell: (n_cells, n_phases)."""
    cells = (np.arange(model.G.cells.num) if region.cells is None
             else np.asarray(region.cells, dtype=np.int64).ravel())
    z = model.G.cells.centroids[cells, 2]
    nph = model.n_phases
    contacts = np.asarray(region.contacts, dtype=float).ravel()
    assert contacts.size == nph - 1, (
        f"a {nph}-phase model needs {nph - 1} contacts, got {contacts.size}"
    )
    if gravity == 0.0:
        raise ValueError(
            "equilibrium initialisation with zero gravity is meaningless - "
            "there is nothing to sort the fluids by"
        )

    z_min = min(z.min(), contacts.min(), region.datum_depth)
    z_max = max(z.max(), contacts.max(), region.datum_depth)

    rho = _phase_densities(model, region)
    ref = _reference_index(model, region)
    pc_sign = (np.ones(nph) if region.pc_sign is None
               else np.asarray(region.pc_sign, dtype=float))
    contacts_pc = (np.zeros(nph - 1) if region.contacts_pc is None
                   else np.asarray(region.contacts_pc, dtype=float))

    p = np.zeros((cells.size, nph))

    # 1. the reference phase, from the datum
    p_ref_fn = _integrate_pressure(rho[ref], region.datum_pressure,
                                   region.datum_depth, gravity, z, z_min, z_max)
    p[:, ref] = p_ref_fn(z)

    # 2. every other phase, anchored at its contact
    k = 0
    for i in range(nph):
        if i == ref:
            continue
        contact = float(contacts[k])
        p_at_contact = float(p_ref_fn(np.array([contact]))[0]
                             + pc_sign[i] * contacts_pc[k])
        p[:, i] = _integrate_pressure(rho[i], p_at_contact, contact, gravity,
                                      z, z_min, z_max)(z)
        k += 1
    return p


def _invert_capillary(dp: np.ndarray, pc_fn: Callable, sign: float,
                      n: int = 10001) -> np.ndarray:
    """Find s such that sign*pc(s) = dp. Monotone inversion by interpolation."""
    s = np.linspace(0.0, 1.0, n)
    pc = sign * np.asarray(pc_fn(s), dtype=float)
    if pc[0] == pc[-1]:
        return np.full(dp.shape, np.nan)
    order = np.argsort(pc)
    pc_sorted, s_sorted = pc[order], s[order]
    pc_u, keep = np.unique(pc_sorted, return_index=True)
    f = interp1d(pc_u, s_sorted[keep], kind="linear",
                 bounds_error=False, fill_value=(s_sorted[keep][0],
                                                 s_sorted[keep][-1]))
    return f(dp)


def _connate_saturations(model, region: EquilibriumRegion) -> np.ndarray:
    """Per-phase minimum saturations for the equilibrium fill.

    MRST reads these from the relative-permeability endpoints
    (`getMinimumPhaseSaturations`): the connate water a gas cap holds, and any
    residual the other phases carry. Without them the gas cap comes out with
    zero water where a real reservoir holds `swcon`. The region may override;
    otherwise water takes the fluid's connate `swcon` and the rest take zero.
    """
    if region.s_min is not None:
        return np.asarray(region.s_min, dtype=float)
    nph = model.n_phases
    s_min = np.zeros(nph)
    swcon = getattr(model.fluid, "swcon", None)
    if swcon is None:
        swcon = getattr(getattr(model, "raw_fluid", None), "swcon", 0.0)
    s_min[0] = float(swcon)                   # water is phase 0
    return s_min


def _sharp_or_pc(dp, pc_fn, sign, s_lo, s_hi):
    """One phase's saturation from its pressure excess `dp` (MRST
    `solveSaturations`): the capillary-curve inverse, or a sharp contact when
    there is no capillary pressure - present at `s_hi` where `dp > 0`, at the
    connate `s_lo` where not."""
    if pc_fn is None:
        return np.where(dp > 0, s_hi, s_lo)
    si = _invert_capillary(dp, pc_fn, sign)
    si = np.where(np.isnan(si), s_hi, si)
    return np.clip(si, s_lo, s_hi)


def initialize_equilibrium_saturations_ad(model, region: EquilibriumRegion,
                                       pressures: np.ndarray) -> np.ndarray:
    """Saturations from the phase pressures, by inverting capillarity.

    Ports MRST's `initializeEquilibriumSaturations`, including the two-step
    resolution it does for cells the reference phase cannot fill: each
    non-reference phase is placed against the reference first, and where that
    leaves the reference phase *negative* - a cell with no room for oil - the
    remaining two phases are re-solved directly against **each other** (water
    versus gas), which is what turns a would-be 50/50 mush into a clean
    gas-over-water contact.
    """
    nph = model.n_phases
    ref = _reference_index(model, region)
    n = pressures.shape[0]

    s_min = _connate_saturations(model, region)
    # A phase can fill a cell only up to what the *other* phases' connate leaves
    # for it (MRST's `getMaximumPhaseSaturations`): a gas cap tops out at
    # `1 - swcon`, so the connate water it holds is not overwritten.
    if region.s_max is None:
        s_max = 1.0 - (s_min.sum() - s_min)
    else:
        s_max = np.asarray(region.s_max, dtype=float)
    pc_fns = (region.pc_functions if region.pc_functions is not None
              else [None] * nph)
    pc_sign = (np.ones(nph) if region.pc_sign is None
               else np.asarray(region.pc_sign, dtype=float))

    s = np.zeros((n, nph))
    p_ref = pressures[:, ref]

    for i in range(nph):
        if i == ref:
            continue
        s[:, i] = _sharp_or_pc(pressures[:, i] - p_ref, pc_fns[i], pc_sign[i],
                               s_min[i], s_max[i])

    s[:, ref] = 1.0 - s.sum(axis=1)

    # Cells with no room for the reference phase (three phases claim more than a
    # unit volume): re-solve the outer two phases against each other. With oil
    # as the middle reference, that is water (first) versus gas (last).
    bad = s[:, ref] < 0.0
    if np.any(bad) and nph == 3:
        first, last = 0, nph - 1
        assert ref not in (first, last), "reference phase must be interior"
        s[bad, :] = 0.0
        if pc_fns[first] is None and pc_fns[last] is None:
            dp = pressures[bad, last] - pressures[bad, first]   # gas vs water
            s_last = np.where(dp > 0, s_max[last], s_min[last])
        else:
            def pc_wg(S):
                pw = 0.0 if pc_fns[first] is None else pc_sign[first] * pc_fns[first](S)
                pg = 0.0 if pc_fns[last] is None else pc_sign[last] * pc_fns[last](S)
                return pw - pg
            s_last = _sharp_or_pc(pressures[bad, last] - pressures[bad, first],
                                  pc_wg, 1.0, s_min[last], s_max[last])
        s[bad, last] = s_last
        s[bad, first] = 1.0 - s_last
    elif np.any(bad):
        s[bad] = np.maximum(s[bad], 0.0)
        s[bad] /= s[bad].sum(axis=1, keepdims=True)
    return s


def _reference_pressure(model, region: EquilibriumRegion, p: np.ndarray,
                        s: np.ndarray) -> np.ndarray:
    """Which phase's pressure the model should *carry* (MRST initStateBlackOilAD).

    A flow model has one pressure variable, and every other phase's pressure is
    derived from it by the capillary pressure. So which phase should it be?

    The reference phase - except where the reference phase is **immobile**. In
    the water leg below the contact there is no oil at all, so the oil pressure
    there is a fiction (an extrapolation of a phase that is not present), and
    carrying it means the water sees a gradient of `rho_oil g` instead of
    `rho_water g`. The water then flows, and the "equilibrium" state falls over
    the moment you run it. Below the contact the model must carry the *water*
    pressure.

    Getting this wrong is not subtle in its consequences and completely
    invisible in the initial state: it looks perfectly hydrostatic, and then
    moves.
    """
    nph = model.n_phases
    ref = _reference_index(model, region)

    kr = model.fluid.kr([s[:, i] for i in range(nph)])
    kr = np.column_stack([np.asarray(value(k), dtype=float) for k in kr])
    ref_immobile = kr[:, ref] < 1e-8
    max_sat = s.max(axis=1)

    pc_fns = (region.pc_functions if region.pc_functions is not None
              else [None] * nph)
    pc_sign = (np.ones(nph) if region.pc_sign is None
               else np.asarray(region.pc_sign, dtype=float))

    out = p[:, ref].copy()
    for i in range(nph):
        if i == ref:
            continue
        dominant = (s[:, i] == max_sat) & ref_immobile
        if not np.any(dominant):
            continue
        pc_i = (np.zeros(s.shape[0]) if pc_fns[i] is None
                else pc_sign[i] * np.asarray(pc_fns[i](s[:, i]), dtype=float))
        out[dominant] = p[dominant, i] - pc_i[dominant]
    return out


def init_state_equil(model, region: EquilibriumRegion,
                     gravity: float) -> State:
    """The equilibrium initial state (MRST's `initStateBlackOilAD`)."""
    p = initialize_equilibrium_pressures_ad(model, region, gravity)
    s = initialize_equilibrium_saturations_ad(model, region, p)
    ref = _reference_index(model, region)

    st = State(pressure=_reference_pressure(model, region, p, s),
               sW=s[:, 0].copy())
    if model.n_phases > 2:
        st["sO"] = s[:, 1].copy()
        st["sG"] = s[:, 2].copy()
    if model.fluid.disgas:
        # MRST's `initStateBlackOilAD`: take the region's dissolved-gas
        # composition, cap it at saturation, and force it to saturation wherever
        # there is free gas. Absent a region `rs`, the composition *is* the
        # saturated value, so this reduces to "saturated everywhere" - which was
        # the old behaviour, and is wrong the moment an undersaturated oil leg
        # is specified.
        #
        # `rsSat` is evaluated at the **oil-phase** pressure, not the pressure
        # the state carries. They differ below the water-oil contact, where the
        # state carries water pressure but the oil composition still answers to
        # the oil pressure it would have there.
        p_oil = p[:, ref]
        rs_max = np.asarray(value(model.raw_fluid.rs_sat(p_oil)), dtype=float)
        rs_given = _as_composition(region.rs, np.atleast_1d(p_oil),
                                   model.raw_fluid.rs_sat)
        rs = np.minimum(rs_given, rs_max)
        rs[s[:, 2] > 0] = rs_max[s[:, 2] > 0]
        st["rs"] = rs
        # MRST's `getCellStatusVO`, not `sG > 0 ? 3 : 1`. The two differ on a
        # cell with **no free oil**, which the ad-hoc rule calls status 1
        # (undersaturated oil) - and status 1's switched variable is `rs`,
        # which reaches the equations only through `rs * b_o * s_o`. With
        # `s_o = 0` it multiplies nothing, its column is structurally empty,
        # and the Jacobian is singular before any numerics.
        #
        # `getCellStatusVO` declares a water-only cell to have *both*
        # hydrocarbon phases, so its variable is `sG` - which enters through
        # `s_g` directly and is determined whatever its value. That is the
        # only well-posed choice, and it is why Norne's water leg made the
        # whole field model unsolvable.
        st1, st2, st3 = get_cell_status_vo(
            s[:, 1], s[:, 0], s[:, 2],
            disgas=bool(model.fluid.disgas), vapoil=bool(model.fluid.vapoil),
        )
        st["status"] = 1*np.asarray(st1) + 2*np.asarray(st2) + 3*np.asarray(st3)
    if model.fluid.vapoil:
        p_gas = p[:, 2]                       # rvSat answers to the gas pressure
        rv_max = np.asarray(value(model.raw_fluid.rv_sat(p_gas)), dtype=float)
        rv_given = _as_composition(region.rv, np.atleast_1d(p_gas),
                                   model.raw_fluid.rv_sat)
        rv = np.minimum(rv_given, rv_max)
        rv[s[:, 1] > 0] = rv_max[s[:, 1] > 0]
        st["rv"] = rv
    return st


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.

if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModelAD
    from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
    from pymrst.params.gravity_const import NORM_GRAVITY
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    G_ACC = NORM_GRAVITY
    RHO_W, RHO_O = 1000.0, 800.0

    def _model(nz=20, top=1000.0, thick=100.0, pc_ow=None, c=(0.0, 0.0)):
        """`c = 0` makes the hydrostatic gradient exactly linear, which is what
        the closed-form tests check against. A *compressible* fluid is needed
        to run the model at all, though: with no wells, no boundary and an
        incompressible fluid the pressure is only defined up to a constant and
        the Jacobian is singular."""
        g = cart_grid([1, 1, nz], [10.0, 10.0, thick])
        g.nodes.coords[:, 2] += top          # shift *before* the geometry, or
        g = compute_geometry(g)              # the faces and cells disagree
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[RHO_W, RHO_O],
            n=[2, 2], c=list(c), p_ref=200 * BARSA, pc_ow=pc_ow,
        )
        return g, GenericBlackOilModelAD(g, rock, fluid, [], gravity=G_ACC)

    def test_pressure_follows_the_hydrostatic_gradient():
        """With an incompressible fluid the ODE has a closed form: the oil
        pressure is linear in depth with slope rho_o g."""
        g, m = _model()
        woc = 1050.0
        region = EquilibriumRegion(datum_depth=1000.0,
                                   datum_pressure=200 * BARSA,
                                   contacts=[woc])
        p = initialize_equilibrium_pressures_ad(m, region, G_ACC)
        z = g.cells.centroids[:, 2]

        p_oil = 200 * BARSA + RHO_O * G_ACC * (z - 1000.0)
        assert np.allclose(p[:, 1], p_oil, rtol=1e-8), "oil gradient wrong"

        # water is anchored at the contact and follows its own, steeper slope
        p_at_woc = 200 * BARSA + RHO_O * G_ACC * (woc - 1000.0)
        p_wat = p_at_woc + RHO_W * G_ACC * (z - woc)
        assert np.allclose(p[:, 0], p_wat, rtol=1e-8), "water gradient wrong"
        print("ok test_pressure_follows_the_hydrostatic_gradient")

    def test_the_phases_cross_exactly_at_the_contact():
        """The definition of a contact: it is where the two phase pressures
        are equal. If they cross anywhere else, the initialisation is wrong."""
        g, m = _model()
        woc = 1050.0
        region = EquilibriumRegion(datum_depth=1000.0,
                                   datum_pressure=200 * BARSA,
                                   contacts=[woc])
        p = initialize_equilibrium_pressures_ad(m, region, G_ACC)
        z = g.cells.centroids[:, 2]

        dp = p[:, 0] - p[:, 1]                    # water minus oil
        crossing = np.interp(0.0, dp, z)          # dp is increasing with depth
        assert abs(crossing - woc) < 1e-6, (
            f"the phase pressures cross at {crossing:.4f} m, not at the "
            f"contact at {woc} m"
        )
        print(f"ok test_the_phases_cross_exactly_at_the_contact "
              f"({crossing:.6f} m)")

    def test_a_sharp_contact_without_capillary_pressure():
        """No pc means water below the contact, oil above it - and nothing in
        between. The transition zone has collapsed to a step."""
        g, m = _model(nz=20)
        woc = 1050.0
        region = EquilibriumRegion(datum_depth=1000.0,
                                   datum_pressure=200 * BARSA,
                                   contacts=[woc])
        st = init_state_equil(m, region, G_ACC)
        z = g.cells.centroids[:, 2]

        assert np.all(st["sW"][z > woc] == 1.0), "water below the contact"
        assert np.all(st["sW"][z < woc] == 0.0), "oil above it"
        assert set(np.unique(st["sW"])) <= {0.0, 1.0}, "and nothing in between"
        print("ok test_a_sharp_contact_without_capillary_pressure")

    def test_capillary_pressure_makes_a_transition_zone():
        """Turn pc on and the step becomes a gradient - the capillary
        transition zone - sitting *above* the contact, because capillarity
        holds water up against gravity."""
        pc0 = 0.5 * BARSA
        g, m = _model(nz=40, pc_ow=lambda sw: pc0 * (1.0 - sw))
        woc = 1050.0
        region = EquilibriumRegion(
            datum_depth=1000.0, datum_pressure=200 * BARSA, contacts=[woc],
            pc_functions=[lambda sw: pc0 * (1.0 - sw), None],
            pc_sign=np.array([-1.0, 1.0]),      # water carries -pcOW
        )
        st = init_state_equil(m, region, G_ACC)
        z = g.cells.centroids[:, 2]

        sw = st["sW"]
        assert np.all(np.diff(sw) >= -1e-12), "water must increase with depth"
        partial = (sw > 1e-6) & (sw < 1 - 1e-6)
        assert partial.sum() >= 3, (
            f"a capillary transition zone should span several cells, got "
            f"{partial.sum()}"
        )
        assert np.all(z[partial] < woc + 1e-9), (
            "the transition zone sits above the contact - capillarity holds "
            "water up against gravity, it does not pull oil down"
        )
        assert sw[z > woc].min() > 0.9, "still water-filled well below"
        print(f"ok test_capillary_pressure_makes_a_transition_zone "
              f"({partial.sum()} cells, "
              f"{z[partial].min():.1f}-{z[partial].max():.1f} m)")

    def test_equilibrium_is_actually_at_rest():
        """THE test: a state called 'equilibrium' must not move.

        Run the model from it with no wells, no boundary and no sources. If the
        initialisation is right, nothing happens - and if it is not, the field
        immediately falls towards the state it should have started in.
        """
        from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
        from pymrst.utils.units import DAY

        g, m = _model(nz=20, c=(1e-6 / BARSA, 1e-4 / BARSA))
        region = EquilibriumRegion(datum_depth=1000.0,
                                   datum_pressure=200 * BARSA,
                                   contacts=[1050.0])
        st = init_state_equil(m, region, G_ACC)
        st["qWs"] = np.zeros(0)
        st["qOs"] = np.zeros(0)
        st["bhp"] = np.zeros(0)

        p0, s0 = st["pressure"].copy(), st["sW"].copy()
        solver = NonLinearSolver(max_iterations=30)
        for _ in range(3):
            st, rep = solver.solve_ministep(m, st, st.copy(), 100 * DAY)
            assert rep.converged

        dp = np.abs(st["pressure"] - p0).max() / np.abs(p0).max()
        ds = np.abs(st["sW"] - s0).max()
        assert dp < 1e-9, f"the field moved: pressure changed by {dp:.2e}"
        assert ds < 1e-9, f"the field moved: saturation changed by {ds:.2e}"
        print(f"ok test_equilibrium_is_actually_at_rest "
              f"(300 days: dp {dp:.1e}, ds {ds:.1e})")

    test_pressure_follows_the_hydrostatic_gradient()
    test_the_phases_cross_exactly_at_the_contact()
    test_a_sharp_contact_without_capillary_pressure()
    test_capillary_pressure_makes_a_transition_zone()
    test_equilibrium_is_actually_at_rest()
    print("\nAll tests passed!")
