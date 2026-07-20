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


def _reference_index(model) -> int:
    """Oil is the reference phase when there is oil; otherwise the first."""
    names = model.fluid.phases
    return names.index("oil") if "oil" in names else 0


def _phase_densities(model) -> List[Callable[[np.ndarray], np.ndarray]]:
    """rho_i(p) = b_i(p) * rhoS_i - pressure-dependent, hence the ODE."""
    rho_s = model.fluid.rho_s
    nph = model.n_phases

    def make(i):
        def rho(p):
            b = model.fluid.b(np.atleast_1d(np.asarray(p, dtype=float)))
            return np.asarray(value(b[i]), dtype=float) * rho_s[i]
        return rho

    return [make(i) for i in range(nph)]


def _integrate_pressure(rho: Callable, p0: float, z0: float, g: float,
                        z_eval: np.ndarray, z_min: float, z_max: float):
    """Solve dp/dz = g rho(p) from (z0, p0), and return p(z) as a callable.

    MRST uses `ode45`; this uses the same explicit Runge-Kutta (RK45) with
    comparable tolerances.
    """
    def dp(z, p):
        return g * rho(p)

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


def initialize_equilibrium_pressures(model, region: EquilibriumRegion,
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

    rho = _phase_densities(model)
    ref = _reference_index(model)
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


def initialize_equilibrium_saturations(model, region: EquilibriumRegion,
                                       pressures: np.ndarray) -> np.ndarray:
    """Saturations from the phase pressures, by inverting capillarity."""
    nph = model.n_phases
    ref = _reference_index(model)
    n = pressures.shape[0]

    s_min = (np.zeros(nph) if region.s_min is None
             else np.asarray(region.s_min, dtype=float))
    s_max = (np.ones(nph) if region.s_max is None
             else np.asarray(region.s_max, dtype=float))
    pc_fns = (region.pc_functions if region.pc_functions is not None
              else [None] * nph)
    pc_sign = (np.ones(nph) if region.pc_sign is None
               else np.asarray(region.pc_sign, dtype=float))

    s = np.zeros((n, nph))
    p_ref = pressures[:, ref]

    for i in range(nph):
        if i == ref:
            continue
        dp = pressures[:, i] - p_ref
        if pc_fns[i] is None:
            # No capillary pressure: the transition zone collapses to a
            # contact. The phase is present exactly where its own pressure
            # exceeds the reference phase's - i.e. on its side of the contact.
            s[:, i] = np.where(dp > 0, s_max[i], s_min[i])
        else:
            si = _invert_capillary(dp, pc_fns[i], pc_sign[i])
            si = np.where(np.isnan(si), s_max[i], si)
            s[:, i] = np.clip(si, s_min[i], s_max[i])

    s[:, ref] = 1.0 - s.sum(axis=1)
    # A cell can end up over-filled where two contacts nearly coincide; give
    # it back to the reference phase rather than emit a negative saturation.
    bad = s[:, ref] < 0.0
    if np.any(bad):
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
    ref = _reference_index(model)

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
    p = initialize_equilibrium_pressures(model, region, gravity)
    s = initialize_equilibrium_saturations(model, region, p)
    ref = _reference_index(model)

    st = State(pressure=_reference_pressure(model, region, p, s),
               sW=s[:, 0].copy())
    if model.n_phases > 2:
        st["sO"] = s[:, 1].copy()
        st["sG"] = s[:, 2].copy()
    if model.fluid.disgas:
        # Saturated where there is free gas, undersaturated where there is not.
        rs_sat = np.asarray(value(model.raw_fluid.rs_sat(st["pressure"])),
                            dtype=float)
        st["rs"] = rs_sat.copy()
        st["status"] = np.where(s[:, 2] > 0, 3, 1).astype(np.int64)
    return st


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
    from pymrst.modules.ad_props.simple_adi_fluid import init_simple_adi_fluid
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
        return g, GenericBlackOilModel(g, rock, fluid, [], gravity=G_ACC)

    def test_pressure_follows_the_hydrostatic_gradient():
        """With an incompressible fluid the ODE has a closed form: the oil
        pressure is linear in depth with slope rho_o g."""
        g, m = _model()
        woc = 1050.0
        region = EquilibriumRegion(datum_depth=1000.0,
                                   datum_pressure=200 * BARSA,
                                   contacts=[woc])
        p = initialize_equilibrium_pressures(m, region, G_ACC)
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
        p = initialize_equilibrium_pressures(m, region, G_ACC)
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
