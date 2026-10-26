"""Two-phase incompressible transport, ported from MRST incomp/transport.

Solves  s_t + div[f(s)(v + mo(s) G)] = f(s) q  for the water saturation
with single-point phase-upwind mobility weighting:

- explicit_transport: forward Euler with a CFL-style step estimate
  (explicitTransport.m).
- implicit_transport: backward Euler solved by Newton-Raphson with a
  bisection line search and automatic substepping (implicitTransport.m,
  newtonRaphson2ph.m).

Both build on the residual/Jacobian of twophaseJacobian.m. Capillary
pressure is not yet ported (fluids here have no pc).
"""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.params.gravity_const import gravity_or_zero
from pymrst.params.rock import Rock, perm_tensor, pore_volume
from pymrst.params.wells_and_bc import BoundaryCondition, Source, Well
from pymrst.modules.incomp.fluid import Fluid
from pymrst.solvers.state import State

ResidualFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]
JacobianFn = Callable[[np.ndarray, np.ndarray, float], sp.csr_array]


def transport_source(
    state: State,
    g: Grid,
    fluid: Fluid,
    wells: Optional[Sequence[Well]] = None,
    src: Optional[Source] = None,
    bc: Optional[BoundaryCondition] = None,
) -> np.ndarray:
    """Net water-relevant source per cell (m^3/s): inflow is scaled by the
    fractional flow of the injected composition, outflow kept total."""
    cells_list, flux_list, compi_list = [], [], []

    for k, w in enumerate(wells or []):
        cells_list.append(w.cells)
        flux_list.append(state.wellSol[k].flux)
        compi_list.append(np.tile(w.compi[:fluid.n_phases],
                                  (w.cells.size, 1)))
    if src is not None and src.cell.size > 0:
        cells_list.append(src.cell)
        flux_list.append(src.rate)
        compi_list.append(src.sat[:, :fluid.n_phases])
    if bc is not None and bc.face.size > 0:
        assert bc.sat.size > 0, "transport needs bc saturations"
        n = g.faces.neighbors[bc.face]
        assert np.all(np.sum(n == -1, axis=1) == 1), (
            "boundary condition supplied on internal face"
        )
        bdry_cell = n.sum(axis=1) + 1
        flux_sign = np.where(n[:, 0] == -1, 1.0, -1.0)
        types = np.array(bc.type)
        is_dir = types == "pressure"
        is_neu = types == "flux"
        cells_list.append(bdry_cell[is_dir])
        flux_list.append(flux_sign[is_dir] * state.flux[bc.face[is_dir]])
        cells_list.append(bdry_cell[is_neu])
        flux_list.append(bc.value[is_neu])
        compi_list.append(bc.sat[is_dir][:, :fluid.n_phases])
        compi_list.append(bc.sat[is_neu][:, :fluid.n_phases])

    if not cells_list:
        return np.zeros(g.cells.num)

    cells = np.concatenate(cells_list)
    flux = np.concatenate(flux_list)
    compi = np.vstack(compi_list)

    inj = np.zeros(g.cells.num, dtype=bool)
    prod = np.zeros(g.cells.num, dtype=bool)
    inj[cells[flux > 0]] = True
    prod[cells[flux < 0]] = True
    assert not np.any(inj & prod), (
        "injection and production in the same cell is unsupported"
    )

    inflow = flux > 0
    if np.any(inflow):
        m = fluid.mobility(compi[inflow])
        f = m[:, 0] / m.sum(axis=1)
        flux = flux.copy()
        flux[inflow] *= f
    return np.bincount(cells, weights=flux, minlength=g.cells.num)


def gravity_flux(
    g: Grid, rock: Rock, delta_rho_g: np.ndarray
) -> np.ndarray:
    """Harmonic average over half-faces of n'K(g(rho_w - rho_o)) per face."""
    if not np.linalg.norm(delta_rho_g) > 0:
        return np.zeros(g.faces.num)
    k, r, c = perm_tensor(rock, g.griddim)
    cell_no = g.cell_no()
    cf = g.cells.faces
    nkg = np.sum(g.faces.normals[cf][:, r] * k[cell_no] * delta_rho_g[c],
                 axis=1)
    with np.errstate(divide="ignore"):
        acc = np.bincount(cf, weights=1.0 / nkg, minlength=g.faces.num)
        out = 2.0 / acc
    out[~np.isfinite(out)] = 0.0
    return out


def _upwind_indices(
    n_int: np.ndarray, dflux: np.ndarray, gflux: np.ndarray, mob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Water/oil upwind cell per internal face (upwindIndices port)."""
    iw = np.full(dflux.size, -1, dtype=np.int64)
    io = np.full(dflux.size, -1, dtype=np.int64)

    # Water: sign of water flux is mobility-independent when v, g agree.
    a = ~(dflux < 0) & ~(gflux < 0)
    b = ~(dflux > 0) & ~(gflux > 0)
    c = a | b
    n = n_int.copy()
    n[b] = n[b][:, ::-1]
    iw[c] = n[c, 0]

    # Oil: mobility-independent when v, g disagree.
    a = ~(dflux < 0) & ~(gflux > 0)
    b = ~(dflux > 0) & ~(gflux < 0)
    c = a | b
    n = n_int.copy()
    n[b] = n[b][:, ::-1]
    io[c] = n[c, 0]

    undecided = (iw < 0) | (io < 0)
    if np.any(undecided):
        mw = np.where(iw >= 0, mob[np.maximum(iw, 0), 0], np.nan)
        mo = np.where(io >= 0, mob[np.maximum(io, 0), 1], np.nan)
        vw = dflux + gflux * mo
        vo = dflux - gflux * mw
        n = n_int.copy()
        n[vw < 0] = n[vw < 0][:, ::-1]
        iw[iw < 0] = n[iw < 0, 0]
        n = n_int.copy()
        n[vo < 0] = n[vo < 0][:, ::-1]
        io[io < 0] = n[io < 0, 0]
    return iw, io


def two_phase_residual_jacobian(
    g: Grid,
    state: State,
    rock: Rock,
    fluid: Fluid,
    wells: Optional[Sequence[Well]] = None,
    src: Optional[Source] = None,
    bc: Optional[BoundaryCondition] = None,
    gravity: Optional[np.ndarray] = None,
) -> Tuple[ResidualFn, JacobianFn, np.ndarray, np.ndarray]:
    """Build residual F(s, s0, dt) and Jacobian dF/ds (twophaseJacobian).

    Returns (residual, jacobian, gflux, q). The Darcy fluxes and source
    strengths are frozen from `state`, as in the sequential splitting.
    """
    assert fluid.n_phases == 2, "two-phase transport requires a 2-phase fluid"
    q = transport_source(state, g, fluid, wells=wells, src=src, bc=bc)

    g_vec = gravity_or_zero(gravity, g.griddim)
    gflux = gravity_flux(g, rock, g_vec * (fluid.rho[0] - fluid.rho[1]))

    neighbors = g.faces.neighbors
    internal = np.all(neighbors >= 0, axis=1)
    n_int = neighbors[internal]
    ic1, ic2 = n_int[:, 0], n_int[:, 1]
    dflux = state.flux[internal]
    gflux_int = gflux[internal]
    pv = pore_volume(g, rock)
    nc = g.cells.num

    def upwind(mob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _upwind_indices(n_int, dflux, gflux_int, mob)

    def residual(s: np.ndarray, s0: np.ndarray, dt: float) -> np.ndarray:
        sat = np.column_stack([s, 1.0 - s])
        m = fluid.mobility(sat)
        f_cell = m[:, 0] / m.sum(axis=1)
        q_w = np.maximum(q, 0.0) + np.minimum(q, 0.0) * f_cell
        iw, io = upwind(m)
        mw, mo = m[iw, 0], m[io, 1]
        f_face = mw / (mw + mo)
        v_water = f_face * (dflux + gflux_int * mo)
        div = np.bincount(np.concatenate([ic1, ic2]),
                          weights=np.concatenate([v_water, -v_water]),
                          minlength=nc)
        return s - s0 + dt / pv * (div - q_w)

    def jacobian(s: np.ndarray, s0: np.ndarray, dt: float) -> sp.csr_array:
        sat = np.column_stack([s, 1.0 - s])
        m = fluid.mobility(sat)
        dm = fluid.d_mobility(sat)
        dm = np.column_stack([dm[:, 0], -dm[:, 1]])  # d/ds_w of each phase

        iw, io = upwind(m)
        mw, mo = m[iw, 0], m[io, 1]
        mt = mw + mo
        f_face = np.column_stack([mw / mt, mo / mt])

        dmw_fo = f_face[:, 1] * dm[iw, 0]
        dmo_fw = f_face[:, 0] * dm[io, 1]
        dvw_ds1 = (dflux + gflux_int * mo) * dmw_fo / mt
        dvw_ds2 = -(dflux - gflux_int * mw) * dmo_fw / mt

        mt = m.sum(axis=1)
        df = (m[:, 1] * dm[:, 0] - m[:, 0] * dm[:, 1]) / mt**2
        d = 1.0 - dt / pv * (np.minimum(q, 0.0) * df)

        cid = np.arange(nc)
        rows = np.concatenate([cid, ic1, ic2, ic1, ic2])
        cols = np.concatenate([cid, iw, iw, io, io])
        vals = np.concatenate([
            d,
            dt / pv[ic1] * dvw_ds1, -(dt / pv[ic2] * dvw_ds1),
            dt / pv[ic1] * dvw_ds2, -(dt / pv[ic2] * dvw_ds2),
        ])
        return sp.coo_array((vals, (rows, cols)), shape=(nc, nc)).tocsr()

    return residual, jacobian, gflux, q


def _estimate_dt(
    g: Grid,
    state: State,
    rock: Rock,
    fluid: Fluid,
    gflux_grav: np.ndarray,
    q: np.ndarray,
) -> float:
    """Wave-speed based stable time step (estimate_dt in explicitTransport)."""
    internal = np.all(g.faces.neighbors >= 0, axis=1)
    n_int = g.faces.neighbors[internal]

    m = fluid.mobility(state.s)
    dm = fluid.d_mobility(state.s)
    dm = np.column_stack([dm[:, 0], -dm[:, 1]])

    df = (m[:, 1] * dm[:, 0] - m[:, 0] * dm[:, 1]) / m.sum(axis=1) ** 2
    wavespeed = np.max(
        np.max(np.abs(df[n_int]), axis=1) * np.abs(state.flux[internal]),
        initial=0.0,
    )

    # Gravity segregation wave speed from a saturation sweep.
    s1 = np.linspace(0.0, 1.0, 101)
    m_loc = fluid.relperm(np.column_stack([s1, 1 - s1])) / fluid.mu.min()
    seg = m_loc[:, 0] * m_loc[:, 1] / m_loc.sum(axis=1)
    dg = np.max(np.abs(np.diff(seg)) / np.diff(s1))
    drho = abs(fluid.rho[1] - fluid.rho[0])
    wavespeed = max(
        wavespeed,
        np.max(np.abs(dg * gflux_grav[internal] * drho), initial=0.0),
    )

    # Source-driven wave speeds over the reachable saturation windows.
    for sel, lo_hi in ((q > 0, "up"), (q < 0, "down")):
        if not np.any(sel):
            continue
        if lo_hi == "up":
            s_grid = np.arange(state.s[sel, 0].min(), 1.0 + 1e-12, 0.05)
        else:
            s_grid = np.arange(state.s[sel, 0].max(), -1e-12, -0.05)
        if s_grid.size < 2:
            continue
        mg = fluid.mobility(np.column_stack([s_grid, 1 - s_grid]))
        fg = mg[:, 0] / mg.sum(axis=1)
        maxdf = np.max(np.diff(fg) / np.diff(s_grid))
        wavespeed = max(wavespeed, np.max(np.abs(q[sel] * maxdf)))

    if not wavespeed > 0:
        return np.inf
    return float(np.min(pore_volume(g, rock) / wavespeed))


def explicit_transport(
    state: State,
    g: Grid,
    tf: float,
    rock: Rock,
    fluid: Fluid,
    wells: Optional[Sequence[Well]] = None,
    src: Optional[Source] = None,
    bc: Optional[BoundaryCondition] = None,
    gravity: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
    max_dt: float = np.inf,
    dt_factor: float = 0.5,
    satwarn: float = np.sqrt(np.finfo(float).eps),
) -> State:
    """Advance saturations tf seconds with forward-Euler upwind steps.

    dt=None estimates a stable step each substep (MRST 'computedt');
    an explicit dt gives fixed substeps, useful for exact comparisons.
    """
    residual, _, gflux, q = two_phase_residual_jacobian(
        g, state, rock, fluid, wells=wells, src=src, bc=bc, gravity=gravity
    )
    # MRST estimates dt with gravity flux built from -g and unit rho diff.
    g_vec = gravity_or_zero(gravity, g.griddim)
    gflux_est = gravity_flux(g, rock, -g_vec)

    s = state.s[:, 0].copy()
    t = 0.0
    while t < tf:
        if dt is None:
            step = min(max_dt,
                       dt_factor * _estimate_dt(g, state, rock, fluid,
                                                gflux_est, q))
        else:
            step = dt
        step = min(tf - t, step)
        s = s - residual(s, s, step)
        t += step

        assert not np.any(s > 1 + satwarn) and not np.any(s < -satwarn), (
            "explicit_transport: saturation outside [0, 1] beyond satwarn"
        )
        s = np.clip(s, 0.0, 1.0)
        state.s = np.column_stack([s, 1.0 - s])

    assert not np.any(np.isnan(s)), "explicit_transport: transport failed"
    return state


def implicit_transport(
    state: State,
    g: Grid,
    tf: float,
    rock: Rock,
    fluid: Fluid,
    wells: Optional[Sequence[Well]] = None,
    src: Optional[Source] = None,
    bc: Optional[BoundaryCondition] = None,
    gravity: Optional[np.ndarray] = None,
    nltol: float = 1e-6,
    lstrials: int = 20,
    maxnewt: int = 25,
    tsref: int = 12,
    resred: float = 0.99,
) -> State:
    """Advance saturations tf seconds with backward Euler + Newton-Raphson.

    Substeps are halved on Newton failure and grown by 1.5x after easy
    steps, exactly as MRST's newtonRaphson2ph.
    """
    residual, jacobian, _, _ = two_phase_residual_jacobian(
        g, state, rock, fluid, wells=wells, src=src, bc=bc, gravity=gravity
    )

    s = state.s[:, 0].copy()
    mints = tf * 2.0 ** (-tsref)
    t, dt, dtprev, count = 0.0, tf, -tf, 0

    while t < tf and dt >= mints:
        dt = min(dt, tf - t)
        while True:
            s0 = s.copy()
            sn = np.minimum(1.0, s + 0.05)
            res = residual(sn, s0, dt)
            err = np.abs(res).max()
            it = 0
            linfail = False
            while err > nltol and not linfail and it < maxnewt:
                jac = jacobian(sn, s0, dt)
                ds = -spla.spsolve(jac.tocsc(), res)
                sn, res, linfail = _linesearch(
                    sn, ds, resred * err,
                    lambda x: residual(x, s0, dt), lstrials
                )
                it += 1
                err = np.abs(res).max()

            count -= 1
            if err > nltol:  # Newton failed: reduce the substep.
                dt = dtprev if (count > 0 and dtprev > 0) else dt / 2.0
                count = 5
            else:
                t += dt
                dtprev = -dt
                if t < tf and it <= 5 and count < 0:  # easy step: grow.
                    dtprev = dt
                    dt = min(1.5 * dt, tf - t)
                    count = 5
                break
        s = sn

    assert t >= tf and not np.any(np.isnan(s)), (
        f"implicit_transport failed: t={t:.3e} of {tf:.3e}, "
        f"minimum substep {mints:.3e} reached"
    )
    state.s = np.column_stack([s, 1.0 - s])
    return state


def _linesearch(
    s: np.ndarray,
    ds: np.ndarray,
    target: float,
    res_fn: Callable[[np.ndarray], np.ndarray],
    ntrials: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Bisection line search: halve the update until the residual drops."""
    alph = 0
    fail = True
    sn, res = s, res_fn(s)
    for _ in range(ntrials):
        sn = np.clip(s + ds * 2.0**alph, 0.0, 1.0)
        res = res_fn(sn)
        alph -= 1
        if np.abs(res).max() < target:
            fail = False
            break
    return sn, res, fail


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.gravity_const import gravity_vector
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_source
    from pymrst.modules.incomp.fluid import init_simple_fluid
    from pymrst.modules.incomp.incomp_tpfa import incomp_tpfa_ad as incomp_tpfa
    from pymrst.solvers.state import init_res_sol
    from pymrst.solvers.trans import compute_trans
    from pymrst.utils.units import CENTI, DARCY, DAY, MILLI, POISE

    def _bl_setup(nx: int):
        """1D horizontal displacement: unit fluids, unit fractional flow."""
        g = compute_geometry(cart_grid([nx, 1, 1], [100.0, 1.0, 1.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        fluid = init_simple_fluid(mu=[1e-3, 1e-3], rho=[1000.0, 1000.0],
                                  n=[2.0, 2.0])
        t_half = compute_trans(g, rock)
        rate = g.cells.volumes[0] * 0.2 * nx / (100 * DAY)  # 1 PV / 100 days
        src = add_source(None, np.array([0]), rate, sat=np.array([[1.0, 0.0]]))
        src = add_source(src, np.array([nx - 1]), -rate,
                         sat=np.array([[0.0, 1.0]]))
        st = init_res_sol(g, 0.0, s0=np.array([[0.0, 1.0]]))
        st = incomp_tpfa(st, g, t_half, fluid, src=src)
        return g, rock, fluid, src, st

    def _bl_analytic(x_d: np.ndarray, t_d: float) -> np.ndarray:
        """Buckley-Leverett solution for f = s^2/(s^2+(1-s)^2), M=1."""
        s = np.linspace(0.0, 1.0, 100001)
        f = s**2 / (s**2 + (1 - s) ** 2)
        dfds = np.gradient(f, s)
        sf = np.sqrt(0.5)  # front saturation: f(sf)/sf = f'(sf)
        vf = (f[np.searchsorted(s, sf)] / sf)
        sat = np.zeros_like(x_d)
        behind = x_d <= vf * t_d
        # invert x/t = f'(s) on the rarefaction branch s in [sf, 1]
        branch = s >= sf
        sat[behind] = np.interp(x_d[behind] / t_d,
                                dfds[branch][::-1], s[branch][::-1])
        return sat

    def test_mass_balance_explicit():
        """Explicit transport conserves water volume exactly."""
        g, rock, fluid, src, st = _bl_setup(50)
        pv = pore_volume(g, rock)
        w0 = float(pv @ st.s[:, 0])
        tf = 20 * DAY
        st = explicit_transport(st, g, tf, rock, fluid, src=src)
        injected = src.rate[0] * tf   # producer still flows pure oil
        assert st.s[-1, 0] < 1e-10, "front must not have broken through"
        w1 = float(pv @ st.s[:, 0])
        assert np.isclose(w1 - w0, injected, rtol=1e-10)
        print("ok test_mass_balance_explicit")

    def test_buckley_leverett_explicit():
        """Explicit solution converges to the analytic BL profile."""
        g, rock, fluid, src, st = _bl_setup(200)
        t_d = 0.3  # injected pore volumes
        st = explicit_transport(st, g, t_d * 100 * DAY, rock, fluid, src=src)
        x_d = (g.cells.centroids[:, 0]) / 100.0
        exact = _bl_analytic(x_d, t_d)
        err = np.mean(np.abs(st.s[:, 0] - exact))
        assert err < 0.02, f"BL L1 error {err:.4f} too large"
        print(f"ok test_buckley_leverett_explicit (L1 err {err:.4f})")

    def test_implicit_matches_explicit():
        """Implicit and explicit agree in L1 and front position; the shock
        itself is smeared over a couple of cells by backward Euler."""
        g, rock, fluid, src, st_e = _bl_setup(100)
        _, _, _, src2, st_i = _bl_setup(100)
        tf = 10 * DAY
        st_e = explicit_transport(st_e, g, tf, rock, fluid, src=src)
        # Drive implicit with small horizons so BE smearing vanishes.
        for _ in range(40):
            st_i = implicit_transport(st_i, g, tf / 40, rock, fluid, src=src2)
        assert np.mean(np.abs(st_e.s[:, 0] - st_i.s[:, 0])) < 0.01
        front_e = int(np.argmin(st_e.s[:, 0] > 0.4))
        front_i = int(np.argmin(st_i.s[:, 0] > 0.4))
        assert abs(front_e - front_i) <= 1
        assert np.all((st_i.s >= 0) & (st_i.s <= 1))
        pv = pore_volume(g, rock)
        assert np.isclose(pv @ st_i.s[:, 0], src.rate[0] * tf, rtol=1e-6)
        print("ok test_implicit_matches_explicit")

    def test_gravity_segregation():
        """Heavy water sinks in a closed column (implicit, gravity only)."""
        g = compute_geometry(cart_grid([1, 1, 20], [1.0, 1.0, 20.0]))
        rock = make_rock(g, 1000 * MILLI * DARCY, 0.3)
        fluid = init_simple_fluid(mu=[1e-3, 1e-3], rho=[1000.0, 700.0],
                                  n=[2.0, 2.0])
        st = init_res_sol(g, 0.0, s0=np.array([[0.5, 0.5]]))
        st.flux = np.zeros(g.faces.num)  # no Darcy flow, only segregation
        st = implicit_transport(st, g, 200 * DAY, rock, fluid,
                                gravity=gravity_vector(3))
        z = g.cells.centroids[:, 2]
        assert np.isclose(st.s[:, 0].mean(), 0.5, atol=1e-8)  # conservation
        assert np.all(np.diff(st.s[:, 0]) > 0)   # monotone: heavy sinks
        assert st.s[z < 5, 0].mean() < 0.15      # top drains toward oil
        assert st.s[z > 15, 0].mean() > 0.85     # bottom fills with water
        print("ok test_gravity_segregation")

    test_mass_balance_explicit()
    test_buckley_leverett_explicit()
    test_implicit_matches_explicit()
    test_gravity_segregation()
    print("\nAll tests passed!")
