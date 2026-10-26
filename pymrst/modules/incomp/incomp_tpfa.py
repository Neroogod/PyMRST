"""MRST ``incomp/incompTPFA.m``: the incompressible pressure equation
with two-point fluxes,

    v = -lam K (grad p - omega g),   div v = q,

for cell pressures and well bottom-hole pressures, then face pressures,
face fluxes and well connection fluxes. As in MRST: one-sided
transmissibilities, or one per interface with `use_trans`; the grid's
non-neighbouring connections; boundary conditions, sources, wells (`wells`
or `W`), periodic `bcp`; capillary pressure in the `pc_form` given;
gravity from MRST's global `gravity()`; `LinSolve`, `MatrixOutput` and the
well-eliminating `reduce`. Indices are 0-based.

`incomp_tpfa_ad`, below, is PyMRST's earlier version, kept for its callers;
it takes gravity as an argument (off by default) instead of reading the
global setting. Its notes follow.

Incompressible TPFA pressure solver, ported from MRST incompTPFA.m.

Solves for cell pressures (and well bottom-hole pressures) of

    v = -lam K (grad p - omega g),   div v = q

with two-point flux approximation, then reconstructs face pressures,
face fluxes, and well connection fluxes. Supports Dirichlet/Neumann
boundary conditions, cell sources, bhp/rate wells, and gravity.
"""

import copy
import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.params.gravity_const import gravity_or_zero
from pymrst.params.wells_and_bc import (
    BoundaryCondition,
    Source,
    Well,
    incomp_well_pressure_drop,
)
from pymrst.modules.incomp.fluid import Fluid
from pymrst.modules.incomp.cap_pressure_rhs import cap_pressure_rhs
from pymrst.modules.incomp.compute_incomp_well_pressure_drop import compute_incomp_well_pressure_drop
from pymrst.modules.incomp.compute_pressure_rhs import compute_pressure_rhs
from pymrst.modules.incomp.fluid.utils.get_incomp_capillary_pressure import (
    get_incomp_capillary_pressure,
)
from pymrst.solvers.state import State, WellSol


def incomp_tpfa(state: Any, G: Grid, T: np.ndarray, fluid: Any, bc: Any = None, src: Any = None,
                wells: Any = None, gravity: Any = None, *, W: Any = None, bcp: Any = None,
                LinSolve: Optional[Callable] = None, MatrixOutput: bool = False,
                Verbose: bool = False, condition_number: bool = False,
                pc_form: str = "nonwetting", reduce: bool = False,
                use_trans: bool = False) -> Any:
    """MRST `state = incompTPFA(state, G, T, fluid, ...)`. `T` holds one-sided
    transmissibilities, or with `use_trans` one transmissibility per
    interface. A new state is returned; the input is not modified."""
    from pymrst.modules.incomp.check_driving_forces_incomp import check_driving_forces_incomp
    from pymrst.modules.incomp.treat_legacy_force_options import treat_legacy_force_options
    from pymrst.params.gravity.gravity import gravity as global_gravity
    from pymrst.utils.gridtools.get_cell_no_faces import get_cell_no_faces
    from pymrst.utils.gridtools.get_neighbourship import get_neighbourship
    opt = treat_legacy_force_options({
        "bc": bc, "src": src, "wells": wells, "W": W, "bcp": bcp,
        "MatrixOutput": MatrixOutput,
        "gravity": global_gravity() if gravity is None else np.asarray(gravity, dtype=float)})
    state = copy.copy(state)
    if not check_driving_forces_incomp(G, opt):
        return state
    g_vec = np.resize(opt["gravity"], 3)
    grav_on = np.linalg.norm(g_vec[:G.griddim]) > 0 or hasattr(G, "grav_pressure")
    Wl = list(opt["wells"] or [])
    if not MatrixOutput and all(x is None for x in (opt["bc"], opt["src"], opt["bcp"])) \
            and not Wl and not grav_on:
        warnings.warn("No external driving forces present in model--state remains unchanged.\n")

    neighborship, n_isnnc = get_neighbourship(G, topological=True, include_boundary=True,
                                              with_nnc_flag=True)
    cellNo, cf, cn_isnnc = get_cell_no_faces(G)
    nif = neighborship.shape[0]
    ncf = cf.size
    nc = G.cells.num
    nw = len(Wl)
    n = nc + nw
    mob, omega, rho = _dynamic_quantities(state, fluid)
    totmob = mob.sum(axis=1)
    T, ft = _compute_trans(T, cellNo, cf, neighborship, totmob, use_trans)
    i = np.all(neighborship >= 0, axis=1)
    hh = np.zeros(nif)
    dF = np.zeros(nif, dtype=bool)
    grav = np.zeros(ncf)
    ff = np.zeros(ncf)
    ff_r, gg, hh_r, grav_r, dF_r, dC = compute_pressure_rhs(G, omega, opt["bc"], opt["src"])
    ff[~cn_isnnc], hh[~n_isnnc], grav[~cn_isnnc], dF[~n_isnnc] = ff_r, hh_r, grav_r, dF_r
    pc = get_incomp_capillary_pressure(state, fluid)
    if pc is not None and np.any(np.abs(pc) > 0):
        grav = grav + cap_pressure_rhs(G, mob, pc, pc_form)

    sgn = 2.0*(neighborship[cf, 0] == cellNo) - 1
    j = i[cf] | dF[cf]
    fg = np.bincount(cf[j], grav[j]*sgn[j], minlength=nif)
    if opt["bcp"] is not None:
        bcp_face = np.asarray(opt["bcp"].face, dtype=np.int64).ravel()
        np.add.at(fg, bcp_face, np.asarray(opt["bcp"].value, dtype=float).ravel())
        warnings.warn("Face pressures are not well defined for periodic boundary faces")
        if np.any(G.faces.neighbors[:, 0] == G.faces.neighbors[:, 1]):
            raise ValueError("Periodic boundary: This code do not work if a face is in and outflow")
    rhs = (np.bincount(cellNo, -ft[cf]*(sgn*fg[cf] + ff), minlength=n)
           + np.concatenate([gg, np.zeros(nw)])
           + np.bincount(cellNo, -hh[cf], minlength=n))

    d = np.zeros(nc)
    C_rows, C_cols, C_vals = [], [], []
    D = np.zeros(nw)
    norm_g = float(np.linalg.norm(global_gravity()))
    well_sols = list(getattr(state, "wellSol", None) or [])
    while len(well_sols) < nw:
        well_sols.append(WellSol(flux=np.zeros(0), pressure=np.nan))
    well_sols = [copy.copy(ws) for ws in well_sols]
    for k, Wk in enumerate(Wl):
        wc = np.asarray(Wk.cells, dtype=np.int64).ravel()
        w = k + nc
        wi = np.asarray(Wk.WI, dtype=float).ravel()*totmob[wc]
        dp = compute_incomp_well_pressure_drop(Wk, mob, rho, norm_g)
        np.add.at(d, wc, wi)
        well_sols[k].cdp = dp
        kind = str(Wk.type).lower()
        if kind == "bhp":
            ww = wi.max()
            rhs[w] += ww*float(np.ravel(Wk.val)[0])
            np.add.at(rhs, wc, wi*(float(np.ravel(Wk.val)[0]) + dp))
            D[k] = ww
        elif kind == "rate":
            rhs[w] += float(np.ravel(Wk.val)[0])
            np.add.at(rhs, wc, wi*dp)
            C_rows.extend([k]*wc.size)
            C_cols.extend(wc.tolist())
            C_vals.extend((-wi).tolist())
            D[k] = wi.sum()
            rhs[w] -= wi @ dp
        else:
            raise ValueError("Unsupported well type.")
    C = sp.csr_array((C_vals, (C_rows, C_cols)), shape=(nw, nc))
    Dm = sp.diags_array(D, shape=(nw, nw), format="csr")
    d = (d + np.bincount(cellNo[dF[cf]], T[dF[cf]], minlength=nc)
         + np.bincount(neighborship[i].ravel(order="F"), np.tile(ft[i], 2), minlength=nc))
    I = np.concatenate([neighborship[i, 0], neighborship[i, 1], np.arange(nc)])
    J = np.concatenate([neighborship[i, 1], neighborship[i, 0], np.arange(nc)])
    V = np.concatenate([-ft[i], -ft[i], d])
    A = sp.csr_array((V, (I, J)), shape=(nc, nc))
    A = sp.block_array([[A, C.T], [C, Dm]], format="lil")
    if not np.any(dF) and (not Wl or not any(str(x.type).lower() == "bhp" for x in Wl)):
        if A[0, 0] > 0:
            A[0, 0] = 2*A[0, 0]
        else:
            jmax = int(np.argmax(A.diagonal()))
            A[jmax, jmax] = 2*A[jmax, jmax]
    A = A.tocsr()
    if condition_number:
        print("***************************************")
        print(f"Conditon number is :   {np.linalg.cond(A.toarray())}")
        print("***************************************")
    if MatrixOutput:
        state.A = A
        state.rhs = rhs
    solve = LinSolve if LinSolve is not None else (lambda M, b: spla.spsolve(sp.csc_matrix(M), b))
    if reduce:
        rhs_r, A_r, rhs_w = rhs[:nc], A[:nc, :nc], rhs[nc:]
        Dinv = sp.diags_array(1/D, shape=(nw, nw)) if nw else Dm
        rhs_r = rhs_r - C.T @ (Dinv @ rhs_w)
        A_r = A_r - C.T @ Dinv @ C
        state.A, state.rhs = A_r, rhs_r
        p_r = np.asarray(solve(A_r, rhs_r)).ravel()
        p = np.full(rhs.shape, np.nan)
        p[:nc] = p_r
        p[nc:] = Dinv @ (rhs_w - C @ p_r)
    else:
        p = np.asarray(solve(A, rhs)).ravel()

    fpress = (np.bincount(cf, (p[cellNo] + grav)*T, minlength=nif)
              / np.bincount(cf, T, minlength=nif))
    nf = G.faces.num
    b = np.zeros(nif, dtype=bool)
    b[:nf] = np.any(np.asarray(G.faces.neighbors) < 0, axis=1)
    fpress[b] = fpress[b] - hh[b]/ft[b]
    fpress[dF] = dC
    ext = ~i
    sgn_b = 2.0*(neighborship[ext, 1] < 0) - 1
    ni = neighborship[i]
    flux = np.zeros(nif)
    flux[i] = -(ft[i]*(p[ni[:, 1]] - p[ni[:, 0]] - fg[i]))
    c = neighborship[ext].max(axis=1)
    fg = np.bincount(cf, grav, minlength=nif)
    flux[ext] = -sgn_b*ft[ext]*(fpress[ext] - p[c] - fg[ext])
    state.pressure = p[:nc]
    state.flux = flux
    state.facePressure = fpress
    for k, Wk in enumerate(Wl):
        wc = np.asarray(Wk.cells, dtype=np.int64).ravel()
        dp = well_sols[k].cdp
        well_sols[k].flux = np.asarray(Wk.WI, dtype=float).ravel()*totmob[wc]*(p[nc + k] + dp - p[wc])
        well_sols[k].pressure = float(p[nc + k])
    if nw or well_sols:
        state.wellSol = well_sols
    return state


def _dynamic_quantities(state: Any, fluid: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from pymrst.modules.incomp.fluid.utils.get_incomp_props import get_incomp_props
    rho, kr, mu = get_incomp_props(state, fluid)
    mob = kr/mu
    totmob = mob.sum(axis=1)
    omega = (mob*rho).sum(axis=1)/totmob
    return mob, omega, rho


def _compute_trans(T: np.ndarray, cellNo: np.ndarray, cellFaces: np.ndarray,
                   neighborship: np.ndarray, totmob: np.ndarray,
                   use_trans: bool) -> Tuple[np.ndarray, np.ndarray]:
    niface = neighborship.shape[0]
    T = np.asarray(T, dtype=float).ravel()
    if use_trans:
        neighborcount = (neighborship >= 0).sum(axis=1)
        assert T.size == niface, (
            f"Expected one transmissibility for each interface (={niface}) but got {T.size}.")
        fmob = np.bincount(cellFaces, totmob[cellNo], minlength=niface)
        fmob = fmob/neighborcount
        ft = T*fmob
        th = ft*neighborcount
        return th[cellFaces], ft
    assert T.size == cellNo.size, (
        f"Expected one one-sided transmissibility for each half face (={cellNo.size}), "
        f"but got {T.size}.")
    T = T*totmob[cellNo]
    with np.errstate(divide="ignore"):
        ft = 1/np.bincount(cellFaces, 1/T, minlength=niface)
    return T, ft


incompTPFA = incomp_tpfa


# ---------------------------------------------------------------------------
# PyMRST's earlier version.

def incomp_tpfa_ad(
    state: State,
    g: Grid,
    t_half: np.ndarray,
    fluid: Fluid,
    bc: Optional[BoundaryCondition] = None,
    src: Optional[Source] = None,
    wells: Optional[Sequence[Well]] = None,
    gravity: Optional[np.ndarray] = None,
) -> State:
    """Solve the incompressible pressure equation; update state in place.

    t_half are the one-sided transmissibilities from compute_trans (one
    per half-face). gravity is a griddim vector or None (off).
    """
    wells = list(wells or [])
    g_vec = gravity_or_zero(gravity, g.griddim)
    grav_on = np.linalg.norm(g_vec) > 0

    neighbors = g.faces.neighbors
    internal = np.all(neighbors >= 0, axis=1)
    cell_no = g.cell_no()
    cf = g.cells.faces
    nf, nc, nw = g.faces.num, g.cells.num, len(wells)
    n = nc + nw

    # Dynamic quantities
    mob = fluid.mobility(state.s)
    totmob = mob.sum(axis=1)
    omega = (mob * fluid.rho).sum(axis=1) / totmob

    # Mobility-weighted half-face and harmonic face transmissibilities.
    t_hf = t_half * totmob[cell_no]
    ft = 1.0 / np.bincount(cf, weights=1.0 / t_hf, minlength=nf)

    # Right-hand-side contributions (computePressureRHS).
    grav_hf, ff, gg, hh, d_face, d_val = _pressure_rhs(
        g, omega, bc, src, g_vec, grav_on, cell_no, cf
    )

    sgn = np.where(neighbors[cf, 0] == cell_no, 1.0, -1.0)
    j = internal[cf] | d_face[cf]
    fg = np.bincount(cf[j], weights=grav_hf[j] * sgn[j], minlength=nf)

    rhs = np.zeros(n)
    rhs[:nc] += np.bincount(cell_no, weights=-ft[cf] * (sgn * fg[cf] + ff),
                            minlength=nc)
    rhs[:nc] += gg
    rhs[:nc] += np.bincount(cell_no, weights=-hh[cf], minlength=nc)

    # Wells: extra unknowns (bhp) coupled through well indices.
    d = np.zeros(nc)
    c_rows, c_cols, c_vals = [], [], []
    d_well = np.zeros(nw)
    cdp: List[np.ndarray] = []
    for k, w in enumerate(wells):
        wc = w.cells
        wi = w.WI * totmob[wc]
        dp = incomp_well_pressure_drop(w, mob, fluid.rho,
                                       float(np.linalg.norm(g_vec)))
        cdp.append(dp)
        d[wc] += wi
        if w.type == "bhp":
            ww = wi.max()
            rhs[nc + k] += ww * w.val
            rhs[wc] += wi * (w.val + dp)
            d_well[k] = ww
        elif w.type == "rate":
            rhs[nc + k] += w.val
            rhs[wc] += wi * dp
            c_rows.extend([k] * wc.size)
            c_cols.extend(wc.tolist())
            c_vals.extend((-wi).tolist())
            d_well[k] = wi.sum()
            rhs[nc + k] -= wi @ dp
        else:
            raise ValueError(f"unsupported well type '{w.type}'")

    # Diagonal: internal faces both sides + Dirichlet half-faces.
    hf_dirichlet = d_face[cf]
    d += np.bincount(cell_no[hf_dirichlet], weights=t_hf[hf_dirichlet],
                     minlength=nc)
    d += np.bincount(neighbors[internal].ravel(order="F"),
                     weights=np.tile(ft[internal], 2), minlength=nc)

    ni = neighbors[internal]
    rows = np.concatenate([ni[:, 0], ni[:, 1], np.arange(nc)])
    cols = np.concatenate([ni[:, 1], ni[:, 0], np.arange(nc)])
    vals = np.concatenate([-ft[internal], -ft[internal], d])
    a = sp.coo_array((vals, (rows, cols)), shape=(nc, nc)).tocsr()

    if nw > 0:
        c = sp.coo_array((c_vals, (c_rows, c_cols)), shape=(nw, nc)).tocsr()
        a = sp.block_array(
            [[a, c.T], [c, sp.diags_array(d_well, format="csr")]], format="csr"
        )

    # Remove the pressure null-space if nothing anchors the pressure level.
    if not np.any(d_face) and not any(w.type == "bhp" for w in wells):
        a = a.tolil()
        if a[0, 0] > 0:
            a[0, 0] = 2.0 * a[0, 0]
        else:
            jmax = int(np.argmax(a.diagonal()))
            a[jmax, jmax] = 2.0 * a[jmax, jmax]
        a = a.tocsr()

    p = spla.spsolve(a.tocsc(), rhs)

    # Reconstruct face pressures and fluxes.
    fpress = (np.bincount(cf, weights=(p[cell_no] + grav_hf) * t_hf,
                          minlength=nf)
              / np.bincount(cf, weights=t_hf, minlength=nf))
    boundary = np.any(neighbors == -1, axis=1)
    fpress[boundary] -= hh[boundary] / ft[boundary]
    fpress[d_face] = d_val[d_face]

    flux = np.zeros(nf)
    flux[internal] = -ft[internal] * (p[ni[:, 1]] - p[ni[:, 0]] - fg[internal])
    bsgn = np.where(neighbors[~internal, 1] == -1, 1.0, -1.0)
    bcell = neighbors[~internal].sum(axis=1) + 1  # the non-(-1) neighbor
    fg_all = np.bincount(cf, weights=grav_hf, minlength=nf)
    flux[~internal] = -bsgn * ft[~internal] * (
        fpress[~internal] - p[bcell] - fg_all[~internal]
    )

    state.pressure = p[:nc]
    state.flux = flux
    state.facePressure = fpress
    state.wellSol = []
    for k, w in enumerate(wells):
        wc = w.cells
        state.wellSol.append(WellSol(
            flux=w.WI * totmob[wc] * (p[nc + k] + cdp[k] - p[wc]),
            pressure=float(p[nc + k]),
            cdp=cdp[k],
        ))
    return state


def _pressure_rhs(
    g: Grid,
    omega: np.ndarray,
    bc: Optional[BoundaryCondition],
    src: Optional[Source],
    g_vec: np.ndarray,
    grav_on: bool,
    cell_no: np.ndarray,
    cf: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Port of computePressureRHS: gravity, Dirichlet, Neumann, sources."""
    nf, nc = g.faces.num, g.cells.num

    if grav_on:
        cvec = g.faces.centroids[cf] - g.cells.centroids[cell_no]
        grav_hf = omega[cell_no] * (cvec @ g_vec)
    else:
        grav_hf = np.zeros(cf.size)

    ff = np.zeros(cf.size)
    gg = np.zeros(nc)
    hh = np.zeros(nf)
    d_face = np.zeros(nf, dtype=bool)
    d_val = np.zeros(nf)

    if src is not None and src.cell.size > 0:
        assert src.cell.max() < nc, "source term refers to nonexistent cell"
        gg += np.bincount(src.cell, weights=src.rate, minlength=nc)

    if bc is not None and bc.face.size > 0:
        assert bc.face.max() < nf, "boundary condition on nonexistent face"
        assert np.unique(bc.face).size == bc.face.size, (
            "repeated faces in boundary condition"
        )
        types = np.array(bc.type)
        is_press = types == "pressure"
        d_face[bc.face[is_press]] = True
        d_val[bc.face[is_press]] = bc.value[is_press]
        hf_d = d_face[cf]
        ff[hf_d] = -d_val[cf[hf_d]]
        is_flux = types == "flux"
        hh[bc.face[is_flux]] = -bc.value[is_flux]

    return grav_hf, ff, gg, hh, d_face, d_val


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.gravity_const import gravity_vector
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_bc, add_source, add_well
    from pymrst.modules.incomp.fluid import init_single_fluid
    from pymrst.solvers.state import init_res_sol, init_state
    from pymrst.solvers.trans import compute_trans
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def test_linear_pressure_drop():
        """Dirichlet 200->100 bar across a box gives a linear profile."""
        g = compute_geometry(cart_grid([10, 1, 1], [100.0, 10.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
        t = compute_trans(g, rock)
        west = np.flatnonzero((np.any(g.faces.neighbors == -1, axis=1))
                              & (g.faces.centroids[:, 0] < 1e-10))
        east = np.flatnonzero((np.any(g.faces.neighbors == -1, axis=1))
                              & (g.faces.centroids[:, 0] > 100.0 - 1e-10))
        bc = add_bc(None, west, "pressure", 200 * BARSA)
        bc = add_bc(bc, east, "pressure", 100 * BARSA)
        st = incomp_tpfa_ad(init_res_sol(g, 0.0), g, t, fluid, bc=bc)
        expect = 200e5 - (np.arange(10) + 0.5) / 10.0 * 100e5
        assert np.allclose(st.pressure, expect, rtol=1e-12)
        q_analytic = (100 * MILLI * DARCY) * 100.0 * 100e5 / (1e-3 * 100.0)
        internal = np.all(g.faces.neighbors >= 0, axis=1)
        assert np.allclose(st.flux[internal], q_analytic, rtol=1e-12)
        print("ok test_linear_pressure_drop")

    def test_mass_conservation_with_sources():
        """Injector + producer sources: fluxes balance in every cell."""
        g = compute_geometry(cart_grid([8, 8, 1], [100.0, 100.0, 1.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
        t = compute_trans(g, rock)
        rate = 1e-3
        src = add_source(None, np.array([0]), rate)
        src = add_source(src, np.array([63]), -rate)
        st = incomp_tpfa_ad(init_res_sol(g, 0.0), g, t, fluid, src=src)
        cell_no, cf = g.cell_no(), g.cells.faces
        sgn = np.where(g.faces.neighbors[cf, 0] == cell_no, 1.0, -1.0)
        div = np.zeros(g.cells.num)
        np.add.at(div, cell_no, sgn * st.flux[cf])
        q = np.zeros(g.cells.num)
        q[0], q[63] = rate, -rate
        assert np.max(np.abs(div - q)) < 1e-12 * rate / 1e-3
        print("ok test_mass_conservation_with_sources")

    def test_hydrostatic_column():
        """Gravity column reproduces p = p_top + rho*g*z exactly."""
        g = compute_geometry(cart_grid([1, 1, 20], [1.0, 1.0, 200.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
        t = compute_trans(g, rock)
        top = np.flatnonzero((np.any(g.faces.neighbors == -1, axis=1))
                             & (g.faces.centroids[:, 2] < 1e-10))
        bc = add_bc(None, top, "pressure", 100 * BARSA)
        gvec = gravity_vector(3)
        st = incomp_tpfa_ad(init_res_sol(g, 0.0), g, t, fluid, bc=bc,
                         gravity=gvec)
        z = g.cells.centroids[:, 2]
        assert np.allclose(st.pressure, 100e5 + 1000.0 * 9.80665 * z,
                           rtol=1e-12)
        assert np.max(np.abs(st.flux)) < 1e-16
        print("ok test_hydrostatic_column")

    def test_rate_and_bhp_wells():
        """Rate injector + bhp producer: well fluxes are consistent."""
        g = compute_geometry(cart_grid([5, 5, 1], [500.0, 500.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
        t = compute_trans(g, rock)
        rate = 5e-3
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=rate, radius=0.1)
        w = add_well(w, g, rock, np.array([24]), well_type="bhp",
                     val=100 * BARSA, radius=0.1)
        st = incomp_tpfa_ad(init_state(g, w, 0.0), g, t, fluid, wells=w)
        assert np.isclose(st.wellSol[0].flux.sum(), rate, rtol=1e-10)
        assert np.isclose(st.wellSol[1].flux.sum(), -rate, rtol=1e-10)
        assert st.wellSol[0].pressure > st.pressure[0] > st.pressure[24]
        assert np.isclose(st.wellSol[1].pressure, 100 * BARSA)
        print("ok test_rate_and_bhp_wells")

    def _five_spot():
        g = compute_geometry(cart_grid([5, 5, 2], [500.0, 500.0, 20.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
        w = add_well(None, g, rock, np.array([0, 25]), well_type="rate", val=5e-3, radius=0.1)
        w = add_well(w, g, rock, np.array([24, 49]), well_type="bhp", val=100 * BARSA, radius=0.1)
        return g, rock, fluid, w

    def test_faithful_matches_the_earlier_version():
        from pymrst.params.gravity.gravity import isolated_gravity
        g, rock, fluid, w = _five_spot()
        t = compute_trans(g, rock)
        with isolated_gravity("reset", "on"):
            new = incomp_tpfa(init_state(g, w, 0.0), g, t, fluid, wells=w)
        old = incomp_tpfa_ad(init_state(g, w, 0.0), g, t, fluid, wells=w, gravity=gravity_vector(3))
        assert np.allclose(new.pressure, old.pressure, rtol=1e-13)
        assert np.allclose(new.flux, old.flux, rtol=1e-10, atol=1e-18)
        for a, b in zip(new.wellSol, old.wellSol):
            assert np.allclose(a.flux, b.flux, rtol=1e-10) and np.isclose(a.pressure, b.pressure)
        print("ok test_faithful_matches_the_earlier_version")

    def test_interface_transmissibilities_with_use_trans():
        """MRST's `use_trans`: harmonic face transmissibilities give the same
        solve as the one-sided ones they come from."""
        from pymrst.params.gravity.gravity import isolated_gravity
        g, rock, fluid, w = _five_spot()
        t = compute_trans(g, rock)
        T = 1 / np.bincount(g.cells.faces[:, 0] if g.cells.faces.ndim > 1 else g.cells.faces,
                            1 / t, minlength=g.faces.num)
        with isolated_gravity("off"):
            half = incomp_tpfa(init_state(g, w, 0.0), g, t, fluid, wells=w)
            face = incomp_tpfa(init_state(g, w, 0.0), g, T, fluid, wells=w, use_trans=True)
        internal = np.all(g.faces.neighbors >= 0, axis=1)
        assert np.allclose(face.pressure, half.pressure, rtol=1e-12)
        assert np.allclose(face.flux[internal], half.flux[internal], rtol=0,
                           atol=1e-10*np.abs(half.flux).max())
        print("ok test_interface_transmissibilities_with_use_trans")

    def test_reduce_and_matrix_output():
        from pymrst.params.gravity.gravity import isolated_gravity
        g, rock, fluid, w = _five_spot()
        t = compute_trans(g, rock)
        with isolated_gravity("off"):
            full = incomp_tpfa(init_state(g, w, 0.0), g, t, fluid, W=w, MatrixOutput=True)
            red = incomp_tpfa(init_state(g, w, 0.0), g, t, fluid, W=w, reduce=True)
        assert full.A.shape == (g.cells.num + 2, g.cells.num + 2) and full.rhs.size == g.cells.num + 2
        assert red.A.shape == (g.cells.num, g.cells.num)
        assert np.allclose(red.pressure, full.pressure, rtol=1e-12)
        assert np.isclose(red.wellSol[0].pressure, full.wellSol[0].pressure, rtol=1e-12)
        print("ok test_reduce_and_matrix_output")

    def test_input_state_is_not_modified():
        from pymrst.params.gravity.gravity import isolated_gravity
        g, rock, fluid, w = _five_spot()
        state = init_state(g, w, 0.0)
        with isolated_gravity("off"):
            incomp_tpfa(state, g, compute_trans(g, rock), fluid, wells=w)
        assert not np.any(state.pressure) and state.wellSol[0].pressure == 5e-3
        print("ok test_input_state_is_not_modified")

    test_faithful_matches_the_earlier_version()
    test_interface_transmissibilities_with_use_trans()
    test_reduce_and_matrix_output()
    test_input_state_is_not_modified()
    test_linear_pressure_drop()
    test_mass_conservation_with_sources()
    test_hydrostatic_column()
    test_rate_and_bhp_wells()
    print("\nAll tests passed!")
