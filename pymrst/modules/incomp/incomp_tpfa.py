"""Incompressible TPFA pressure solver, ported from MRST incompTPFA.m.

Solves for cell pressures (and well bottom-hole pressures) of

    v = -lam K (grad p - omega g),   div v = q

with two-point flux approximation, then reconstructs face pressures,
face fluxes, and well connection fluxes. Supports Dirichlet/Neumann
boundary conditions, cell sources, bhp/rate wells, and gravity.
"""

from typing import List, Optional, Sequence, Tuple

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
from pymrst.solvers.state import State, WellSol


def incomp_tpfa(
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
        st = incomp_tpfa(init_res_sol(g, 0.0), g, t, fluid, bc=bc)
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
        st = incomp_tpfa(init_res_sol(g, 0.0), g, t, fluid, src=src)
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
        st = incomp_tpfa(init_res_sol(g, 0.0), g, t, fluid, bc=bc,
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
        st = incomp_tpfa(init_state(g, w, 0.0), g, t, fluid, wells=w)
        assert np.isclose(st.wellSol[0].flux.sum(), rate, rtol=1e-10)
        assert np.isclose(st.wellSol[1].flux.sum(), -rate, rtol=1e-10)
        assert st.wellSol[0].pressure > st.pressure[0] > st.pressure[24]
        assert np.isclose(st.wellSol[1].pressure, 100 * BARSA)
        print("ok test_rate_and_bhp_wells")

    test_linear_pressure_drop()
    test_mass_conservation_with_sources()
    test_hydrostatic_column()
    test_rate_and_bhp_wells()
    print("\nAll tests passed!")
