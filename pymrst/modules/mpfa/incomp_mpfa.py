"""Incompressible single-phase MPFA pressure solve (MRST `incompMPFA`, Dirichlet).

Assembles and solves the incompressible pressure system with the multi-point flux
approximation transmissibility: `A p = q`, where `A = div · M · (K grad)` is built
from `compute_multipoint_trans`. This is the point of MPFA - on a grid that is not
K-orthogonal (skewed cells, or a full/rotated permeability tensor) it reproduces
a linear pressure field *exactly*, whereas TPFA does not.

This port covers the case the discriminating test needs: a single incompressible
phase (unit mobility) driven by Dirichlet (pressure) boundary conditions, with
optional cell sources. Mobility weighting, Neumann/flux BC, wells and gravity -
the rest of MRST's `incompMPFAlegacy` - are left for a later pass.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pymrst.solvers.state_types import WellSol


@dataclass
class MPFAResult:
    """Pressure/flux result from :func:`incomp_mpfa`."""

    pressure: np.ndarray
    flux: np.ndarray
    boundary_pressure: np.ndarray
    subface_flux: np.ndarray
    well_sol: list[WellSol]


def incomp_mpfa(
    g,
    T: Dict,
    bc_pressure: Optional[Dict[int, float]] = None,
    q: Optional[np.ndarray] = None,
    *,
    bc=None,
    src=None,
    mobility: Union[float, np.ndarray] = 1.0,
    density: Union[float, np.ndarray] = 0.0,
    gravity: Optional[np.ndarray] = None,
    wells: Optional[Sequence] = None,
    return_result: bool = False,
) -> Union[np.ndarray, MPFAResult]:
    """Solve a single-phase MPFA problem.

    ``bc_pressure`` preserves the original compact API and maps boundary faces
    to prescribed pressures.  The MRST-style ``bc`` object may instead contain
    mixed ``"pressure"`` and ``"flux"`` conditions; unspecified boundary faces
    are homogeneous no-flow. ``q`` or ``src`` supplies cell sources (injection
    positive). ``mobility`` is scalar or one total-mobility value per cell.
    ``density`` and ``gravity`` add the single-phase hydrostatic contribution.
    ``wells`` supports MRST-style rate and BHP controls.

    By default only cell pressure is returned for backward compatibility.
    Set ``return_result=True`` to also receive conservative face/subface fluxes
    and the solved boundary pressures.
    """
    nc = g.cells.num
    nf = g.faces.num
    sb = np.asarray(T["sb"], dtype=bool)
    d1 = T["d1"]

    mob = np.asarray(mobility, dtype=float).ravel()
    if mob.size == 1:
        mob = np.full(nc, mob.item())
    if mob.size != nc or np.any(mob <= 0):
        raise ValueError("mobility must be positive and scalar or one value per cell")

    # MRST's face mobility is the harmonic combination of adjacent cell
    # mobilities (one-sided on the boundary), copied to every MPFA subface.
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    cellno = np.repeat(np.arange(nc), np.diff(g.cells.facePos))
    inv_face_mob = np.bincount(
        cell_faces, weights=1.0 / mob[cellno], minlength=nf
    )
    face_mob = 1.0 / inv_face_mob
    subface_mob = np.asarray(d1 @ face_mob).ravel()
    M = sp.diags_array(subface_mob, format="csr")

    A = (T["e_div"] @ M @ T["cf_trans"]).tocsc()
    n_bsub = int(sb.sum())
    rhs = np.zeros(nc + n_bsub)
    if q is not None and src is not None:
        raise ValueError("pass either q or src, not both")
    if q is not None:
        q_arr = np.asarray(q, dtype=float).ravel()
        if q_arr.size != nc:
            raise ValueError("q must contain one source rate per cell")
        rhs[:nc] += q_arr
    if src is not None and src.cell.size:
        if np.any(src.cell < 0) or np.any(src.cell >= nc):
            raise ValueError("source refers to a nonexistent cell")
        np.add.at(rhs[:nc], src.cell, src.rate)

    rho = np.asarray(density, dtype=float).ravel()
    if rho.size == 1:
        rho = np.full(nc, rho.item())
    if rho.size != nc:
        raise ValueError("density must be scalar or one value per cell")
    gvec = np.zeros(g.griddim) if gravity is None else np.asarray(gravity, dtype=float).ravel()
    if gvec.size != g.griddim:
        raise ValueError("gravity must have one component per grid dimension")
    grav_half = -rho[np.asarray(T["cno"], dtype=np.int64)] * (
        np.asarray(T["R"], dtype=float) @ gvec
    )
    gravity_flux = np.asarray(T["cf_trans_g"] @ grav_half).ravel()
    rhs += np.asarray(T["e_div"] @ (M @ gravity_flux)).ravel()
    wells = list(wells or [])

    b_sub = np.flatnonzero(sb)                       # boundary sub-face indices
    face_of_subf = np.asarray(d1.argmax(axis=1)).ravel()
    bnd_face = face_of_subf[b_sub]
    prescribed_pressure = {} if bc_pressure is None else dict(bc_pressure)
    prescribed_flux = {}
    if bc is not None and bc.face.size:
        for face, kind, value in zip(bc.face, bc.type, bc.value):
            face = int(face)
            if kind == "pressure":
                prescribed_pressure[face] = float(value)
            elif kind == "flux":
                prescribed_flux[face] = float(value)
            else:
                raise ValueError(f"unsupported boundary condition type {kind!r}")

    boundary_faces = set(map(int, bnd_face))
    unknown_faces = (set(prescribed_pressure) | set(prescribed_flux)) - boundary_faces
    if unknown_faces:
        raise ValueError(f"boundary conditions refer to non-boundary faces {unknown_faces}")

    # Each face flux is distributed equally over its node subfaces, matching
    # incompMPFAlegacy's hh/counts convention. Unspecified faces stay no-flow.
    for face, value in prescribed_flux.items():
        rows = np.flatnonzero(bnd_face == face)
        rhs[nc + rows] = value / rows.size

    # MRST incompMPFAlegacy well block.  The direct kernel is single-phase:
    # WI is weighted by total mobility and dZ uses the supplied density.
    nbase = nc + n_bsub
    nw = len(wells)
    well_dp = []
    if nw:
        cell_diag = np.zeros(nbase)
        c_rows, c_cols, c_data = [], [], []
        d_well = np.zeros(nw)
        rhs = np.r_[rhs, np.zeros(nw)]
        norm_g = float(np.linalg.norm(gvec))
        for k, well in enumerate(wells):
            wc = np.asarray(well.cells, dtype=np.int64)
            if np.any(wc < 0) or np.any(wc >= nc):
                raise ValueError(f"well {k} refers to a nonexistent cell")
            wi = np.asarray(well.WI, dtype=float) * mob[wc]
            rho_w = float(np.average(rho[wc], weights=np.maximum(wi, 1e-300)))
            dp = norm_g * np.asarray(well.dZ, dtype=float) * rho_w
            well_dp.append(dp)
            np.add.at(cell_diag, wc, wi)
            if well.type == "bhp":
                d_well[k] = float(np.max(wi))
                rhs[nbase + k] += d_well[k] * well.val
                np.add.at(rhs, wc, wi * (well.val + dp))
            elif well.type == "rate":
                d_well[k] = float(np.sum(wi))
                rhs[nbase + k] += well.val - wi @ dp
                np.add.at(rhs, wc, wi * dp)
                c_rows.extend([k] * wc.size)
                c_cols.extend(wc.tolist())
                c_data.extend((-wi).tolist())
            else:
                raise ValueError(f"unsupported well type {well.type!r}")
        A = A + sp.diags_array(cell_diag, format="csc")
        C = sp.csc_array(
            (c_data, (c_rows, c_cols)), shape=(nw, nbase)
        )
        A = sp.bmat(
            [[A, C.T], [C, sp.diags_array(d_well, format="csc")]],
            format="csc",
        )

    dir_rows = np.array(
        [i for i, face in enumerate(bnd_face) if int(face) in prescribed_pressure],
        dtype=np.int64,
    )
    dir_values = np.array(
        [prescribed_pressure[int(bnd_face[i])] for i in dir_rows], dtype=float
    )
    fixed = nc + dir_rows
    free = np.setdiff1d(np.arange(nbase + nw), fixed)
    if fixed.size:
        rhs_free = rhs[free] - A[free][:, fixed] @ dir_values
        solution = np.empty(nbase + nw)
        solution[fixed] = dir_values
        solution[free] = spsolve(A[free][:, free].tocsc(), rhs_free)
    elif any(well.type == "bhp" for well in wells):
        solution = spsolve(A.tocsc(), rhs)
    else:
        # Pure Neumann problems determine pressure only up to a constant.
        # Impose zero mean cell pressure with a Lagrange multiplier without
        # discarding a conservation equation.
        constraint = np.zeros(nbase + nw)
        constraint[:nc] = 1.0 / nc
        augmented = sp.bmat(
            [[A, sp.csc_array(constraint[:, None])],
             [sp.csc_array(constraint[None, :]), None]],
            format="csc",
        )
        solution = spsolve(augmented, np.r_[rhs, 0.0])[:-1]

    subface_flux = M @ (T["cf_trans"] @ solution[:nbase] - gravity_flux)
    flux = np.asarray(d1.T @ subface_flux).ravel()
    internal = np.all(np.asarray(g.faces.neighbors) >= 0, axis=1)
    flux[internal] *= 0.5
    well_sol = []
    for k, well in enumerate(wells):
        wc = np.asarray(well.cells, dtype=np.int64)
        wi = np.asarray(well.WI, dtype=float) * mob[wc]
        bhp = float(solution[nbase + k])
        well_sol.append(WellSol(
            flux=wi * (bhp + well_dp[k] - solution[wc]),
            pressure=bhp,
            cdp=well_dp[k],
        ))
    result = MPFAResult(
        pressure=solution[:nc],
        flux=flux,
        boundary_pressure=solution[nc:nbase],
        subface_flux=np.asarray(subface_flux).ravel(),
        well_sol=well_sol,
    )
    return result if return_result else result.pressure


def incompMPFA(
    state,
    g,
    mpfa_t,
    fluid,
    *,
    useTensorAssembly=False,
    **options,
):
    """MRST-compatible public dispatcher.

    Both assembly variants feed the common conservative operator kernel, but
    the explicit dispatch preserves MRST's public/private call structure.
    """
    if useTensorAssembly:
        from pymrst.modules.mpfa.private.incomp_mpfa_tensor_assembly import (
            incomp_mpfa_tensor_assembly,
        )

        return incomp_mpfa_tensor_assembly(
            state, g, mpfa_t, fluid, **options
        )
    from pymrst.modules.mpfa.private.incomp_mpfa_legacy import (
        incomp_mpfa_legacy,
    )

    return incomp_mpfa_legacy(state, g, mpfa_t, fluid, **options)


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.mpfa.compute_multipoint_trans import compute_multipoint_trans
    from pymrst.params.rock import make_rock

    def _skewed_grid(nx=5, amplitude=0.25):
        """Smoothly twist interior nodes while leaving the boundary fixed."""
        length = float(nx)
        g = cart_grid([nx, nx], [length, length])
        if amplitude:
            c = np.asarray(g.nodes.coords, dtype=float)
            x, y = c[:, 0].copy(), c[:, 1].copy()
            c[:, 0] += amplitude * np.sin(np.pi*x/length) * np.sin(np.pi*y/length)
            c[:, 1] += 0.7*amplitude * np.sin(2*np.pi*x/length) * np.sin(np.pi*y/length)
            g.nodes.coords = c
        return compute_geometry(g)

    def _boundary_faces(g):
        neigh = np.asarray(g.faces.neighbors)
        return np.flatnonzero(np.any(neigh < 0, axis=1))

    def _linear(P, a=(3.0, 2.0, 0.0), b=5.0):
        return P @ np.asarray(a[:P.shape[1]]) + b

    def test_mpfa_reproduces_linear_field_on_skewed_full_tensor_grid():
        """The defining MPFA property: on a skewed grid with a full (off-diagonal)
        permeability tensor, the linear pressure field is recovered exactly."""
        g = _skewed_grid()
        perm = np.tile([2.0, 0.6, 1.0], (g.cells.num, 1)) * 1e-13   # kxx,kxy,kyy
        rock = make_rock(g, perm, 0.2)
        T = compute_multipoint_trans(g, rock)

        fc = np.asarray(g.faces.centroids)
        bc = {int(f): float(_linear(fc[f:f + 1])[0]) for f in _boundary_faces(g)}
        p = incomp_mpfa(g, T, bc)
        err = np.max(np.abs(p - _linear(np.asarray(g.cells.centroids))))
        assert err < 1e-9, err
        print(f"ok test_mpfa_reproduces_linear_field_on_skewed_full_tensor_grid "
              f"(max err {err:.2e})")

    def test_tpfa_fails_where_mpfa_succeeds():
        """The same case with a two-point (TPFA) approximation does *not* recover
        the linear field - the reason MPFA exists."""
        g = _skewed_grid()
        perm = np.tile([2.0, 0.6, 1.0], (g.cells.num, 1)) * 1e-13
        rock = make_rock(g, perm, 0.2)

        # a crude TPFA: half-transmissibilities from n·K·(x_f - x_c), harmonic mean
        from pymrst.solvers.compute_trans import compute_trans
        t_half = compute_trans(g, rock)
        neigh = np.asarray(g.faces.neighbors)
        internal = np.all(neigh >= 0, axis=1)
        cf = np.asarray(g.cells.faces, dtype=np.int64)
        face_pos = np.asarray(g.cells.facePos)
        cellno = np.repeat(np.arange(g.cells.num), np.diff(face_pos))
        tf = np.bincount(cf, weights=1.0 / t_half, minlength=g.faces.num)
        tface = np.zeros(g.faces.num)
        tface[internal] = 1.0 / tf[internal]
        fc = np.asarray(g.faces.centroids)

        nc = g.cells.num
        A = sp.lil_array((nc, nc))
        rhs = np.zeros(nc)
        bnd = _boundary_faces(g)
        for f in range(g.faces.num):
            c1, c2 = neigh[f]
            tt = tface[f]
            if c1 >= 0 and c2 >= 0:
                A[c1, c1] += tt; A[c1, c2] -= tt
                A[c2, c2] += tt; A[c2, c1] -= tt
            else:
                c = c1 if c1 >= 0 else c2
                # boundary half-trans, pressure BC
                cno_f = np.flatnonzero((cf == f) & (cellno == c))
                if cno_f.size:
                    th = t_half[cno_f[0]]
                    A[c, c] += th
                    rhs[c] += th * float(_linear(fc[f:f + 1])[0])
        p = spsolve(sp.csr_array(A), rhs)
        err = np.max(np.abs(p - _linear(np.asarray(g.cells.centroids))))
        assert err > 1e-3, f"TPFA unexpectedly exact ({err:.2e})"
        print(f"ok test_tpfa_fails_where_mpfa_succeeds (TPFA err {err:.2e} >> MPFA)")

    def test_uniform_pressure_gives_uniform_solution():
        g = _skewed_grid(3)
        rock = make_rock(g, 1e-13, 0.2)
        T = compute_multipoint_trans(g, rock)
        bc = {int(f): 100.0 for f in _boundary_faces(g)}
        p = incomp_mpfa(g, T, bc)
        assert np.allclose(p, 100.0, atol=1e-8), p
        print("ok test_uniform_pressure_gives_uniform_solution")

    test_mpfa_reproduces_linear_field_on_skewed_full_tensor_grid()
    test_tpfa_fails_where_mpfa_succeeds()
    test_uniform_pressure_gives_uniform_solution()
    print("\nAll tests passed!")
