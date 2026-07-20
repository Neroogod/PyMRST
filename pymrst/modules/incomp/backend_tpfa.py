"""Experimental backend-aware incompressible TPFA pressure solve.

This is the first PyMRST GPU/tensor execution slice. It intentionally covers a
small, validation-friendly subset of ``incomp_tpfa``: single-phase total
mobility, pressure/flux boundary conditions, explicit cell sources, standard
rate/BHP wells, and gravity.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp

from pymrst.backends import ArrayBackend, get_backend
from pymrst.gridprocessing.grid import Grid
from pymrst.params.gravity_const import gravity_or_zero
from pymrst.params.wells_and_bc import BoundaryCondition, Source, Well


def incomp_tpfa_backend_pressure(
    g: Grid,
    t_half: np.ndarray,
    *,
    totmob,
    bc: Optional[BoundaryCondition] = None,
    src: Optional[Source] = None,
    wells: Optional[Sequence[Well]] = None,
    omega=None,
    gravity: Optional[np.ndarray] = None,
    backend: str | ArrayBackend = "numpy",
    device: str | None = None,
    dtype: str = "float64",
    linear_solver: str = "auto",
    return_wells: bool = False,
):
    """Solve the pressure equation using the selected array backend.

    Returns ``(pressure, flux)`` as backend arrays. With ``return_wells=True``,
    returns ``(pressure, flux, well_data)`` where ``well_data`` contains BHPs
    and perforation fluxes. Convert with ``get_backend(...).to_numpy`` when
    NumPy arrays are needed.
    """
    be = get_backend(backend, device=device, dtype=dtype, linear_solver=linear_solver)
    a, rhs, meta = assemble_backend_tpfa_matrix(
        g,
        t_half,
        totmob,
        bc=bc,
        src=src,
        wells=wells,
        omega=omega,
        gravity=gravity,
        dtype=dtype,
    )
    p = be.solve(a, rhs)
    p_cells = p[: g.cells.num]
    flux = _reconstruct_flux(
        g,
        meta["ft"],
        p_cells,
        meta["d_face"],
        meta["d_val"],
        meta["fg"],
        meta["fg_boundary"],
        be,
    )
    if return_wells:
        return p_cells, flux, _reconstruct_wells(g, wells or [], meta, p, be)
    return p_cells, flux


def assemble_backend_tpfa_matrix(
    g: Grid,
    t_half: np.ndarray,
    totmob,
    *,
    bc: Optional[BoundaryCondition] = None,
    src: Optional[Source] = None,
    wells: Optional[Sequence[Well]] = None,
    omega=None,
    gravity: Optional[np.ndarray] = None,
    dtype: str = "float64",
):
    """Assemble the TPFA linear system for the experimental backend solver."""
    neighbors = g.faces.neighbors
    internal = np.all(neighbors >= 0, axis=1)
    cell_no = g.cell_no()
    cf = g.cells.faces
    wells = list(wells or [])
    nf, nc, nw = g.faces.num, g.cells.num, len(wells)

    totmob = np.asarray(totmob, dtype=dtype)
    if totmob.size == 1:
        totmob = np.full(nc, float(totmob), dtype=dtype)
    omega = np.asarray(omega if omega is not None else 0.0, dtype=dtype)
    if omega.size == 1:
        omega = np.full(nc, float(omega), dtype=dtype)
    g_vec = gravity_or_zero(gravity, g.griddim)
    grav_on = np.linalg.norm(g_vec) > 0
    t_hf = np.asarray(t_half, dtype=dtype) * totmob[cell_no]
    with np.errstate(divide="ignore"):
        ft = 1.0 / np.bincount(cf, weights=1.0 / t_hf, minlength=nf)

    rhs = np.zeros(nc + nw, dtype=dtype)
    d_face = np.zeros(nf, dtype=bool)
    d_val = np.zeros(nf, dtype=dtype)
    h_flux = np.zeros(nf, dtype=dtype)

    if src is not None and src.cell.size > 0:
        rhs[:nc] += np.bincount(src.cell, weights=src.rate, minlength=nc)

    if bc is not None and bc.face.size > 0:
        types = np.asarray(bc.type)
        is_pressure = types == "pressure"
        d_face[bc.face[is_pressure]] = True
        d_val[bc.face[is_pressure]] = bc.value[is_pressure]
        is_flux = types == "flux"
        h_flux[bc.face[is_flux]] = bc.value[is_flux]

    grav_hf = _gravity_half_faces(g, omega, g_vec, grav_on, cell_no, cf, dtype)
    sgn = np.where(neighbors[cf, 0] == cell_no, 1.0, -1.0)
    j = internal[cf] | d_face[cf]
    fg = np.bincount(cf[j], weights=grav_hf[j] * sgn[j], minlength=nf)
    fg_boundary = np.bincount(cf, weights=grav_hf, minlength=nf)
    rhs[:nc] += np.bincount(
        cell_no,
        weights=-ft[cf] * (sgn * fg[cf]),
        minlength=nc,
    )

    # Interior connections.
    ni = neighbors[internal]
    d = np.bincount(ni.ravel(order="F"), weights=np.tile(ft[internal], 2), minlength=nc)
    rows = [ni[:, 0], ni[:, 1]]
    cols = [ni[:, 1], ni[:, 0]]
    vals = [-ft[internal], -ft[internal]]

    # Dirichlet boundary half-faces.
    hf_dirichlet = d_face[cf]
    if np.any(hf_dirichlet):
        t_dir = t_hf[hf_dirichlet]
        c_dir = cell_no[hf_dirichlet]
        f_dir = cf[hf_dirichlet]
        d += np.bincount(c_dir, weights=t_dir, minlength=nc)
        rhs[:nc] += np.bincount(c_dir, weights=t_dir * d_val[f_dir], minlength=nc)

    # Neumann boundary fluxes: positive flux enters the reservoir.
    if np.any(h_flux):
        boundary_flux_faces = np.flatnonzero(h_flux != 0.0)
        bcell = _boundary_cells(g, boundary_flux_faces)
        rhs[:nc] += np.bincount(bcell, weights=h_flux[boundary_flux_faces], minlength=nc)

    c_rows, c_cols, c_vals = [], [], []
    d_well = np.zeros(nw, dtype=dtype)
    well_wi = []
    well_cdp = []
    for k, w in enumerate(wells):
        wc = np.asarray(w.cells, dtype=np.int64)
        wi = np.asarray(w.WI, dtype=dtype) * totmob[wc]
        cdp = _well_gravity_drop(g, w, omega, g_vec, grav_on, dtype)
        well_wi.append(wi)
        well_cdp.append(cdp)
        d[wc] += wi
        if w.type == "bhp":
            ww = float(np.max(wi))
            np.add.at(rhs, wc, wi * (float(w.val) + cdp))
            rhs[nc + k] += ww * float(w.val)
            d_well[k] = ww
        elif w.type == "rate":
            rhs[nc + k] += float(w.val)
            np.add.at(rhs, wc, wi * cdp)
            c_rows.extend([k] * wc.size)
            c_cols.extend(wc.tolist())
            c_vals.extend((-wi).tolist())
            d_well[k] = float(np.sum(wi))
            rhs[nc + k] -= float(wi @ cdp)
        else:
            raise ValueError(f"unsupported well type {w.type!r}")

    rows.append(np.arange(nc))
    cols.append(np.arange(nc))
    vals.append(d)
    a = sp.coo_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nc, nc),
    ).tocsr()

    if nw > 0:
        c = sp.coo_array((c_vals, (c_rows, c_cols)), shape=(nw, nc)).tocsr()
        a = sp.block_array(
            [[a, c.T], [c, sp.diags_array(d_well, format="csr")]], format="csr"
        )

    if not np.any(d_face) and not any(w.type == "bhp" for w in wells):
        a = a.tolil()
        pivot = int(np.argmax(a.diagonal()))
        a[pivot, pivot] = 2.0 * a[pivot, pivot] if a[pivot, pivot] else 1.0
        a = a.tocsr()

    return a, rhs, {
        "ft": ft,
        "d_face": d_face,
        "d_val": d_val,
        "fg": fg,
        "fg_boundary": fg_boundary,
        "totmob": totmob,
        "well_wi": well_wi,
        "well_cdp": well_cdp,
    }


def _reconstruct_flux(
    g: Grid,
    ft: np.ndarray,
    pressure,
    d_face,
    d_val,
    fg,
    fg_boundary,
    be: ArrayBackend,
):
    p = pressure
    neighbors = g.faces.neighbors
    internal = np.all(neighbors >= 0, axis=1)
    flux = be.zeros(g.faces.num)
    ni = neighbors[internal]
    if be.is_torch:
        idx = be.as_int(np.flatnonzero(internal))
        n0 = be.as_int(ni[:, 0])
        n1 = be.as_int(ni[:, 1])
        flux[idx] = -be.asarray(ft[internal]) * (p[n1] - p[n0] - be.asarray(fg[internal]))
        bfaces = np.flatnonzero(d_face & ~internal)
        if bfaces.size:
            bcells = be.as_int(_boundary_cells(g, bfaces))
            bsgn = be.asarray(_boundary_sign(g, bfaces))
            flux[be.as_int(bfaces)] = -bsgn * be.asarray(ft[bfaces]) * (
                be.asarray(d_val[bfaces]) - p[bcells] - be.asarray(fg_boundary[bfaces])
            )
    else:
        flux[internal] = -ft[internal] * (p[ni[:, 1]] - p[ni[:, 0]] - fg[internal])
        bfaces = np.flatnonzero(d_face & ~internal)
        if bfaces.size:
            bcells = _boundary_cells(g, bfaces)
            flux[bfaces] = -_boundary_sign(g, bfaces) * ft[bfaces] * (
                d_val[bfaces] - p[bcells] - fg_boundary[bfaces]
            )
    return flux


def _boundary_cells(g: Grid, faces: np.ndarray) -> np.ndarray:
    n = g.faces.neighbors[np.asarray(faces, dtype=np.int64)]
    return np.where(n[:, 0] >= 0, n[:, 0], n[:, 1])


def _boundary_sign(g: Grid, faces: np.ndarray) -> np.ndarray:
    n = g.faces.neighbors[np.asarray(faces, dtype=np.int64)]
    return np.where(n[:, 1] == -1, 1.0, -1.0)


def _gravity_half_faces(g, omega, g_vec, grav_on, cell_no, cf, dtype):
    if not grav_on:
        return np.zeros(cf.size, dtype=dtype)
    cvec = g.faces.centroids[cf] - g.cells.centroids[cell_no]
    return omega[cell_no] * (cvec @ g_vec)


def _well_gravity_drop(g, w: Well, omega, g_vec, grav_on, dtype):
    if not grav_on:
        return np.zeros(np.asarray(w.cells).size, dtype=dtype)
    wc = np.asarray(w.cells, dtype=np.int64)
    dz = np.asarray(w.dZ, dtype=dtype)
    gnorm = float(np.linalg.norm(g_vec))
    return omega[wc] * gnorm * dz


def _reconstruct_wells(
    g: Grid,
    wells: Sequence[Well],
    meta,
    pressure,
    be: ArrayBackend,
) -> dict[str, object]:
    nc = g.cells.num
    bhp = pressure[nc:nc + len(wells)]
    perf_flux = []
    for k, w in enumerate(wells):
        wc = np.asarray(w.cells, dtype=np.int64)
        if be.is_torch:
            flux = be.asarray(meta["well_wi"][k]) * (
                bhp[k] + be.asarray(meta["well_cdp"][k]) - pressure[be.as_int(wc)]
            )
        else:
            flux = meta["well_wi"][k] * (bhp[k] + meta["well_cdp"][k] - pressure[wc])
        perf_flux.append(flux)
    return {"bhp": bhp, "flux": perf_flux}
