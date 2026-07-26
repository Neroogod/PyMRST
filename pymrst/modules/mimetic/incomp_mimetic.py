"""Conservative incompressible mimetic solve, ported from ``incompMimetic.m``."""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pymrst.solvers.state_types import WellSol


@dataclass
class MimeticResult:
    pressure: np.ndarray
    face_pressure: np.ndarray
    flux: np.ndarray
    half_flux: np.ndarray
    well_sol: list[WellSol] = field(default_factory=list)
    A: sp.csr_array | None = None


def _source_vector(g, src):
    values = np.zeros(g.cells.num)
    if src is not None:
        np.add.at(
            values,
            np.asarray(src.cell, dtype=np.int64),
            np.asarray(src.rate, dtype=float),
        )
    return values


def incomp_mimetic(
    g, system, *, bc=None, src=None, mobility=1.0,
    density=0.0, gravity=None, wells=None,
    matrix_output=False, return_result=True,
):
    """Solve the hybrid mimetic pressure/face-pressure system.

    Flux unknowns are eliminated cellwise with ``q = BI(Cp-Dlambda)``.
    Cell rows enforce source balance and face rows enforce flux continuity,
    prescribed boundary flux, or prescribed interface pressure.
    """
    if system.BI is None:
        if system.B is None:
            raise ValueError("mimetic system contains neither B nor BI")
        blocks = []
        for cell in range(g.cells.num):
            start, stop = map(int, g.cells.facePos[cell:cell + 2])
            blocks.append(np.linalg.inv(system.B[start:stop, start:stop].toarray()))
        base_bi = sp.block_diag(blocks, format="csr")
    else:
        base_bi = system.BI

    nhf = len(g.cells.faces)
    nc, nf = g.cells.num, g.faces.num
    cellno = np.repeat(np.arange(nc), np.diff(g.cells.facePos))
    faces = np.asarray(g.cells.faces, dtype=np.int64)
    C = sp.csr_array(
        (np.ones(nhf), (np.arange(nhf), cellno)), shape=(nhf, nc)
    )
    D = sp.csr_array(
        (np.ones(nhf), (np.arange(nhf), faces)), shape=(nhf, nf)
    )
    mob = np.asarray(mobility, dtype=float).ravel()
    if mob.size == 1:
        mob = np.full(nc, mob[0])
    if mob.size != nc:
        raise ValueError("mobility must be scalar or one value per cell")
    BI = sp.diags_array(mob[cellno]) @ base_bi
    rho = np.asarray(density, dtype=float).ravel()
    if rho.size == 1:
        rho = np.full(nc, rho[0])
    if rho.size != nc:
        raise ValueError("density must be scalar or one value per cell")
    gravity_vector = (
        np.zeros(g.griddim)
        if gravity is None else np.asarray(gravity, dtype=float).ravel()
    )
    if gravity_vector.size != g.griddim:
        raise ValueError("gravity must have one component per grid dimension")
    centroid_vectors = (
        np.asarray(g.faces.centroids)[faces]
        - np.asarray(g.cells.centroids)[cellno]
    )
    gravity_half = rho[cellno]*(centroid_vectors @ gravity_vector)

    A = sp.bmat(
        [[C.T @ BI @ C, -C.T @ BI @ D],
         [-D.T @ BI @ C, D.T @ BI @ D]],
        format="lil",
    )
    gravity_flux = np.asarray(BI @ gravity_half).ravel()
    rhs = np.r_[
        _source_vector(g, src) - np.asarray(C.T @ gravity_flux).ravel(),
        np.asarray(D.T @ gravity_flux).ravel(),
    ]
    pressure_bc = {}
    flux_bc = {}
    if bc is not None:
        for face, kind, value in zip(bc.face, bc.type, bc.value):
            target = pressure_bc if kind == "pressure" else flux_bc
            target[int(face)] = float(value)

    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    boundary = np.any(neighbors < 0, axis=1)
    for face in np.flatnonzero(boundary):
        row = nc + int(face)
        if int(face) in pressure_bc:
            A.rows[row] = [row]
            A.data[row] = [1.0]
            rhs[row] = pressure_bc[int(face)]
        else:
            # The symmetric face equation is minus outward half-face flux.
            rhs[row] = -flux_bc.get(int(face), 0.0)

    wells = list(wells or [])
    well_pressure_drop = []
    if wells:
        nw = len(wells)
        A.resize((nc + nf + nw, nc + nf + nw))
        rhs = np.r_[rhs, np.zeros(nw)]
        gravity_magnitude = float(np.linalg.norm(gravity_vector))
        for index, well in enumerate(wells):
            cells = np.asarray(well.cells, dtype=np.int64)
            if np.any(cells < 0) or np.any(cells >= nc):
                raise ValueError(f"well {index} refers to a nonexistent cell")
            wi = np.asarray(well.WI, dtype=float).ravel()*mob[cells]
            if wi.size != cells.size:
                raise ValueError("well WI must contain one value per completion")
            weighted_density = float(np.average(
                rho[cells], weights=np.maximum(wi, np.finfo(float).tiny)
            ))
            dp = (
                gravity_magnitude*np.asarray(well.dZ, dtype=float).ravel()
                * weighted_density
            )
            well_pressure_drop.append(dp)
            well_column = nc + nf + index
            for cell, transmissibility, drop in zip(cells, wi, dp):
                A[cell, cell] += transmissibility
                A[cell, well_column] -= transmissibility
                rhs[cell] += transmissibility*drop
            if well.type == "rate":
                for cell, transmissibility in zip(cells, wi):
                    A[well_column, cell] -= transmissibility
                A[well_column, well_column] = float(np.sum(wi))
                rhs[well_column] = float(well.val) - float(wi @ dp)
            elif well.type == "bhp":
                A.rows[well_column] = [well_column]
                A.data[well_column] = [1.0]
                rhs[well_column] = float(well.val)
            else:
                raise ValueError(f"unsupported well type {well.type!r}")

    A = A.tocsr()
    anchored = bool(pressure_bc) or any(w.type == "bhp" for w in wells)
    if not anchored:
        # Pure Neumann: fix the otherwise free pressure constant without
        # deleting a local conservation row.
        constraint = np.r_[
            np.full(nc, 1.0/nc), np.zeros(nf + len(wells))
        ]
        augmented = sp.bmat(
            [[A, sp.csr_array(constraint[:, None])],
             [sp.csr_array(constraint[None, :]), None]],
            format="csr",
        )
        solution = spsolve(augmented.tocsc(), np.r_[rhs, 0.0])[:-1]
    else:
        solution = spsolve(A.tocsc(), rhs)

    pressure = solution[:nc]
    face_pressure = solution[nc:]
    face_pressure = face_pressure[:nf]
    half_flux = np.asarray(
        BI @ (C @ pressure - D @ face_pressure + gravity_half)
    ).ravel()
    flux = np.zeros(nf)
    for face in range(nf):
        rows = np.flatnonzero(faces == face)
        n0, _ = neighbors[face]
        if n0 >= 0:
            row = rows[cellno[rows] == n0][0]
            flux[face] = half_flux[row]
        else:
            flux[face] = -half_flux[rows[0]]
    well_sol = []
    for index, well in enumerate(wells):
        cells = np.asarray(well.cells, dtype=np.int64)
        wi = np.asarray(well.WI, dtype=float).ravel()*mob[cells]
        bhp = float(solution[nc + nf + index])
        well_sol.append(WellSol(
            flux=wi*(bhp + well_pressure_drop[index] - pressure[cells]),
            pressure=bhp,
            cdp=well_pressure_drop[index],
        ))
    result = MimeticResult(
        pressure=pressure,
        face_pressure=face_pressure,
        flux=flux,
        half_flux=half_flux,
        well_sol=well_sol,
        A=A if matrix_output else None,
    )
    return result if return_result else result.pressure


def incompMimetic(state, g, system, fluid=None, **options):
    """MRST-compatible dispatcher updating a state-like object when supplied."""
    del fluid
    result = incomp_mimetic(g, system, **options)
    if state is None:
        return result
    state.pressure = result.pressure
    state.facePressure = result.face_pressure
    state.flux = result.flux
    state.wellSol = result.well_sol
    if result.A is not None:
        state.A = result.A
    return state


__all__ = ["MimeticResult", "incomp_mimetic", "incompMimetic"]
