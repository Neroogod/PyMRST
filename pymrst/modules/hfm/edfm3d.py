"""Planar 3D EDFM workflow helpers.

This is a focused 3D HFM slice for rectangular planar fractures embedded in a
3D matrix grid. It provides projection/intersection data, fracture cells, NNCs,
and a pressure solve using the same EDFMGrid surface as the 2D implementation.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.hfm.edfm2d import EDFMGrid, solve_incomp_edfm

TOL = 1e-10


@dataclass
class PlanarFracture:
    """Rectangular planar fracture in 3D."""

    center: np.ndarray
    normal: np.ndarray
    axis1: np.ndarray
    half_length1: float
    half_length2: float

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = np.asarray(self.normal, dtype=float)
        n = n / np.linalg.norm(n)
        a1 = np.asarray(self.axis1, dtype=float)
        a1 = a1 - n * float(a1 @ n)
        a1 = a1 / np.linalg.norm(a1)
        a2 = np.cross(n, a1)
        return n, a1, a2


@dataclass
class Projection3D:
    """Matrix-cell projection data for one planar fracture."""

    matrix_cells: np.ndarray
    fracture_cells: np.ndarray
    centroids: np.ndarray
    areas: np.ndarray
    distances: np.ndarray
    local_coordinates: np.ndarray


def _cell_bbox(g: Grid, cell: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = g.cells.nodes[g.cells.nodePos[cell]:g.cells.nodePos[cell + 1]]
    pts = g.nodes.coords[nodes, :3]
    return np.min(pts, axis=0), np.max(pts, axis=0)


def project_planar_fracture_to_grid(
    g: Grid,
    fracture: PlanarFracture,
    *,
    aperture: float,
) -> Projection3D:
    """Project a rectangular planar fracture onto matrix cells.

    The routine is exact for grid-aligned rectangular fractures and conservative
    for mildly skew planes. Cells are selected by projecting their centroids
    into the fracture frame and requiring the plane to pass through their
    bounding box.
    """
    if g.griddim != 3:
        raise ValueError("3D EDFM projection requires a 3D grid")
    n, a1, a2 = fracture.frame()
    center = np.asarray(fracture.center, dtype=float)
    cells, centroids, areas, distances, local = [], [], [], [], []
    for c in range(g.cells.num):
        lo, hi = _cell_bbox(g, c)
        corners = np.array([
            [lo[0], lo[1], lo[2]], [lo[0], lo[1], hi[2]],
            [lo[0], hi[1], lo[2]], [lo[0], hi[1], hi[2]],
            [hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], lo[2]], [hi[0], hi[1], hi[2]],
        ])
        signed = (corners - center) @ n
        if np.min(signed) > TOL or np.max(signed) < -TOL:
            continue
        # If the fracture lies exactly on a matrix face, MRST's pEDFM path
        # assigns the connection to one side. Pick the positive side so the
        # fracture area is not double-counted.
        cell_signed = float((g.cells.centroids[c, :3] - center) @ n)
        if abs(np.min(signed)) <= TOL and abs(np.max(signed)) <= TOL:
            continue
        if abs(np.min(np.abs(signed))) <= TOL and cell_signed < -TOL:
            continue
        x = g.cells.centroids[c, :3] - center
        xi = float(x @ a1)
        eta = float(x @ a2)
        corner_local = (corners - center) @ np.column_stack([a1, a2])
        lo1 = max(float(np.min(corner_local[:, 0])), -fracture.half_length1)
        hi1 = min(float(np.max(corner_local[:, 0])), fracture.half_length1)
        lo2 = max(float(np.min(corner_local[:, 1])), -fracture.half_length2)
        hi2 = min(float(np.max(corner_local[:, 1])), fracture.half_length2)
        if hi1 <= lo1 + TOL or hi2 <= lo2 + TOL:
            continue
        area = float((hi1 - lo1) * (hi2 - lo2))
        d = max(float(np.mean(np.abs(signed))), aperture / 4.0)
        cells.append(c)
        xi_c = 0.5 * (lo1 + hi1)
        eta_c = 0.5 * (lo2 + hi2)
        centroids.append(center + xi_c * a1 + eta_c * a2)
        areas.append(area)
        distances.append(d)
        local.append([xi_c, eta_c])
    fcells = np.arange(len(cells), dtype=np.int64)
    return Projection3D(
        matrix_cells=np.asarray(cells, dtype=np.int64),
        fracture_cells=fcells,
        centroids=np.asarray(centroids, dtype=float).reshape(-1, 3),
        areas=np.asarray(areas, dtype=float),
        distances=np.asarray(distances, dtype=float),
        local_coordinates=np.asarray(local, dtype=float).reshape(-1, 2),
    )


def build_edfm_3d(
    g: Grid,
    fractures: Sequence[PlanarFracture],
    *,
    aperture: float,
    perm_matrix: np.ndarray,
    perm_frac: float,
    poro_matrix: np.ndarray,
    poro_frac: float = 1.0,
) -> EDFMGrid:
    """Build a 3D EDFM grid for planar rectangular fractures."""
    perm_matrix = np.asarray(perm_matrix, dtype=float).ravel()
    poro_matrix = np.asarray(poro_matrix, dtype=float).ravel()
    nm = g.cells.num
    projections = [
        project_planar_fracture_to_grid(g, f, aperture=aperture)
        for f in fractures
    ]
    nf = sum(p.matrix_cells.size for p in projections)
    frac_vol = np.zeros(nf)
    frac_cen = np.zeros((nf, 3))
    frac_line = np.zeros(nf, dtype=np.int64)
    conn, trans, ci, ctype = [], [], [], []
    offset = nm
    for i, proj in enumerate(projections):
        for k, m in enumerate(proj.matrix_cells):
            fcell = offset + k
            area = proj.areas[k]
            frac_vol[fcell - nm] = area * aperture
            frac_cen[fcell - nm] = proj.centroids[k]
            frac_line[fcell - nm] = i
            ci_mf = area / max(proj.distances[k], TOL)
            pv_m = poro_matrix[m] * g.cells.volumes[m]
            pv_f = poro_frac * frac_vol[fcell - nm]
            w1 = pv_m / perm_matrix[m]
            w2 = pv_f / perm_frac
            conn.append((int(m), int(fcell)))
            trans.append(ci_mf * (pv_m + pv_f) / (w1 + w2))
            ci.append(ci_mf)
            ctype.append("frac_matrix")

        # Projection-neighbor fracture connections in local coordinates.
        coords = proj.local_coordinates
        for a in range(coords.shape[0]):
            delta = coords - coords[a]
            dist = np.linalg.norm(delta, axis=1)
            order = np.argsort(dist)
            for b in order[1:3]:
                if b <= a or dist[b] <= TOL:
                    continue
                c1, c2 = offset + a, offset + b
                area = 0.5 * (proj.areas[a] + proj.areas[b])
                t = perm_frac * aperture * np.sqrt(area) / dist[b]
                conn.append((int(c1), int(c2)))
                trans.append(t)
                ci.append(0.0)
                ctype.append("frac")
        offset += proj.matrix_cells.size

    # Matrix-matrix connections use a simple TPFA transmissibility.
    internal = np.all(g.faces.neighbors >= 0, axis=1)
    for f in np.flatnonzero(internal):
        a, b = g.faces.neighbors[f]
        d = np.linalg.norm(g.cells.centroids[a, :3] - g.cells.centroids[b, :3])
        k_h = 2.0 / (1.0 / perm_matrix[a] + 1.0 / perm_matrix[b])
        conn.append((int(a), int(b)))
        trans.append(k_h * g.faces.areas[f] / max(d, TOL))
        ci.append(0.0)
        ctype.append("matrix")

    return EDFMGrid(
        g=g,
        n_matrix=nm,
        frac_cells=nf,
        frac_volumes=frac_vol,
        frac_centroids=frac_cen,
        frac_line=frac_line,
        perm=np.concatenate([perm_matrix, np.full(nf, perm_frac)]),
        poro=np.concatenate([poro_matrix, np.full(nf, poro_frac)]),
        conn=np.asarray(conn, dtype=np.int64).reshape(-1, 2),
        trans=np.asarray(trans, dtype=float),
        ci=np.asarray(ci, dtype=float),
        conn_type=ctype,
    )


def solve_incomp_edfm_3d(*args, **kwargs):
    """Alias for the generic EDFM pressure solve on a 3D EDFM grid."""
    return solve_incomp_edfm(*args, **kwargs)
