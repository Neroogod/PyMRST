"""Planar 3D EDFM workflow helpers.

This is a focused 3D HFM slice for rectangular planar fractures embedded in a
3D matrix grid. It provides projection/intersection data, fracture cells, NNCs,
and a pressure solve using the same EDFMGrid surface as the 2D implementation.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import ConvexHull

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
    polygon_points: np.ndarray | None = None

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = np.asarray(self.normal, dtype=float)
        n = n / np.linalg.norm(n)
        a1 = np.asarray(self.axis1, dtype=float)
        a1 = a1 - n * float(a1 @ n)
        a1 = a1 / np.linalg.norm(a1)
        a2 = np.cross(n, a1)
        return n, a1, a2

    def local_polygon(self) -> np.ndarray:
        """Fracture boundary in its local, counter-clockwise plane frame."""
        _, a1, a2 = self.frame()
        if self.polygon_points is None:
            return np.array([
                [-self.half_length1, -self.half_length2],
                [ self.half_length1, -self.half_length2],
                [ self.half_length1,  self.half_length2],
                [-self.half_length1,  self.half_length2],
            ])
        points = np.asarray(self.polygon_points, dtype=float)
        local = (points - np.asarray(self.center, dtype=float)) @ np.column_stack([a1, a2])
        hull = ConvexHull(local)
        if hull.volume <= TOL:
            raise ValueError("fracture polygon has zero area")
        # A convex owner is intentional: MRST's shale path computes the target
        # with convexpolygonarea before generating fracture cells.
        if hull.vertices.size != points.shape[0]:
            raise ValueError("fracture polygon must be convex and non-redundant")
        return local[hull.vertices]


@dataclass
class Projection3D:
    """Matrix-cell projection data for one planar fracture."""

    matrix_cells: np.ndarray
    fracture_cells: np.ndarray
    centroids: np.ndarray
    areas: np.ndarray
    distances: np.ndarray
    local_coordinates: np.ndarray
    polygons: tuple[np.ndarray, ...] = ()
    boundary: np.ndarray | None = None


def _cell_vertices_edges(g: Grid, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """Return local vertices and unique topological edges for one grid cell."""
    faces = np.asarray(
        g.cells.faces[g.cells.facePos[cell]:g.cells.facePos[cell + 1]],
        dtype=np.int64,
    ).reshape(-1)
    face_nodes = [
        np.asarray(g.faces.nodes[g.faces.nodePos[face]:g.faces.nodePos[face + 1]], dtype=np.int64)
        for face in faces
    ]
    global_nodes = np.unique(np.concatenate(face_nodes))
    local_index = {int(node): index for index, node in enumerate(global_nodes)}
    edges = set()
    for nodes in face_nodes:
        for index, first in enumerate(nodes):
            second = nodes[(index + 1) % nodes.size]
            pair = tuple(sorted((local_index[int(first)], local_index[int(second)])))
            edges.add(pair)
    return (
        np.asarray(g.nodes.coords[global_nodes, :3], dtype=float),
        np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2),
    )


def project_planar_fracture_to_grid(
    g: Grid,
    fracture: PlanarFracture,
    *,
    aperture: float,
) -> Projection3D:
    """Project a convex planar fracture onto convex polyhedral matrix cells."""
    if g.griddim != 3:
        raise ValueError("3D EDFM projection requires a 3D grid")
    n, a1, a2 = fracture.frame()
    center = np.asarray(fracture.center, dtype=float)
    fracture_polygon = fracture.local_polygon()
    cells, centroids, areas, distances, local, polygons, boundary = [], [], [], [], [], [], []
    for c in range(g.cells.num):
        vertices, edges = _cell_vertices_edges(g, c)
        signed = (vertices - center) @ n
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
        is_boundary = (
            np.min(np.abs(signed)) <= TOL
            and not (np.min(signed) < -TOL and np.max(signed) > TOL)
        )
        section = _plane_polyhedron_section(vertices, edges, signed)
        if section.shape[0] < 3:
            continue
        section_local = (section - center) @ np.column_stack([a1, a2])
        cell_polygon = section_local[ConvexHull(section_local).vertices]
        clipped = _clean_polygon(_clip_convex_polygon(fracture_polygon, cell_polygon))
        area, polygon_centroid = _polygon_area_centroid(clipped)
        if area <= TOL:
            continue
        d = max(float(np.mean(np.abs(signed))), aperture / 4.0)
        cells.append(c)
        xi_c, eta_c = polygon_centroid
        centroids.append(center + xi_c * a1 + eta_c * a2)
        areas.append(area)
        distances.append(d)
        local.append([xi_c, eta_c])
        polygons.append(center + clipped[:, :1] * a1 + clipped[:, 1:] * a2)
        boundary.append(is_boundary)
    fcells = np.arange(len(cells), dtype=np.int64)
    return Projection3D(
        matrix_cells=np.asarray(cells, dtype=np.int64),
        fracture_cells=fcells,
        centroids=np.asarray(centroids, dtype=float).reshape(-1, 3),
        areas=np.asarray(areas, dtype=float),
        distances=np.asarray(distances, dtype=float),
        local_coordinates=np.asarray(local, dtype=float).reshape(-1, 2),
        polygons=tuple(polygons),
        boundary=np.asarray(boundary, dtype=bool),
    )


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _plane_polyhedron_section(
    vertices: np.ndarray, edges: np.ndarray, signed: np.ndarray
) -> np.ndarray:
    """Return unique points where a plane cuts a cell's actual edges."""
    points = []
    for first, second in edges:
        d1, d2 = signed[first], signed[second]
        p1, p2 = vertices[first], vertices[second]
        if abs(d1) <= TOL:
            points.append(p1)
        if abs(d2) <= TOL:
            points.append(p2)
        if d1 * d2 < -(TOL**2):
            fraction = d1 / (d1 - d2)
            points.append(p1 + fraction * (p2 - p1))
    if not points:
        return np.empty((0, 3))
    points = np.asarray(points)
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    rounded = np.round(points / (TOL * scale)).astype(np.int64)
    _, keep = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(keep)]


def _clip_convex_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping for two counter-clockwise convex polygons."""
    output = np.asarray(subject, dtype=float)
    for index in range(len(clip)):
        edge_start = clip[index]
        edge_end = clip[(index + 1) % len(clip)]
        edge = edge_end - edge_start
        source = output
        if source.size == 0:
            break
        result = []
        previous = source[-1]
        previous_inside = _cross2(edge, previous - edge_start) >= -TOL
        for current in source:
            current_inside = _cross2(edge, current - edge_start) >= -TOL
            if current_inside != previous_inside:
                segment = current - previous
                denominator = _cross2(segment, edge)
                if abs(denominator) > TOL:
                    fraction = _cross2(edge_start - previous, edge) / denominator
                    result.append(previous + fraction * segment)
            if current_inside:
                result.append(current)
            previous = current
            previous_inside = current_inside
        output = np.asarray(result, dtype=float).reshape(-1, 2)
    return output


def _polygon_area_centroid(points: np.ndarray) -> tuple[float, np.ndarray]:
    if len(points) < 3:
        return 0.0, np.zeros(2)
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    signed_area = 0.5 * np.sum(cross)
    if abs(signed_area) <= TOL:
        return 0.0, np.zeros(2)
    centroid = np.sum((points + following) * cross[:, None], axis=0) / (6 * signed_area)
    return abs(float(signed_area)), centroid


def _clean_polygon(points: np.ndarray) -> np.ndarray:
    """Remove duplicate consecutive vertices introduced by boundary clipping."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        return points
    keep = [0]
    for index in range(1, len(points)):
        if np.linalg.norm(points[index] - points[keep[-1]]) > TOL:
            keep.append(index)
    cleaned = points[keep]
    if len(cleaned) > 1 and np.linalg.norm(cleaned[-1] - cleaned[0]) <= TOL:
        cleaned = cleaned[:-1]
    return cleaned


def build_edfm_3d(
    g: Grid,
    fractures: Sequence[PlanarFracture],
    *,
    aperture: float | Sequence[float],
    perm_matrix: np.ndarray,
    perm_frac: float | Sequence[float],
    poro_matrix: np.ndarray,
    poro_frac: float | Sequence[float] = 1.0,
) -> EDFMGrid:
    """Build a 3D EDFM grid for planar rectangular fractures."""
    perm_matrix = np.asarray(perm_matrix, dtype=float).ravel()
    poro_matrix = np.asarray(poro_matrix, dtype=float).ravel()
    nm = g.cells.num
    count = len(fractures)

    def per_fracture(values, name):
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 1:
            return np.full(count, array.item())
        if array.size != count:
            raise ValueError(f"{name} must be scalar or have one value per fracture")
        return array

    apertures = per_fracture(aperture, "aperture")
    frac_permeabilities = per_fracture(perm_frac, "perm_frac")
    frac_porosities = per_fracture(poro_frac, "poro_frac")
    if np.any(apertures <= 0) or np.any(frac_permeabilities <= 0):
        raise ValueError("fracture aperture and permeability must be positive")
    if np.any((frac_porosities < 0) | (frac_porosities > 1)):
        raise ValueError("fracture porosity must lie in [0, 1]")
    projections = [
        project_planar_fracture_to_grid(g, f, aperture=apertures[i])
        for i, f in enumerate(fractures)
    ]
    nf = sum(p.matrix_cells.size for p in projections)
    frac_vol = np.zeros(nf)
    frac_cen = np.zeros((nf, 3))
    frac_line = np.zeros(nf, dtype=np.int64)
    frac_perm = np.zeros(nf)
    frac_poro = np.zeros(nf)
    conn, trans, ci, ctype = [], [], [], []
    offset = nm
    for i, proj in enumerate(projections):
        aperture_i = apertures[i]
        perm_frac_i = frac_permeabilities[i]
        poro_frac_i = frac_porosities[i]
        for k, m in enumerate(proj.matrix_cells):
            fcell = offset + k
            area = proj.areas[k]
            frac_vol[fcell - nm] = area * aperture_i
            frac_cen[fcell - nm] = proj.centroids[k]
            frac_line[fcell - nm] = i
            frac_perm[fcell - nm] = perm_frac_i
            frac_poro[fcell - nm] = poro_frac_i
            ci_mf = area / max(proj.distances[k], TOL)
            pv_m = poro_matrix[m] * g.cells.volumes[m]
            pv_f = poro_frac_i * frac_vol[fcell - nm]
            w1 = pv_m / perm_matrix[m]
            w2 = pv_f / perm_frac_i
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
                t = perm_frac_i * aperture_i * np.sqrt(area) / dist[b]
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
        perm=np.concatenate([perm_matrix, frac_perm]),
        poro=np.concatenate([poro_matrix, frac_poro]),
        conn=np.asarray(conn, dtype=np.int64).reshape(-1, 2),
        trans=np.asarray(trans, dtype=float),
        ci=np.asarray(ci, dtype=float),
        conn_type=ctype,
    )


def solve_incomp_edfm_3d(*args, **kwargs):
    """Alias for the generic EDFM pressure solve on a 3D EDFM grid."""
    return solve_incomp_edfm(*args, **kwargs)
