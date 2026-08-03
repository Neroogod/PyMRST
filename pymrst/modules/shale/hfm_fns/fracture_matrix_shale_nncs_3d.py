"""3D fracture/matrix shale NNC construction on ``hfm`` projections."""

from collections.abc import Sequence

import numpy as np

from pymrst.modules.hfm import EDFMGrid, PlanarFracture, Projection3D
from pymrst.modules.shale.data import ShaleNNC


def fracture_matrix_shale_nncs_3d(
    edfm: EDFMGrid,
    fractures: Sequence[PlanarFracture],
    projections: Sequence[Projection3D],
    *,
    existing: ShaleNNC | None = None,
    tolerance: float = 1e-10,
) -> ShaleNNC:
    """Return matrix/fracture NNCs with exact projection area and normals."""
    if len(fractures) != len(projections):
        raise ValueError("fractures and projections must have the same length")
    mask = np.asarray([kind == "frac_matrix" for kind in edfm.conn_type])
    raw = {
        tuple(map(int, pair)): float(trans)
        for pair, trans in zip(edfm.conn[mask], edfm.trans[mask])
    }
    cells: list[tuple[int, int]] = []
    trans: list[float] = []
    areas: list[float] = []
    normals: list[np.ndarray] = []
    points: list[np.ndarray] = []
    boundary: list[bool] = []
    fracture_offset = edfm.n_matrix
    volumes = np.concatenate([
        np.asarray(edfm.g.cells.volumes, dtype=float), edfm.frac_volumes
    ])
    for fracture, projection in zip(fractures, projections):
        n, _, _ = fracture.frame()
        flags = projection.boundary
        if flags is None:
            flags = np.zeros(projection.areas.size, dtype=bool)
        for local, (matrix_cell, area, center, on_boundary) in enumerate(zip(
            projection.matrix_cells, projection.areas, projection.centroids, flags
        )):
            fracture_cell = fracture_offset + local
            pair = (int(matrix_cell), int(fracture_cell))
            cells.append(pair)
            # hfm's geometric half-connection uses one fracture surface.
            # MRST shale's ``fracmatnnc`` CI represents both sides before the
            # subsequent ``adjustT`` contained-cell correction, giving a
            # factor two relative to that generic EDFM half-connection.
            trans.append(2.0 * raw[pair])
            areas.append(float(area))
            normals.append(n)
            points.append(center)
            boundary.append(bool(on_boundary))
            if on_boundary:
                other = _opposite_face_cell(
                    edfm.g, int(matrix_cell), fracture, tolerance
                )
                if other is not None:
                    cells.append((other, int(fracture_cell)))
                    trans.append(2.0 * _connection_transmissibility(
                        edfm, volumes, other, int(fracture_cell),
                        float(area), fracture, tolerance,
                    ))
                    areas.append(float(area))
                    normals.append(n)
                    points.append(center)
                    boundary.append(True)
        fracture_offset += projection.matrix_cells.size
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
    trans = np.asarray(trans, dtype=float)
    # Exact MRST adjustT: double rows whose fracture cell occurs once. A face
    # intersection has two matrix connections and is therefore unchanged.
    _, inverse, counts = np.unique(cells[:, 1], return_inverse=True, return_counts=True)
    trans = trans * np.where(counts[inverse] == 1, 2.0, 1.0)
    result = ShaleNNC(
        cells=cells,
        transmissibility=trans,
        connection_type=tuple(
            "fracmat boundary" if flag else "fracmat interior"
            for flag in boundary
        ),
        area=np.asarray(areas),
        normal=np.asarray(normals).reshape(-1, 3),
        plane_point=np.asarray(points).reshape(-1, 3),
    )
    if existing is None or existing.cells.size == 0:
        return result
    return _merge_unique(existing, result)


def _opposite_face_cell(grid, matrix_cell, fracture, tolerance):
    normal, _, _ = fracture.frame()
    center = np.asarray(fracture.center, dtype=float)
    faces = grid.cells.faces[
        grid.cells.facePos[matrix_cell]:grid.cells.facePos[matrix_cell + 1]
    ]
    for face in np.asarray(faces, dtype=np.int64).reshape(-1):
        nodes = grid.faces.nodes[grid.faces.nodePos[face]:grid.faces.nodePos[face + 1]]
        signed = (grid.nodes.coords[nodes, :3] - center) @ normal
        if np.all(np.abs(signed) <= tolerance):
            neighbors = grid.faces.neighbors[face]
            other = neighbors[neighbors != matrix_cell]
            if other.size and other[0] >= 0:
                return int(other[0])
    return None


def _connection_transmissibility(
    edfm, volumes, matrix_cell, fracture_cell, area, fracture, tolerance
):
    normal, _, _ = fracture.frame()
    center = np.asarray(fracture.center, dtype=float)
    faces = edfm.g.cells.faces[
        edfm.g.cells.facePos[matrix_cell]:edfm.g.cells.facePos[matrix_cell + 1]
    ]
    nodes = np.unique(np.concatenate([
        edfm.g.faces.nodes[edfm.g.faces.nodePos[face]:edfm.g.faces.nodePos[face + 1]]
        for face in np.asarray(faces, dtype=np.int64).reshape(-1)
    ]))
    distance = np.mean(np.abs((edfm.g.nodes.coords[nodes, :3] - center) @ normal))
    distance = max(float(distance), tolerance)
    ci = area / distance
    pv_matrix = edfm.poro[matrix_cell] * volumes[matrix_cell]
    pv_fracture = edfm.poro[fracture_cell] * volumes[fracture_cell]
    weight = pv_matrix + pv_fracture
    resistance = (
        pv_matrix / edfm.perm[matrix_cell]
        + pv_fracture / edfm.perm[fracture_cell]
    )
    return float(ci * weight / resistance)


def _merge_unique(existing: ShaleNNC, new: ShaleNNC) -> ShaleNNC:
    """Append NNC rows without duplicating a previously processed cell pair."""
    cells = np.vstack([existing.cells, new.cells])
    keep = []
    seen = set()
    for index, pair in enumerate(cells):
        key = tuple(sorted(map(int, pair)))
        if key not in seen:
            seen.add(key)
            keep.append(index)
    keep = np.asarray(keep, dtype=np.int64)
    types = existing.type + new.type
    return ShaleNNC(
        cells=cells[keep],
        transmissibility=np.concatenate([existing.T, new.T])[keep],
        connection_type=tuple(types[index] for index in keep),
        area=np.concatenate([existing.area, new.area])[keep],
        normal=np.vstack([existing.normal, new.normal])[keep],
        plane_point=np.vstack([existing.plane_point, new.plane_point])[keep],
    )


__all__ = ["fracture_matrix_shale_nncs_3d"]
