"""Global shale cell/NNC assembly built on the consolidated ``hfm`` grid."""

import numpy as np

from pymrst.gridprocessing.grid import Cells, Faces, Grid, Nodes
from pymrst.modules.hfm import EDFMGrid, Projection3D
from pymrst.modules.shale.data import ShaleGlobalGrid, ShaleNNC, ShaleRock


def assemble_shale_global_grid(
    edfm: EDFMGrid,
    projections: tuple[Projection3D, ...] | list[Projection3D] | None = None,
    *,
    tolerance: float = 1e-10,
) -> ShaleGlobalGrid:
    """Convert an ``hfm`` EDFM grid into the shared shale global contract."""
    types = tuple(
        "fracfrac" if value == "frac" else "fracmat"
        for value in edfm.conn_type
    )
    area = np.zeros(edfm.conn.shape[0], dtype=float)
    # hfm stores matrix/fracture connectivity index (area/distance) in CI.
    # Keep it separate from area until the projection owner supplies exact area.
    normals = np.zeros((edfm.conn.shape[0], 3), dtype=float)
    plane_points = np.zeros_like(normals)
    nnc_mask = np.asarray([kind != "matrix" for kind in edfm.conn_type])
    nnc = ShaleNNC(
        cells=edfm.conn[nnc_mask],
        transmissibility=edfm.trans[nnc_mask],
        connection_type=tuple(np.asarray(types, dtype=object)[nnc_mask]),
        area=area[nnc_mask],
        normal=normals[nnc_mask],
        plane_point=plane_points[nnc_mask],
    )
    rock = ShaleRock(
        perm=np.asarray(edfm.perm).reshape(-1, 1),
        poro=np.asarray(edfm.poro),
        is_matrix=np.arange(edfm.n_total) < edfm.n_matrix,
    )
    topology = None
    cell_map = np.arange(edfm.n_matrix, edfm.n_total, dtype=np.int64)
    face_map = np.empty(0, dtype=np.int64)
    node_map = np.empty(0, dtype=np.int64)
    if projections is not None:
        topology, face_map, node_map = _append_fracture_topology(
            edfm, tuple(projections), tolerance
        )
    return ShaleGlobalGrid(
        matrix=edfm.g,
        fracture_volumes=np.asarray(edfm.frac_volumes),
        fracture_centroids=np.asarray(edfm.frac_centroids),
        fracture_plane=np.asarray(edfm.frac_line, dtype=np.int64),
        rock=rock,
        nnc=nnc,
        grid=topology,
        fracture_cell_map=cell_map,
        fracture_face_map=face_map,
        fracture_node_map=node_map,
    )


def _append_fracture_topology(edfm, projections, tolerance):
    matrix = edfm.g
    polygons = [polygon for projection in projections for polygon in projection.polygons]
    if len(polygons) != edfm.frac_cells:
        raise ValueError("projection polygons do not match EDFM fracture cells")

    base_node_count = matrix.nodes.num
    fracture_nodes = []
    node_lookup = {}

    def node_id(point):
        scale = max(float(np.max(np.abs(point))), 1.0)
        key = tuple(np.round(np.asarray(point) / (tolerance * scale)).astype(np.int64))
        if key not in node_lookup:
            node_lookup[key] = base_node_count + len(fracture_nodes)
            fracture_nodes.append(np.asarray(point, dtype=float))
        return node_lookup[key]

    face_lookup = {}
    face_nodes = []
    face_neighbors = []
    face_areas = []
    face_centroids = []
    face_normals = []
    cell_faces = []
    cell_face_pos = [0]
    apertures = []
    offset = 0
    for projection in projections:
        for local_index, polygon in enumerate(projection.polygons):
            frac_index = offset + local_index
            global_cell = edfm.n_matrix + frac_index
            area = projection.areas[local_index]
            aperture = edfm.frac_volumes[frac_index] / area
            apertures.append(aperture)
            ids = [node_id(point) for point in polygon]
            normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
            normal = normal / np.linalg.norm(normal)
            owned = []
            for edge_index, first in enumerate(ids):
                second = ids[(edge_index + 1) % len(ids)]
                key = tuple(sorted((first, second)))
                if key in face_lookup:
                    face = face_lookup[key]
                    if face_neighbors[face][1] >= 0:
                        raise ValueError("fracture edge belongs to more than two cells")
                    face_neighbors[face][1] = global_cell
                else:
                    face = len(face_nodes)
                    face_lookup[key] = face
                    face_nodes.append((first, second))
                    face_neighbors.append([global_cell, -1])
                    p1 = np.asarray(polygon[edge_index])
                    p2 = np.asarray(polygon[(edge_index + 1) % len(polygon)])
                    edge = p2 - p1
                    length = np.linalg.norm(edge)
                    face_areas.append(length * aperture)
                    face_centroids.append(0.5 * (p1 + p2))
                    face_normals.append(np.cross(edge, normal) / length * (length * aperture))
                owned.append(matrix.faces.num + face)
            cell_faces.extend(owned)
            cell_face_pos.append(cell_face_pos[-1] + len(owned))
        offset += len(projection.polygons)

    fracture_face_count = len(face_nodes)
    fracture_node_count = len(fracture_nodes)
    old_half_faces = np.asarray(matrix.cells.faces, dtype=np.int64).reshape(-1)
    new_face_pos = np.concatenate([
        np.asarray(matrix.cells.facePos, dtype=np.int64),
        matrix.cells.facePos[-1] + np.asarray(cell_face_pos[1:], dtype=np.int64),
    ])
    old_face_nodes = np.asarray(matrix.faces.nodes, dtype=np.int64).reshape(-1)
    flat_fracture_face_nodes = np.asarray(face_nodes, dtype=np.int64).reshape(-1)
    new_node_pos = np.concatenate([
        np.asarray(matrix.faces.nodePos, dtype=np.int64),
        matrix.faces.nodePos[-1] + 2 * np.arange(1, fracture_face_count + 1),
    ])
    nodes = np.vstack([
        np.asarray(matrix.nodes.coords, dtype=float),
        np.asarray(fracture_nodes, dtype=float).reshape(-1, 3),
    ])
    global_grid = Grid(
        cells=Cells(
            num=edfm.n_total,
            facePos=new_face_pos,
            faces=np.concatenate([old_half_faces, np.asarray(cell_faces, dtype=np.int64)]),
            indexMap=np.arange(edfm.n_total, dtype=np.int64),
            volumes=np.concatenate([
                np.asarray(matrix.cells.volumes, dtype=float), edfm.frac_volumes
            ]),
            centroids=np.vstack([
                np.asarray(matrix.cells.centroids, dtype=float), edfm.frac_centroids
            ]),
        ),
        faces=Faces(
            num=matrix.faces.num + fracture_face_count,
            nodePos=new_node_pos,
            nodes=np.concatenate([old_face_nodes, flat_fracture_face_nodes]),
            neighbors=np.vstack([
                np.asarray(matrix.faces.neighbors, dtype=np.int64),
                np.asarray(face_neighbors, dtype=np.int64),
            ]),
            tag=None if matrix.faces.tag is None else np.concatenate([
                matrix.faces.tag, np.zeros(fracture_face_count, dtype=matrix.faces.tag.dtype)
            ]),
            areas=np.concatenate([
                np.asarray(matrix.faces.areas, dtype=float), np.asarray(face_areas)
            ]),
            normals=np.vstack([
                np.asarray(matrix.faces.normals, dtype=float), np.asarray(face_normals)
            ]),
            centroids=np.vstack([
                np.asarray(matrix.faces.centroids, dtype=float), np.asarray(face_centroids)
            ]),
        ),
        nodes=Nodes(num=nodes.shape[0], coords=nodes),
        griddim=matrix.griddim,
        type=list(matrix.type) + ["assembleShaleGlobalGrid"],
        cartDims=None,
    )
    global_grid.check()
    return (
        global_grid,
        np.arange(matrix.faces.num, global_grid.faces.num, dtype=np.int64),
        np.arange(base_node_count, base_node_count + fracture_node_count, dtype=np.int64),
    )


__all__ = ["assemble_shale_global_grid"]
