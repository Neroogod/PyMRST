"""Corner-point cell geometry used by AD well and facility models.

Port of MRST ``computeCpGeometry``.  Unlike the processed grid geometry, these
quantities are computed directly from the logical corner-point deck and
therefore retain values for inactive Cartesian cells.
"""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing.process_grdecl import build_corner_pt_nodes


def _field(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def _corner_coordinates(grdecl) -> list[np.ndarray]:
    x, y, z = build_corner_pt_nodes(grdecl)

    def corner(i, j, k):
        return np.column_stack(
            [v[i::2, j::2, k::2].ravel(order="F") for v in (x, y, z)]
        )

    # MATLAB ndgrid(1:2) linear order: i varies first, then j, then k.
    return [
        corner(0, 0, 0),
        corner(1, 0, 0),
        corner(0, 1, 0),
        corner(1, 1, 0),
        corner(0, 0, 1),
        corner(1, 0, 1),
        corner(0, 1, 1),
        corner(1, 1, 1),
    ]


def _face_centers(grdecl) -> tuple[list[np.ndarray], np.ndarray]:
    points = _corner_coordinates(grdecl)
    face_corners = (
        (0, 2, 4, 6),
        (1, 3, 5, 7),
        (0, 1, 4, 5),
        (2, 3, 6, 7),
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    )
    centers = [sum((points[i] for i in ids)) / 4.0 for ids in face_corners]
    return centers, (centers[0] + centers[1]) / 2.0


def _half_face_tags(cells) -> np.ndarray:
    faces = np.asarray(_field(cells, "faces"))
    if faces.ndim == 2 and faces.shape[1] >= 2:
        tags = faces[:, 1].astype(np.int64)
        # Imported MATLAB structs normally retain MRST's 1..6 convention.
        if tags.size and tags.min() >= 1:
            tags = tags - 1
        return tags

    tags = _field(cells, "tags")
    if tags is None:
        raise ValueError(
            "compute_cp_geometry requires logical half-face tags in "
            "grid.cells.tags (0..5)"
        )
    return np.asarray(tags, dtype=np.int64).ravel()


def compute_cp_geometry(grid, grdecl, on_all_cpc: bool = False):
    """Return ``(cell_centers, cell_face_centers, cell_dims)``.

    With ``on_all_cpc=False`` (MRST default), centers are returned for active
    cells and half-faces in processed-grid order.  With ``True``, cell and face
    centers cover all logical Cartesian cells, including inactive cells.
    ``cell_dims`` always covers all logical Cartesian cells, matching MRST.
    """
    face_centers, all_cell_centers = _face_centers(grdecl)
    cell_dims = np.column_stack(
        [
            np.linalg.norm(face_centers[1] - face_centers[0], axis=1),
            np.linalg.norm(face_centers[3] - face_centers[2], axis=1),
            np.linalg.norm(face_centers[5] - face_centers[4], axis=1),
        ]
    )

    if on_all_cpc:
        cart_cell_no = np.repeat(np.arange(all_cell_centers.shape[0]), 6)
        face_type = np.tile(np.arange(6), all_cell_centers.shape[0])
        cell_centers = all_cell_centers
    else:
        cells = _field(grid, "cells")
        face_pos = np.asarray(_field(cells, "facePos"), dtype=np.int64).ravel()
        index_map = np.asarray(_field(cells, "indexMap"), dtype=np.int64).ravel()
        cell_no = np.repeat(np.arange(index_map.size), np.diff(face_pos))
        cart_cell_no = index_map[cell_no]
        face_type = _half_face_tags(cells)
        if face_type.size != cart_cell_no.size:
            raise ValueError("one logical face tag is required per grid half-face")
        if np.any((face_type < 0) | (face_type > 5)):
            raise ValueError("logical half-face tags must be in 0..5 or 1..6")
        cell_centers = all_cell_centers[index_map]

    cell_face_centers = np.zeros((cart_cell_no.size, 3), dtype=float)
    for face in range(6):
        selected = face_type == face
        cell_face_centers[selected] = face_centers[face][cart_cell_no[selected]]

    return cell_centers, cell_face_centers, cell_dims


computeCpGeometry = compute_cp_geometry


__all__ = ["compute_cp_geometry", "computeCpGeometry"]
