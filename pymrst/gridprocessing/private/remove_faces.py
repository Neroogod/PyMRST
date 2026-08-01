"""Remove faces from a grid, ported from MRST private `removeFaces`."""

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from pymrst.gridprocessing.grid_structure import Grid, carry_custom_fields
from pymrst.gridprocessing.private.packed_data import map_excluding, remove_from_packed_data

# per-face fields that stay valid for the faces that survive
_FACE_FIELDS = ("tag", "areas", "normals", "centroids", "diameters")


def remove_faces(g: Grid, faces: np.ndarray) -> Tuple[Grid, np.ndarray]:
    """Delete faces everywhere they appear and return old-to-new face map.

    Per-face data is carried across: tags, areas, normals, centroids,
    diameters and the packed edge fields are all restricted to the surviving
    faces, and cell-face tags are dropped alongside their half-faces. Cell
    geometry is kept as MRST keeps it, but is stale once cells lose a face.
    """
    faces = np.asarray(faces, dtype=np.int64).ravel()
    removed = np.zeros(g.faces.num, dtype=bool)
    removed[faces] = True
    face_map = map_excluding(removed)
    num_nodes = np.diff(g.faces.nodePos)
    kept_nodes = g.faces.nodes[~np.repeat(removed, num_nodes)]
    dropped_half_faces = removed[g.cells.faces]
    cell_faces, face_pos = remove_from_packed_data(g.cells.facePos, g.cells.faces, faces)
    cell_faces = face_map[cell_faces]

    kept = ~removed
    faces_out = replace(
        g.faces,
        num=g.faces.num - int(removed.sum()),
        nodePos=np.concatenate([[0], np.cumsum(num_nodes[kept])]).astype(np.int64),
        nodes=kept_nodes,
        neighbors=g.faces.neighbors[kept].copy(),
        **{name: _subset(getattr(g.faces, name), kept) for name in _FACE_FIELDS},
        **_subset_face_edges(g, kept),
    )
    cells_out = replace(
        g.cells,
        facePos=face_pos,
        faces=cell_faces,
        tags=_subset(g.cells.tags, ~dropped_half_faces),
    )
    out = replace(g, cells=cells_out, faces=faces_out)
    # cpnodes indexes nodes and nnc indexes cells, neither of which this
    # touches, so both carry over unchanged
    carry_custom_fields(g, out)
    return out, face_map


def _subset(values, kept: np.ndarray) -> Optional[np.ndarray]:
    """Restrict an optional per-entity array to the entities that survive."""
    if values is None:
        return None
    return np.asarray(values)[kept].copy()


def _subset_face_edges(g: Grid, kept: np.ndarray) -> dict:
    """Restrict the packed per-face-edge fields to the surviving faces."""
    if g.faces.edges is None or g.faces.edgePos is None:
        return {name: None for name in
                ("edges", "edgePos", "edgeSign", "edgeNormals")}
    edge_pos = np.asarray(g.faces.edgePos, dtype=np.int64)
    counts = np.diff(edge_pos)
    keep_entry = np.repeat(kept, counts)
    return {
        "edges": np.asarray(g.faces.edges)[keep_entry].copy(),
        "edgePos": np.concatenate([[0], np.cumsum(counts[kept])]).astype(np.int64),
        "edgeSign": _subset(g.faces.edgeSign, keep_entry),
        "edgeNormals": _subset(g.faces.edgeNormals, keep_entry),
    }


if __name__ == "__main__":
    from pymrst.gridprocessing.private.add_faces import add_faces
    from pymrst.gridprocessing.private.empty_grid import empty_grid

    SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    RING = np.array([0, 1, 1, 2, 2, 3, 3, 0])

    def _square(**tags):
        grid = empty_grid(1, SQUARE)
        grid, added = add_faces(
            grid, RING, np.full(4, 2), np.array([[0, -1]] * 4), **tags
        )
        return grid, added

    def test_remove_faces_roundtrip() -> None:
        grid, added = _square()
        reduced, face_map = remove_faces(grid, added[:1])
        assert reduced.faces.num == 3
        assert face_map[0] == -1
        assert np.array_equal(reduced.cell_faces(0), [0, 1, 2])
        print("ok test_remove_faces_roundtrip")

    def test_per_face_data_follows_the_surviving_faces() -> None:
        grid, added = _square()
        grid.faces.tag = np.array([10, 11, 12, 13])
        grid.faces.areas = np.array([1.0, 2.0, 3.0, 4.0])
        grid.faces.centroids = np.arange(8, dtype=float).reshape(4, 2)
        grid.cartDims = (1, 1)
        reduced, _ = remove_faces(grid, added[[1]])
        assert np.array_equal(reduced.faces.tag, [10, 12, 13]), reduced.faces.tag
        assert np.array_equal(reduced.faces.areas, [1.0, 3.0, 4.0])
        assert np.array_equal(reduced.faces.centroids, [[0, 1], [4, 5], [6, 7]])
        assert reduced.cartDims == (1, 1), "cartDims must survive"
        print("ok test_per_face_data_follows_the_surviving_faces")

    def test_cell_face_tags_drop_with_their_half_faces() -> None:
        grid, added = _square(tags=np.array([1, 2, 3, 4, 0, 0, 0, 0]))
        reduced, _ = remove_faces(grid, added[[2]])
        assert np.array_equal(reduced.cells.tags, [1, 2, 4]), reduced.cells.tags
        assert reduced.cells.tags.size == reduced.cells.faces.size
        print("ok test_cell_face_tags_drop_with_their_half_faces")

    def test_removing_nothing_keeps_every_field() -> None:
        grid, _ = _square()
        grid.faces.areas = np.ones(4)
        reduced, face_map = remove_faces(grid, np.empty(0, dtype=np.int64))
        assert reduced.faces.num == 4
        assert np.array_equal(reduced.faces.areas, np.ones(4))
        assert np.array_equal(face_map, [0, 1, 2, 3])
        print("ok test_removing_nothing_keeps_every_field")

    test_remove_faces_roundtrip()
    test_per_face_data_follows_the_surviving_faces()
    test_cell_face_tags_drop_with_their_half_faces()
    test_removing_nothing_keeps_every_field()
    print("\nAll tests passed!")
