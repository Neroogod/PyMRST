"""Append faces to a grid, ported from MRST private `addFaces`."""

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from pymrst.gridprocessing.grid_structure import Grid, carry_custom_fields
from pymrst.gridprocessing.private.packed_data import (
    insert_in_packed_data_layout,
    place_with_layout,
)

# per-face fields that only `compute_geometry` can supply, so appending faces
# leaves no defensible value for the new rows
_FACE_GEOMETRY = ("areas", "normals", "centroids", "diameters")
# packed per-face-edge fields, meaningful only as a consistent set
_FACE_EDGES = ("edges", "edgePos", "edgeSign", "edgeNormals")


def add_faces(
    g: Grid,
    face_nodes: np.ndarray,
    num_nodes: np.ndarray,
    neighbors: np.ndarray,
    tags: Optional[np.ndarray] = None,
) -> Tuple[Grid, np.ndarray]:
    """Append faces and register them on their neighbor cells.

    `tags` optionally supplies MRST's cell-face tags for the new half-faces,
    ordered like `neighbors` flattened column first - all left neighbors, then
    all right ones - with entries for cells `-1` dropped, as MRST does. When
    the grid already carries `cells.tags` and none are given, the new
    half-faces are tagged zero.

    Every other field of `g` is carried over. Per-face geometry cannot cover
    the appended faces, so `faces.areas`, `normals`, `centroids`, `diameters`
    and the packed edge fields are cleared; rerun `compute_geometry`. Cell
    geometry is kept as MRST keeps it, but is stale once cells gain a face.
    """
    face_nodes = np.asarray(face_nodes, dtype=np.int64).ravel()
    num_nodes = np.asarray(num_nodes, dtype=np.int64).ravel()
    neighbors = np.asarray(neighbors, dtype=np.int64).reshape(-1, 2)
    assert num_nodes.size == neighbors.shape[0]
    assert num_nodes.sum() == face_nodes.size
    new_faces = g.faces.num + np.arange(num_nodes.size, dtype=np.int64)

    cells_flat = np.concatenate([neighbors[:, 0], neighbors[:, 1]])
    faces_flat = np.concatenate([new_faces, new_faces])
    attached = cells_flat != -1
    cell_faces, face_pos, layout = insert_in_packed_data_layout(
        g.cells.facePos, g.cells.faces, cells_flat[attached], faces_flat[attached]
    )
    cell_tags = _new_cell_tags(g, tags, attached, layout)

    faces = replace(
        g.faces,
        num=g.faces.num + num_nodes.size,
        nodePos=np.concatenate(
            [[0], np.cumsum(np.concatenate([np.diff(g.faces.nodePos), num_nodes]))]
        ).astype(np.int64),
        nodes=np.concatenate([g.faces.nodes, face_nodes]),
        neighbors=np.vstack([g.faces.neighbors, neighbors]),
        tag=_extended_face_tag(g, num_nodes.size),
        **{name: None for name in _FACE_GEOMETRY + _FACE_EDGES},
    )
    cells = replace(g.cells, facePos=face_pos, faces=cell_faces, tags=cell_tags)
    out = replace(g, cells=cells, faces=faces)
    # cpnodes indexes nodes and nnc indexes cells, neither of which this
    # touches, so both carry over unchanged
    carry_custom_fields(g, out)
    return out, new_faces


def _extended_face_tag(g: Grid, num_new: int) -> Optional[np.ndarray]:
    """Tag the new faces zero, the way MRST's `addFaces` does."""
    if g.faces.tag is None:
        return None
    tag = np.asarray(g.faces.tag)
    pad = np.zeros((num_new,) + tag.shape[1:], dtype=tag.dtype)
    return np.concatenate([tag, pad])


def _new_cell_tags(g: Grid, tags, attached: np.ndarray, layout) -> Optional[np.ndarray]:
    """Interleave cell-face tags for the new half-faces, or keep them absent."""
    if tags is None and g.cells.tags is None:
        return None
    num_new = int(attached.sum())
    if tags is None:
        new_tags = np.zeros(num_new, dtype=np.asarray(g.cells.tags).dtype)
    else:
        tags = np.asarray(tags)
        if tags.ndim == 2 and tags.shape[1] == 2:        # one column per neighbor
            tags = np.concatenate([tags[:, 0], tags[:, 1]])
        tags = tags.ravel()
        if tags.size == attached.size:
            tags = tags[attached]
        elif tags.size != num_new:
            raise ValueError(
                "tags must hold one entry per new half-face "
                f"({num_new}) or per neighbor slot ({attached.size})"
            )
        new_tags = tags
    existing = (
        np.asarray(g.cells.tags) if g.cells.tags is not None
        else np.zeros(g.cells.faces.size, dtype=new_tags.dtype)
    )
    return place_with_layout(existing, new_tags, layout)


if __name__ == "__main__":
    from pymrst.gridprocessing.private.empty_grid import empty_grid

    SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    RING = np.array([0, 1, 1, 2, 2, 3, 3, 0])

    def test_add_faces_to_empty_grid() -> None:
        grid = empty_grid(1, SQUARE)
        grid, added = add_faces(grid, RING, np.full(4, 2), np.array([[0, -1]] * 4))
        grid.check()
        assert np.array_equal(added, [0, 1, 2, 3])
        assert np.array_equal(grid.cell_faces(0), [0, 1, 2, 3])
        print("ok test_add_faces_to_empty_grid")

    def test_optional_fields_survive() -> None:
        grid = empty_grid(1, SQUARE)
        grid, _ = add_faces(grid, RING, np.full(4, 2), np.array([[0, -1]] * 4))
        grid.cartDims = (1, 1)
        grid.faces.tag = np.arange(4)
        grid.cells.volumes = np.array([1.0])
        extended, _ = add_faces(
            grid, np.array([0, 2]), np.array([2]), np.array([[0, -1]])
        )
        assert extended.cartDims == (1, 1), "cartDims must survive"
        assert np.array_equal(extended.faces.tag, [0, 1, 2, 3, 0]), extended.faces.tag
        assert np.array_equal(extended.cells.volumes, [1.0])
        print("ok test_optional_fields_survive")

    def test_face_geometry_is_invalidated() -> None:
        grid = empty_grid(1, SQUARE)
        grid, _ = add_faces(grid, RING, np.full(4, 2), np.array([[0, -1]] * 4))
        grid.faces.areas = np.ones(4)
        extended, _ = add_faces(
            grid, np.array([0, 2]), np.array([2]), np.array([[0, -1]])
        )
        assert extended.faces.areas is None, "stale areas must not survive"
        print("ok test_face_geometry_is_invalidated")

    def test_cell_face_tags_track_new_half_faces() -> None:
        grid = empty_grid(1, SQUARE)
        grid, _ = add_faces(
            grid, RING, np.full(4, 2), np.array([[0, -1]] * 4),
            tags=np.array([1, 2, 3, 4, 0, 0, 0, 0]),
        )
        assert np.array_equal(grid.cells.tags, [1, 2, 3, 4]), grid.cells.tags
        extended, _ = add_faces(
            grid, np.array([0, 2]), np.array([2]), np.array([[0, -1]]),
            tags=np.array([[9, 0]]),
        )
        assert extended.cells.tags.size == extended.cells.faces.size
        assert np.array_equal(extended.cells.tags, [1, 2, 3, 4, 9]), extended.cells.tags
        print("ok test_cell_face_tags_track_new_half_faces")

    test_add_faces_to_empty_grid()
    test_optional_fields_survive()
    test_face_geometry_is_invalidated()
    test_cell_face_tags_track_new_half_faces()
    print("\nAll tests passed!")
