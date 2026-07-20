"""Remove faces from a grid, ported from MRST private `removeFaces`."""

from typing import Tuple

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes
from pymrst.gridprocessing.private.packed_data import map_excluding, remove_from_packed_data


def remove_faces(g: Grid, faces: np.ndarray) -> Tuple[Grid, np.ndarray]:
    """Delete faces everywhere they appear and return old-to-new face map."""
    faces = np.asarray(faces, dtype=np.int64).ravel()
    removed = np.zeros(g.faces.num, dtype=bool)
    removed[faces] = True
    face_map = map_excluding(removed)
    num_nodes = np.diff(g.faces.nodePos)
    kept_nodes = g.faces.nodes[~np.repeat(removed, num_nodes)]
    cell_faces, face_pos = remove_from_packed_data(g.cells.facePos, g.cells.faces, faces)
    cell_faces = face_map[cell_faces]
    return _grid_without_faces(g, removed, face_map, kept_nodes, cell_faces, face_pos), face_map


def _grid_without_faces(
    g: Grid,
    removed: np.ndarray,
    face_map: np.ndarray,
    kept_nodes: np.ndarray,
    cell_faces: np.ndarray,
    face_pos: np.ndarray,
) -> Grid:
    num_nodes = np.diff(g.faces.nodePos)
    return Grid(
        cells=Cells(
            num=g.cells.num,
            facePos=face_pos,
            faces=cell_faces,
            indexMap=g.cells.indexMap.copy(),
        ),
        faces=Faces(
            num=g.faces.num - int(removed.sum()),
            nodePos=np.concatenate([[0], np.cumsum(num_nodes[~removed])]).astype(np.int64),
            nodes=kept_nodes,
            neighbors=g.faces.neighbors[~removed].copy(),
        ),
        nodes=Nodes(num=g.nodes.num, coords=g.nodes.coords.copy()),
        griddim=g.griddim,
        type=list(g.type),
    )


if __name__ == "__main__":
    from pymrst.gridprocessing.private.add_faces import add_faces
    from pymrst.gridprocessing.private.empty_grid import empty_grid

    def test_remove_faces_roundtrip() -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        grid = empty_grid(1, coords)
        grid, added = add_faces(
            grid,
            np.array([0, 1, 1, 2, 2, 3, 3, 0]),
            np.full(4, 2),
            np.array([[0, -1]] * 4),
        )
        reduced, face_map = remove_faces(grid, added[:1])
        assert reduced.faces.num == 3
        assert face_map[0] == -1
        assert np.array_equal(reduced.cell_faces(0), [0, 1, 2])
        print("ok test_remove_faces_roundtrip")

    test_remove_faces_roundtrip()
    print("\nAll tests passed!")

