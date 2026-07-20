"""Append faces to a grid, ported from MRST private `addFaces`."""

from typing import Tuple

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes
from pymrst.gridprocessing.private.packed_data import insert_in_packed_data


def add_faces(
    g: Grid,
    face_nodes: np.ndarray,
    num_nodes: np.ndarray,
    neighbors: np.ndarray,
) -> Tuple[Grid, np.ndarray]:
    """Append faces and register them on their neighbor cells."""
    face_nodes = np.asarray(face_nodes, dtype=np.int64).ravel()
    num_nodes = np.asarray(num_nodes, dtype=np.int64).ravel()
    neighbors = np.asarray(neighbors, dtype=np.int64).reshape(-1, 2)
    assert num_nodes.size == neighbors.shape[0]
    assert num_nodes.sum() == face_nodes.size
    new_faces = g.faces.num + np.arange(num_nodes.size, dtype=np.int64)
    cell_faces, face_pos = _append_cell_faces(g, neighbors, new_faces)
    return _with_added_faces(g, cell_faces, face_pos, face_nodes, num_nodes, neighbors), new_faces


def _append_cell_faces(
    g: Grid,
    neighbors: np.ndarray,
    new_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cells_flat = np.concatenate([neighbors[:, 0], neighbors[:, 1]])
    faces_flat = np.concatenate([new_faces, new_faces])
    keep = cells_flat != -1
    return insert_in_packed_data(
        g.cells.facePos,
        g.cells.faces,
        cells_flat[keep],
        faces_flat[keep],
    )


def _with_added_faces(
    g: Grid,
    cell_faces: np.ndarray,
    face_pos: np.ndarray,
    face_nodes: np.ndarray,
    num_nodes: np.ndarray,
    neighbors: np.ndarray,
) -> Grid:
    return Grid(
        cells=Cells(
            num=g.cells.num,
            facePos=face_pos,
            faces=cell_faces,
            indexMap=g.cells.indexMap.copy(),
        ),
        faces=Faces(
            num=g.faces.num + num_nodes.size,
            nodePos=np.concatenate(
                [[0], np.cumsum(np.concatenate([np.diff(g.faces.nodePos), num_nodes]))]
            ).astype(np.int64),
            nodes=np.concatenate([g.faces.nodes, face_nodes]),
            neighbors=np.vstack([g.faces.neighbors, neighbors]),
        ),
        nodes=Nodes(num=g.nodes.num, coords=g.nodes.coords.copy()),
        griddim=g.griddim,
        type=list(g.type),
    )


if __name__ == "__main__":
    from pymrst.gridprocessing.private.empty_grid import empty_grid

    def test_add_faces_to_empty_grid() -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        grid = empty_grid(1, coords)
        face_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 0])
        grid, added = add_faces(grid, face_nodes, np.full(4, 2), np.array([[0, -1]] * 4))
        grid.check()
        assert np.array_equal(added, [0, 1, 2, 3])
        assert np.array_equal(grid.cell_faces(0), [0, 1, 2, 3])
        print("ok test_add_faces_to_empty_grid")

    test_add_faces_to_empty_grid()
    print("\nAll tests passed!")

