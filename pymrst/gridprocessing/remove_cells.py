"""Remove cells from a grid, ported from MRST `removeCells`."""

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes
from pymrst.gridprocessing.private.packed_data import map_excluding


def remove_cells(g: Grid, cells: np.ndarray, *, return_mapping: bool = False):
    """Delete cells, faces that lose both neighbors, and orphan nodes.

    Boolean cell masks are accepted. With ``return_mapping=True``, return
    ``(grid, cellmap, facemap, nodemap)`` where each map contains the original
    entity index for every retained entity, matching MRST's optional outputs.
    """
    cells_in = np.asarray(cells)
    if cells_in.dtype == bool:
        if cells_in.size != g.cells.num:
            raise ValueError("logical cell mask must have one entry per cell")
        cells = np.flatnonzero(cells_in.ravel())
    else:
        cells = np.asarray(cells, dtype=np.int64).ravel()
    if cells.size == 0:
        if return_mapping:
            return (g, np.arange(g.cells.num), np.arange(g.faces.num),
                    np.arange(g.nodes.num))
        return g
    removed = np.zeros(g.cells.num, dtype=bool)
    removed[cells] = True
    cell_map = map_excluding(removed)
    cell_faces, face_pos = _kept_cell_faces(g, removed)
    neighbors = _renumber_neighbors(g.faces.neighbors, cell_map)
    face_removed = np.all(neighbors == -1, axis=1)
    facemap = np.flatnonzero(~face_removed)
    counts = np.diff(g.faces.nodePos)
    retained_face_nodes = g.faces.nodes[
        ~np.repeat(face_removed, counts)
    ]
    nodemap = np.unique(retained_face_nodes)
    cellmap = np.flatnonzero(~removed)
    out = _drop_orphan_faces_and_nodes(
        g, removed, cell_faces, face_pos, neighbors
    )

    # Preserve and consistently update optional corner-point/NNC metadata.
    if hasattr(g.cells, "cpnodes"):
        node_new = np.full(g.nodes.num, -1, dtype=np.int64)
        node_new[nodemap] = np.arange(nodemap.size)
        cpnodes = node_new[np.asarray(g.cells.cpnodes)[cellmap]]
        if np.any(cpnodes < 0):
            raise ValueError("too many nodes removed to preserve corner-point nodes")
        out.cells.cpnodes = cpnodes
    extras = {
        name: value for name, value in vars(g).items()
        if name not in {"cells", "faces", "nodes", "griddim", "type",
                        "cartDims", "edges", "nnc"}
    }
    for name, value in extras.items():
        setattr(out, name, value)
    nnc = getattr(g, "nnc", None)
    if nnc is not None:
        is_dict = isinstance(nnc, dict)
        pairs = np.asarray(nnc["cells"] if is_dict else nnc.cells,
                           dtype=np.int64)
        cell_new = np.full(g.cells.num, -1, dtype=np.int64)
        cell_new[cellmap] = np.arange(cellmap.size)
        keep_nnc = np.all(cell_new[pairs] >= 0, axis=1)
        values = vars(nnc).copy() if not is_dict else dict(nnc)
        values["cells"] = cell_new[pairs[keep_nnc]]
        for name, value in list(values.items()):
            arr = np.asarray(value)
            if name != "cells" and arr.ndim and arr.shape[0] == pairs.shape[0]:
                values[name] = arr[keep_nnc]
        if is_dict:
            out.nnc = values
        else:
            from types import SimpleNamespace
            out.nnc = SimpleNamespace(**values)
    if return_mapping:
        return out, cellmap, facemap, nodemap
    return out


def _kept_cell_faces(g: Grid, removed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_faces = np.diff(g.cells.facePos)
    cell_faces = g.cells.faces[~np.repeat(removed, num_faces)]
    face_pos = np.concatenate([[0], np.cumsum(num_faces[~removed])]).astype(np.int64)
    return cell_faces, face_pos


def _renumber_neighbors(neighbors: np.ndarray, cell_map: np.ndarray) -> np.ndarray:
    out = neighbors.copy()
    for col in (0, 1):
        interior = out[:, col] >= 0
        out[interior, col] = cell_map[out[interior, col]]
    return out


def _drop_orphan_faces_and_nodes(
    g: Grid,
    removed_cells: np.ndarray,
    cell_faces: np.ndarray,
    face_pos: np.ndarray,
    neighbors: np.ndarray,
) -> Grid:
    face_removed = np.all(neighbors == -1, axis=1)
    face_map = map_excluding(face_removed)
    num_nodes = np.diff(g.faces.nodePos)
    face_nodes = g.faces.nodes[~np.repeat(face_removed, num_nodes)]
    cell_faces = face_map[cell_faces]
    assert np.all(cell_faces >= 0), "removeCells: too many faces removed"
    node_removed = np.ones(g.nodes.num, dtype=bool)
    node_removed[face_nodes] = False
    node_map = map_excluding(node_removed)
    face_nodes = node_map[face_nodes]
    assert np.all(face_nodes >= 0), "removeCells: too many nodes removed"
    return Grid(
        cells=Cells(
            num=g.cells.num - int(removed_cells.sum()),
            facePos=face_pos,
            faces=cell_faces,
            # a glued grid has no indexMap - MRST drops it in squareGrid and
            # guards with isfield, so absence has to survive removal too
            indexMap=(None if g.cells.indexMap is None
                      else g.cells.indexMap[~removed_cells].copy()),
            tags=(
                g.cells.tags[~np.repeat(removed_cells, np.diff(g.cells.facePos))].copy()
                if g.cells.tags is not None else None
            ),
            volumes=(
                g.cells.volumes[~removed_cells].copy()
                if g.cells.volumes is not None else None
            ),
            centroids=(
                g.cells.centroids[~removed_cells].copy()
                if g.cells.centroids is not None else None
            ),
        ),
        faces=Faces(
            num=g.faces.num - int(face_removed.sum()),
            nodePos=np.concatenate([[0], np.cumsum(num_nodes[~face_removed])]).astype(np.int64),
            nodes=face_nodes,
            neighbors=neighbors[~face_removed],
            tag=(
                g.faces.tag[~face_removed].copy()
                if g.faces.tag is not None else None
            ),
            areas=(
                g.faces.areas[~face_removed].copy()
                if g.faces.areas is not None else None
            ),
            normals=(
                g.faces.normals[~face_removed].copy()
                if g.faces.normals is not None else None
            ),
            centroids=(
                g.faces.centroids[~face_removed].copy()
                if g.faces.centroids is not None else None
            ),
        ),
        nodes=Nodes(
            num=g.nodes.num - int(node_removed.sum()),
            coords=g.nodes.coords[~node_removed].copy(),
        ),
        griddim=g.griddim,
        type=list(g.type) + ["removeCells"],
        cartDims=g.cartDims,
    )


if __name__ == "__main__":
    from pymrst.gridprocessing.grid_structure import Cells, Faces, Nodes

    def _two_cell_grid() -> Grid:
        coords = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
        )
        face_nodes = np.array([0, 3, 1, 4, 2, 5, 0, 1, 1, 2, 3, 4, 4, 5])
        return Grid(
            cells=Cells(num=2, facePos=np.array([0, 4, 8]), faces=np.array([0, 1, 3, 5, 1, 2, 4, 6]), indexMap=np.arange(2)),
            faces=Faces(num=7, nodePos=np.arange(0, 15, 2), nodes=face_nodes, neighbors=np.array([[-1, 0], [0, 1], [1, -1], [-1, 0], [-1, 1], [0, -1], [1, -1]])),
            nodes=Nodes(num=6, coords=coords),
            griddim=2,
            type=["tiny"],
        )

    def test_remove_cells_drops_orphans() -> None:
        grid = remove_cells(_two_cell_grid(), np.array([1]))
        grid.check()
        assert grid.cells.num == 1
        assert grid.faces.num == 4
        assert grid.nodes.num == 4
        print("ok test_remove_cells_drops_orphans")

    def test_remove_no_cells_is_noop() -> None:
        grid = _two_cell_grid()
        assert remove_cells(grid, np.array([], dtype=np.int64)) is grid
        print("ok test_remove_no_cells_is_noop")

    test_remove_cells_drops_orphans()
    test_remove_no_cells_is_noop()
    print("\nAll tests passed!")
