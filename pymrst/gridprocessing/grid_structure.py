"""The unstructured grid structure, ported from MRST `grid_structure`."""

from dataclasses import dataclass, field, fields
from typing import List, Optional, Tuple

import numpy as np

WEST: int = 0
EAST: int = 1
SOUTH: int = 2
NORTH: int = 3
TOP: int = 4
BOTTOM: int = 5


@dataclass
class Cells:
    """Cell topology and optional geometry fields."""

    num: int
    facePos: np.ndarray
    faces: np.ndarray
    indexMap: np.ndarray
    tags: Optional[np.ndarray] = None
    volumes: Optional[np.ndarray] = None
    centroids: Optional[np.ndarray] = None
    nodes: Optional[np.ndarray] = None
    nodePos: Optional[np.ndarray] = None
    # VEM augmented fields (see modules/vem/utils/compute_vem_geometry).
    edges: Optional[np.ndarray] = None
    edgePos: Optional[np.ndarray] = None
    diameters: Optional[np.ndarray] = None


@dataclass
class Faces:
    """Face topology and optional geometry fields."""

    num: int
    nodePos: np.ndarray
    nodes: np.ndarray
    neighbors: np.ndarray
    tag: Optional[np.ndarray] = None
    areas: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    centroids: Optional[np.ndarray] = None
    # VEM augmented fields (3-D): per face-edge edge index/sign/in-plane normal.
    edges: Optional[np.ndarray] = None
    edgePos: Optional[np.ndarray] = None
    edgeSign: Optional[np.ndarray] = None
    edgeNormals: Optional[np.ndarray] = None
    diameters: Optional[np.ndarray] = None


@dataclass
class Nodes:
    """Node coordinates in meters."""

    num: int
    coords: np.ndarray


@dataclass
class Edges:
    """Edge topology and geometry (VEM 3-D), organized like 2-D faces."""

    num: int
    nodes: np.ndarray
    nodePos: np.ndarray
    lengths: Optional[np.ndarray] = None
    centroids: Optional[np.ndarray] = None


@dataclass
class Grid:
    """Unstructured grid: the PyMRST equivalent of MRST's `G` struct."""

    cells: Cells
    faces: Faces
    nodes: Nodes
    griddim: int
    type: List[str] = field(default_factory=list)
    cartDims: Optional[Tuple[int, ...]] = None
    edges: Optional[Edges] = None

    @property
    def spatial_dim(self) -> int:
        """Return the number of embedding-space dimensions."""
        return self.nodes.coords.shape[1] if self.nodes.coords.ndim > 1 else 1

    def cell_faces(self, c: int) -> np.ndarray:
        """Return face indices of cell `c`."""
        return self.faces_of_cells()[self.cells.facePos[c] : self.cells.facePos[c + 1]]

    def faces_of_cells(self) -> np.ndarray:
        """Return the flat half-face to face-index array."""
        return self.cells.faces

    def face_nodes(self, f: int) -> np.ndarray:
        """Return node indices of face `f` in stored order."""
        return self.faces.nodes[self.faces.nodePos[f] : self.faces.nodePos[f + 1]]

    def cell_no(self) -> np.ndarray:
        """Return the cell index of every half-face."""
        return np.repeat(
            np.arange(self.cells.num, dtype=np.int64), np.diff(self.cells.facePos)
        )

    def check(self) -> None:
        """Assert internal consistency of topology arrays."""
        cells, faces, nodes = self.cells, self.faces, self.nodes
        assert cells.facePos.shape == (cells.num + 1,)
        assert cells.facePos[0] == 0 and cells.facePos[-1] == cells.faces.shape[0]
        assert np.all(np.diff(cells.facePos) > 0), "every cell needs faces"
        assert faces.nodePos.shape == (faces.num + 1,)
        assert faces.nodePos[0] == 0 and faces.nodePos[-1] == faces.nodes.shape[0]
        assert faces.neighbors.shape == (faces.num, 2)
        assert cells.faces.min() >= 0 and cells.faces.max() < faces.num
        assert faces.nodes.min() >= 0 and faces.nodes.max() < nodes.num
        assert faces.neighbors.min() >= -1 and faces.neighbors.max() < cells.num
        assert not np.any(np.all(faces.neighbors == -1, axis=1)), "orphan face"


def carry_custom_fields(source: Grid, target: Grid) -> None:
    """Copy `source`'s undeclared attributes onto `target`.

    Corner-point node maps (`cells.cpnodes`) and non-neighbor connections
    (`g.nnc`) are attached to a grid dynamically rather than declared on the
    dataclasses, so `dataclasses.replace` and any explicit rebuild drop them.
    Attributes `target` already carries are left alone, which lets a caller
    remap an index array first and still pick up everything else.
    """
    for part in ("cells", "faces", "nodes"):
        _carry(getattr(source, part), getattr(target, part))
    _carry(source, target)


def _carry(source, target) -> None:
    declared = {f.name for f in fields(source)}
    for name, value in vars(source).items():
        if name not in declared and not hasattr(target, name):
            setattr(target, name, value)


def remap_cp_nodes(g: Grid, node_map: np.ndarray) -> None:
    """Point a corner-point node map at renumbered nodes.

    `cells.cpnodes` holds node indices, so merging or renumbering nodes leaves
    it addressing the old numbering - silently wrong, or out of range once the
    node array shrinks.
    """
    cpnodes = getattr(g.cells, "cpnodes", None)
    if cpnodes is not None:
        g.cells.cpnodes = node_map[np.asarray(cpnodes, dtype=np.int64)]


def invalidate_geometry(g: Grid) -> None:
    """Drop geometry that moving or merging nodes has made wrong.

    Collapse operations leave these arrays the right length but holding
    pre-collapse values, which is worse than absent: nothing errors and the
    stale numbers flow on. Rerun `compute_geometry`.
    """
    for name in ("areas", "normals", "centroids", "diameters",
                 "edges", "edgePos", "edgeSign", "edgeNormals"):
        setattr(g.faces, name, None)
    for name in ("volumes", "centroids", "diameters", "nodes", "nodePos",
                 "edges", "edgePos"):
        setattr(g.cells, name, None)


def extract_subgrid(g: Grid, cells, *, return_mapping: bool = False):
    """Return the grid made of ``cells`` alone.

    Boolean masks are accepted. With ``return_mapping=True``, return
    ``(subgrid, cells, faces, nodes)`` containing original entity indices,
    matching MRST's optional outputs.
    """
    cells_in = np.asarray(cells)
    if cells_in.dtype == bool:
        if cells_in.size != g.cells.num:
            raise ValueError("logical cell mask must have one entry per cell")
        requested = np.flatnonzero(cells_in.ravel())
    else:
        requested = np.asarray(cells, dtype=np.int64).ravel()
    keep = np.zeros(g.cells.num, dtype=bool)
    keep[requested] = True
    cell_ids = np.flatnonzero(keep)
    new_cell = np.full(g.cells.num, -1, dtype=np.int64)
    new_cell[cell_ids] = np.arange(cell_ids.size)

    face_ids = _surviving_faces(g, keep)
    new_face = np.full(g.faces.num, -1, dtype=np.int64)
    new_face[face_ids] = np.arange(face_ids.size)
    node_ids = _surviving_nodes(g, face_ids)
    new_node = np.full(g.nodes.num, -1, dtype=np.int64)
    new_node[node_ids] = np.arange(node_ids.size)

    cell_faces, cell_tags, face_pos = _subgrid_cell_faces(g, cell_ids, new_face)
    face_nodes, node_pos = _subgrid_face_nodes(g, face_ids, new_node)
    neighbors = np.where(
        g.faces.neighbors[face_ids] >= 0,
        new_cell[np.maximum(g.faces.neighbors[face_ids], 0)],
        -1,
    )

    sub = Grid(
        cells=Cells(
            num=cell_ids.size,
            facePos=np.asarray(face_pos, dtype=np.int64),
            faces=np.asarray(cell_faces, dtype=np.int64),
            tags=cell_tags,
            indexMap=g.cells.indexMap[cell_ids] if g.cells.indexMap is not None else cell_ids,
            volumes=(
                g.cells.volumes[cell_ids].copy()
                if g.cells.volumes is not None else None
            ),
            centroids=(
                g.cells.centroids[cell_ids].copy()
                if g.cells.centroids is not None else None
            ),
        ),
        faces=Faces(
            num=face_ids.size,
            nodePos=np.asarray(node_pos, dtype=np.int64),
            nodes=np.asarray(face_nodes, dtype=np.int64),
            neighbors=neighbors.astype(np.int64),
            tag=(
                g.faces.tag[face_ids].copy()
                if g.faces.tag is not None else None
            ),
            areas=(
                g.faces.areas[face_ids].copy()
                if g.faces.areas is not None else None
            ),
            normals=(
                g.faces.normals[face_ids].copy()
                if g.faces.normals is not None else None
            ),
            centroids=(
                g.faces.centroids[face_ids].copy()
                if g.faces.centroids is not None else None
            ),
        ),
        nodes=Nodes(num=node_ids.size, coords=g.nodes.coords[node_ids].copy()),
        cartDims=g.cartDims,
        griddim=g.griddim,
        type=list(g.type) + ["extractSubgrid"],
    )
    sub.cells.global_ = cell_ids.copy()
    sub.faces.global_ = face_ids.copy()
    sub.nodes.global_ = node_ids.copy()
    # `global` is an MRST field name but a Python keyword, so expose both the
    # convenient global_ spelling and exact dynamic attribute.
    setattr(sub.cells, "global", cell_ids.copy())
    setattr(sub.faces, "global", face_ids.copy())
    setattr(sub.nodes, "global", node_ids.copy())

    if hasattr(g.cells, "cpnodes"):
        sub.cells.cpnodes = new_node[np.asarray(g.cells.cpnodes)[cell_ids]]
        if np.any(sub.cells.cpnodes < 0):
            raise ValueError("too many nodes removed to preserve corner-point nodes")
    extras = {
        name: value for name, value in vars(g).items()
        if name not in {"cells", "faces", "nodes", "griddim", "type",
                        "cartDims", "edges", "nnc"}
    }
    for name, value in extras.items():
        setattr(sub, name, value)
    nnc = getattr(g, "nnc", None)
    if nnc is not None:
        is_dict = isinstance(nnc, dict)
        pairs = np.asarray(nnc["cells"] if is_dict else nnc.cells,
                           dtype=np.int64)
        keep_nnc = np.all(new_cell[pairs] >= 0, axis=1)
        values = dict(nnc) if is_dict else vars(nnc).copy()
        values["cells"] = new_cell[pairs[keep_nnc]]
        for name, value in list(values.items()):
            arr = np.asarray(value)
            if name != "cells" and arr.ndim and arr.shape[0] == pairs.shape[0]:
                values[name] = arr[keep_nnc]
        if is_dict:
            sub.nnc = values
        else:
            from types import SimpleNamespace
            sub.nnc = SimpleNamespace(**values)
    if return_mapping:
        return sub, cell_ids, face_ids, node_ids
    return sub


def _surviving_faces(g: Grid, keep: np.ndarray) -> np.ndarray:
    neighbors = g.faces.neighbors
    touches = np.zeros(g.faces.num, dtype=bool)
    for col in (0, 1):
        ok = neighbors[:, col] >= 0
        face_ix = np.flatnonzero(ok)
        touches[face_ix[keep[neighbors[ok, col]]]] = True
    return np.flatnonzero(touches)


def _surviving_nodes(g: Grid, face_ids: np.ndarray) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [g.faces.nodes[g.faces.nodePos[f] : g.faces.nodePos[f + 1]] for f in face_ids]
        )
    )


def _subgrid_cell_faces(
    g: Grid,
    cell_ids: np.ndarray,
    new_face: np.ndarray,
) -> tuple[list[int], Optional[np.ndarray], list[int]]:
    cell_faces = []
    tags = []
    face_pos = [0]
    for cell in cell_ids:
        lo, hi = g.cells.facePos[cell], g.cells.facePos[cell + 1]
        cell_faces.extend(new_face[g.cells.faces[lo:hi]])
        if g.cells.tags is not None:
            tags.extend(g.cells.tags[lo:hi])
        face_pos.append(len(cell_faces))
    tag_array = np.asarray(tags, dtype=np.int64) if g.cells.tags is not None else None
    return cell_faces, tag_array, face_pos


def _subgrid_face_nodes(
    g: Grid,
    face_ids: np.ndarray,
    new_node: np.ndarray,
) -> tuple[list[int], list[int]]:
    face_nodes = []
    node_pos = [0]
    for face in face_ids:
        lo, hi = g.faces.nodePos[face], g.faces.nodePos[face + 1]
        face_nodes.extend(new_node[g.faces.nodes[lo:hi]])
        node_pos.append(len(face_nodes))
    return face_nodes, node_pos


if __name__ == "__main__":

    def _tiny_grid() -> Grid:
        coords = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
        )
        face_nodes = np.array([0, 3, 1, 4, 2, 5, 0, 1, 1, 2, 3, 4, 4, 5])
        node_pos = np.arange(0, 15, 2)
        neighbors = np.array(
            [[-1, 0], [0, 1], [1, -1], [-1, 0], [-1, 1], [0, -1], [1, -1]]
        )
        cell_faces = np.array([0, 1, 3, 5, 1, 2, 4, 6])
        face_pos = np.array([0, 4, 8])
        return Grid(
            cells=Cells(num=2, facePos=face_pos, faces=cell_faces, indexMap=np.arange(2)),
            faces=Faces(num=7, nodePos=node_pos, nodes=face_nodes, neighbors=neighbors),
            nodes=Nodes(num=6, coords=coords),
            griddim=2,
            type=["tiny"],
        )

    def test_check_passes_on_valid_grid() -> None:
        grid = _tiny_grid()
        grid.check()
        assert grid.spatial_dim == 2
        print("ok test_check_passes_on_valid_grid")

    def test_accessors() -> None:
        grid = _tiny_grid()
        assert np.array_equal(grid.cell_faces(0), [0, 1, 3, 5])
        assert np.array_equal(grid.cell_faces(1), [1, 2, 4, 6])
        assert np.array_equal(grid.face_nodes(1), [1, 4])
        assert np.array_equal(grid.cell_no(), [0, 0, 0, 0, 1, 1, 1, 1])
        print("ok test_accessors")

    def test_extract_subgrid() -> None:
        grid = extract_subgrid(_tiny_grid(), [0])
        grid.check()
        assert grid.cells.num == 1
        assert grid.faces.num == 4
        assert np.all(grid.faces.neighbors.min(axis=1) < 0)
        print("ok test_extract_subgrid")

    test_check_passes_on_valid_grid()
    test_accessors()
    test_extract_subgrid()
    print("\nAll tests passed!")
