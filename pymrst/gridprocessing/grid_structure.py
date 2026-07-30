"""The unstructured grid structure, ported from MRST `grid_structure`."""

from dataclasses import dataclass, field
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


def extract_subgrid(g: Grid, cells) -> Grid:
    """Return the grid made of `cells` alone, ported from `extractSubgrid`."""
    keep = np.zeros(g.cells.num, dtype=bool)
    keep[np.asarray(cells, dtype=np.int64).ravel()] = True
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

    return Grid(
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
