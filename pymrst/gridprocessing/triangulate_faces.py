"""Triangulate polygonal grid faces (MRST `triangulateFaces`).

Splits each selected face into simplices through a new node at the face centroid:
in 3D a `j`-sided polygon becomes `j` triangles `(n_k, n_{k+1}, centroid)`; in 2D
an edge becomes two sub-edges through its midpoint. Every sub-face inherits the
parent face's two cell neighbours, so the grid stays consistent - only the faces
are refined. This is what a solver or plotter that needs simplicial faces (VEM,
some MPFA variants, triangle-based rendering) runs first.
"""

from typing import Optional

import numpy as np

from pymrst.gridprocessing.private.add_faces import add_faces
from pymrst.gridprocessing.private.remove_faces import remove_faces
from pymrst.utils.mcolon import mcolon


def _inherited_tags(g, new_neighbors: np.ndarray, parents: np.ndarray,
                    parent_of_new: np.ndarray):
    """Cell-face tags for the replacement faces, taken from their parents.

    Returns ``None`` when the grid carries no tags, which keeps the behaviour
    unchanged for grids that never had them. Ordering follows `add_faces`:
    the neighbor table flattened column first, with entries for cell -1 dropped.
    """
    if g.cells.tags is None:
        return None
    cell_no = g.cell_no()
    lookup = {
        (int(cell), int(face)): tag
        for cell, face, tag in zip(cell_no, np.asarray(g.cells.faces),
                                   np.asarray(g.cells.tags))
    }
    parent_face = np.asarray(parents, dtype=np.int64)[parent_of_new]
    columns = []
    for side in (0, 1):
        cells = new_neighbors[:, side]
        columns.append([
            lookup.get((int(cell), int(face)), 0) if cell >= 0 else 0
            for cell, face in zip(cells, parent_face)
        ])
    flat = np.concatenate([np.asarray(columns[0]), np.asarray(columns[1])])
    return flat[np.concatenate([new_neighbors[:, 0], new_neighbors[:, 1]]) >= 0]


def triangulate_faces(g, faces: Optional[np.ndarray] = None):
    """Return a grid with the given `faces` (default all) triangulated."""
    coords = np.asarray(g.nodes.coords, dtype=float)
    d = coords.shape[1]
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)

    f = np.arange(g.faces.num) if faces is None else np.asarray(faces, np.int64)

    # gather the selected faces' nodes as one flat run
    counts = (node_pos[f + 1] - node_pos[f]).astype(np.int64)
    seg = fnodes[mcolon(node_pos[f], node_pos[f + 1])]
    face_of_node = np.repeat(np.arange(f.size, dtype=np.int64), counts)
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    # a new node at each face's centroid
    mids = np.column_stack([
        np.bincount(face_of_node, weights=coords[seg, dim], minlength=f.size)
        for dim in range(d)
    ]) / counts[:, None]
    mid_id = g.nodes.num + np.arange(f.size, dtype=np.int64)
    mid_of_node = mid_id[face_of_node]

    if d == 3:
        # fan each face into triangles (node, next node round the face, centroid)
        successor = np.arange(1, seg.size + 1, dtype=np.int64)
        successor[starts[1:] - 1] = starts[:-1]
        new_nodes = np.stack([seg, seg[successor], mid_of_node], axis=1).ravel()
        new_counts = np.full(seg.size, 3, dtype=np.int64)
        new_neigh = neighbors[f][face_of_node]
    else:                                            # 2D: split each edge
        first, second = seg[starts[:-1]], seg[starts[:-1] + 1]
        new_nodes = np.stack([first, mid_id, mid_id, second], axis=1).ravel()
        new_counts = np.full(2 * f.size, 2, dtype=np.int64)
        new_neigh = np.repeat(neighbors[f], 2, axis=0)

    # a piece of a tagged face lies on the same side of the grid as the face
    # it came from, so the cell-face tag has to travel with it
    new_tags = _inherited_tags(g, np.asarray(new_neigh, dtype=np.int64).reshape(-1, 2),
                               f, face_of_node if d == 3 else np.repeat(
                                   np.arange(f.size, dtype=np.int64), 2))

    # register the centroid nodes, then swap the faces
    g.nodes.coords = np.vstack([coords, mids])
    g.nodes.num = g.nodes.coords.shape[0]

    g, _ = remove_faces(g, f)
    g, _ = add_faces(g, np.asarray(new_nodes, dtype=np.int64),
                     np.asarray(new_counts, dtype=np.int64),
                     np.asarray(new_neigh, dtype=np.int64).reshape(-1, 2),
                     tags=new_tags)
    g.type = list(getattr(g, "type", [])) + ["triangulateFaces"]
    return g


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_each_quad_face_becomes_four_triangles():
        """Every face of a Cartesian grid is a quad -> 4 triangles; one centroid
        node is added per face."""
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        nf0, nn0 = g.faces.num, g.nodes.num
        g2 = triangulate_faces(g)
        assert g2.faces.num == 4 * nf0, (g2.faces.num, nf0)
        assert g2.nodes.num == nn0 + nf0
        counts = np.diff(np.asarray(g2.faces.nodePos))
        assert np.all(counts == 3), "every new face is a triangle"
        print(f"ok test_each_quad_face_becomes_four_triangles "
              f"({nf0} -> {g2.faces.num} faces)")

    def test_neighbours_are_preserved():
        """Each triangle keeps its parent face's two cell neighbours, so the
        neighbour-pair multiset is the original with each pair repeated 4x."""
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        parent = np.sort(np.asarray(g.faces.neighbors), axis=1)
        g2 = triangulate_faces(g)
        child = np.sort(np.asarray(g2.faces.neighbors), axis=1)
        from collections import Counter
        cp = Counter(map(tuple, parent.tolist()))
        cc = Counter(map(tuple, child.tolist()))
        assert all(cc[k] == 4 * v for k, v in cp.items()), (cp, cc)
        print("ok test_neighbours_are_preserved")

    def test_subset_of_faces_only():
        """Triangulating a subset leaves the other faces untouched."""
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        nf0 = g.faces.num
        g2 = triangulate_faces(g, np.array([0]))        # one quad -> 4 triangles
        assert g2.faces.num == nf0 - 1 + 4, g2.faces.num
        print("ok test_subset_of_faces_only")

    test_each_quad_face_becomes_four_triangles()
    test_neighbours_are_preserved()
    test_subset_of_faces_only()
    print("\nAll tests passed!")
