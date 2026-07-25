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


def triangulate_faces(g, faces: Optional[np.ndarray] = None):
    """Return a grid with the given `faces` (default all) triangulated."""
    coords = np.asarray(g.nodes.coords, dtype=float)
    d = coords.shape[1]
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)

    f = np.arange(g.faces.num) if faces is None else np.asarray(faces, np.int64)

    # a new node at each face's centroid
    mids = np.array([coords[fnodes[node_pos[i]:node_pos[i + 1]]].mean(axis=0)
                     for i in f])
    mid_id = g.nodes.num + np.arange(f.size, dtype=np.int64)

    new_nodes: list = []
    new_counts: list = []
    new_neigh: list = []
    for fi, face in enumerate(f):
        seg = fnodes[node_pos[face]:node_pos[face + 1]]
        m = seg.size
        mid = mid_id[fi]
        if d == 3:
            for k in range(m):
                new_nodes += [seg[k], seg[(k + 1) % m], mid]
                new_counts.append(3)
                new_neigh.append(neighbors[face])
        else:                                        # 2D: split the edge
            new_nodes += [seg[0], mid]
            new_nodes += [mid, seg[1]]
            new_counts += [2, 2]
            new_neigh += [neighbors[face], neighbors[face]]

    # register the centroid nodes, then swap the faces
    g.nodes.coords = np.vstack([coords, mids])
    g.nodes.num = g.nodes.coords.shape[0]

    g, _ = remove_faces(g, f)
    g, _ = add_faces(g, np.asarray(new_nodes, dtype=np.int64),
                     np.asarray(new_counts, dtype=np.int64),
                     np.asarray(new_neigh, dtype=np.int64).reshape(-1, 2))
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
