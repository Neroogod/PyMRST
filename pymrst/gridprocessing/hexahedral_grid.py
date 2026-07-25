"""Build an unstructured hexahedral grid from cell-node connectivity (MRST
`hexahedralGrid`).

Given a list of hexahedral cells - each named by its eight corner nodes in MRST's
corner order - and the node coordinates, this assembles the full grid: it finds
which cells share a face and builds the face-to-cell map. The trick (MRST's) is a
canonical face orientation: each of a hex's six quad faces is rewritten so its
two smallest node indices come first, in a fixed rotational sense; two cells that
share a face then produce the *identical* node quad, so a `sortrows` groups them
and the shared face is discovered without any geometric tolerance. This is how a
grid coming from an external hex mesher (rather than `cartGrid`/`tensorGrid`)
enters MRST.
"""

import numpy as np

from pymrst.gridprocessing.private.add_faces import add_faces
from pymrst.gridprocessing.private.empty_grid import empty_grid
from pymrst.utils.gridtools.rldecode import rldecode
from pymrst.utils.gridtools.rlencode import rlencode

# the six quad faces of a hex, as columns into the 8 corner nodes (0-based;
# MRST's Wst/Est/Sth/Nth/Bot/Top stacking order is preserved for the tags)
_FACES = {
    "Wst": [7, 6, 5, 4],
    "Est": [0, 1, 2, 3],
    "Sth": [3, 7, 4, 0],
    "Nth": [1, 5, 6, 2],
    "Bot": [4, 5, 1, 0],
    "Top": [3, 2, 6, 7],
}


def hexahedral_grid(hexes, coords):
    """Assemble a 3D grid from hex connectivity `hexes` (nc x 8, 0-based node
    indices) and node coordinates `coords` (n_nodes x 3).

    Note the argument order is `(hexes, coords)`; MRST's `hexahedralGrid` takes
    `(P, H)` - points first. The node indices here are 0-based (MRST's are
    1-based).
    """
    H = np.asarray(hexes, dtype=np.int64)
    P = np.asarray(coords, dtype=float)
    nc = H.shape[0]

    hfaces = np.vstack([H[:, cols] for cols in _FACES.values()])   # (6nc, 4)

    # rotate each quad so its smallest node index is first
    imin = np.argmin(hfaces, axis=1)
    rot = (np.arange(4)[None, :] + imin[:, None]) % 4
    hfaces = np.take_along_axis(hfaces, rot, axis=1)

    # put the smaller of the 2nd/4th nodes into column 2; the choice (1 or 2)
    # is the half-face's orientation sign, and separates a shared face's two
    # sides into the two neighbour columns
    sign = np.argmin(hfaces[:, [1, 3]], axis=1)                    # 0 or 1
    flip = sign == 1
    hfaces[flip] = hfaces[flip][:, [0, 3, 2, 1]]

    cell_no = np.tile(np.arange(1, nc + 1), 6)                     # 1-based
    tag = np.repeat(np.arange(1, 7), nc)

    order = np.lexsort(hfaces.T[::-1])                            # sortrows
    hf_sorted = hfaces[order]
    fnodes, counts = rlencode(hf_sorted)                          # unique faces
    nf = counts.size
    face_of = rldecode(np.arange(nf), counts)                     # half-face -> face

    col = sign[order]                                            # 0/1 neighbour column
    neigh = np.zeros((nf, 2), dtype=np.int64)
    neigh[face_of, col] = cell_no[order]                        # 1-based; 0 = outside
    hftag = np.zeros((nf, 2), dtype=np.int64)
    hftag[face_of, col] = tag[order]

    neigh = neigh - 1                                           # 0-based; outside -> -1
    nnodes = np.full(nf, 4, dtype=np.int64)

    g = empty_grid(nc, P, griddim=3)
    g, _ = add_faces(g, fnodes.ravel(), nnodes, neigh)
    g.type = ["hexahedralGrid"]
    g.hftag = hftag
    return g


def _two_cube_mesh():
    """Two unit cubes stacked in z, sharing their mid plane (12 nodes)."""
    # z-layers of a 2x2 (x,y) corner set
    layer = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    coords = np.vstack([np.column_stack([layer, np.full(4, z)]) for z in (0, 1, 2)])
    a, b, c = np.arange(4), np.arange(4, 8), np.arange(8, 12)
    hexes = np.vstack([np.concatenate([a, b]), np.concatenate([b, c])])
    return hexes, coords


if __name__ == "__main__":
    def test_two_cubes_share_one_internal_face():
        """Two stacked cubes: 2 cells, 11 faces, exactly one internal face
        joining them, 10 boundary faces, 12 nodes."""
        hexes, coords = _two_cube_mesh()
        g = hexahedral_grid(hexes, coords)
        assert g.cells.num == 2
        assert g.nodes.num == 12
        neigh = np.asarray(g.faces.neighbors)
        internal = np.all(neigh >= 0, axis=1)
        assert g.faces.num == 11, g.faces.num
        assert internal.sum() == 1, internal.sum()
        assert set(neigh[internal][0]) == {0, 1}
        assert (~internal).sum() == 10
        print(f"ok test_two_cubes_share_one_internal_face "
              f"({g.faces.num} faces, 1 internal)")

    def test_single_hex_is_all_boundary():
        """A lone cube has six boundary faces and no internal ones."""
        layer = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        coords = np.vstack([np.column_stack([layer, np.full(4, z)]) for z in (0, 1)])
        g = hexahedral_grid(np.arange(8)[None, :], coords)
        assert g.cells.num == 1 and g.faces.num == 6
        neigh = np.asarray(g.faces.neighbors)
        assert np.all(np.any(neigh == -1, axis=1)), "every face is on the boundary"
        print("ok test_single_hex_is_all_boundary")

    def test_every_face_has_four_nodes():
        hexes, coords = _two_cube_mesh()
        g = hexahedral_grid(hexes, coords)
        counts = np.diff(np.asarray(g.faces.nodePos))
        assert np.all(counts == 4), counts
        print("ok test_every_face_has_four_nodes")

    test_two_cubes_share_one_internal_face()
    test_single_hex_is_all_boundary()
    test_every_face_has_four_nodes()
    print("\nAll tests passed!")
