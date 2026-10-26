"""MRST ``coarsegrid/subFaces.m``: the fine faces making up each coarse face -
`nsub[k]` of them for coarse face `k`, listed consecutively in `sub`.

`cg` is a `CoarseGrid` or the grid view `coarse_grid_as_grid` builds from one.
"""

from typing import Any, Tuple

import numpy as np


def sub_faces(g: Any, cg: Any) -> Tuple[np.ndarray, np.ndarray]:
    """MRST `[nsub, sub] = subFaces(g, cg)`."""
    cg = getattr(cg, "coarse_grid", cg)
    assert cg.fconn is not None, (
        "This edition of function 'subFaces' is only supported when accompanied by the "
        "revised edition of function 'generateCoarseGrid'")
    fconn = np.asarray(cg.fconn, dtype=np.int64)
    assert fconn.size == 0 or fconn.max() < g.faces.num, (
        "The coarse grid does not appear to match the fine-scale grid.")
    return np.diff(np.asarray(cg.conn_pos, dtype=np.int64)), fconn


subFaces = sub_faces
__all__ = ["sub_faces", "subFaces"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_ui import partition_ui
    from pymrst.modules.coarsegrid.utils.coarse_grid_as_grid import coarse_grid_as_grid

    G = compute_geometry(cart_grid([4, 2], [4.0, 2.0]))
    CG = generate_coarse_grid(G, partition_ui(G, [2, 1]))

    def test_every_fine_face_on_a_coarse_face_is_listed_once():
        nsub, sub = sub_faces(G, CG)
        assert nsub.sum() == sub.size and np.unique(sub).size == sub.size
        assert np.array_equal(sub_faces(G, coarse_grid_as_grid(CG))[1], sub)
        print("ok test_every_fine_face_on_a_coarse_face_is_listed_once")

    def test_the_shared_coarse_face_has_two_fine_faces():
        nsub, _ = sub_faces(G, CG)
        interior = np.all(np.asarray(CG.neighbors) >= 0, axis=1)
        assert np.array_equal(nsub[interior], [2])
        print("ok test_the_shared_coarse_face_has_two_fine_faces")

    test_every_fine_face_on_a_coarse_face_is_listed_once()
    test_the_shared_coarse_face_has_two_fine_faces()
    print("\nAll tests passed!")
