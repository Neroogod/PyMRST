"""MRST ``agglom/utils/signOfFineFacesOnCoarseFaces.m``: +1 where a fine
face's first cell lies in its coarse face's first block, -1 otherwise, and
the coarse face each fine sub-face belongs to. 0-based; the outside is -1.
"""

from typing import Any, Tuple

import numpy as np


def sign_of_fine_faces_on_coarse_faces(G: Any, CG: Any, nsub: np.ndarray,
                                       sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """MRST `[sgn, coarse_f] = signOfFineFacesOnCoarseFaces(G, CG, nsub, sub)`."""
    nsub = np.asarray(nsub, dtype=np.int64)
    coarse_f = np.repeat(np.arange(nsub.size), nsub)
    out_blck_c = np.repeat(np.asarray(CG.faces.neighbors, dtype=np.int64)[:, 0], nsub)
    p = np.concatenate([[-1], np.asarray(CG.partition, dtype=np.int64)])
    out_blck_f = p[np.asarray(G.faces.neighbors, dtype=np.int64)[sub, 0] + 1]
    sgn = 2*(out_blck_f == out_blck_c).astype(float) - 1
    return sgn, coarse_f


signOfFineFacesOnCoarseFaces = sign_of_fine_faces_on_coarse_faces
__all__ = ["sign_of_fine_faces_on_coarse_faces", "signOfFineFacesOnCoarseFaces"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_ui import partition_ui
    from pymrst.modules.coarsegrid.sub_faces import sub_faces
    from pymrst.modules.coarsegrid.utils.coarse_grid_as_grid import coarse_grid_as_grid
    from pymrst.modules.coarsegrid.utils.fine_to_coarse_sign import fine_to_coarse_sign

    G = compute_geometry(cart_grid([4, 3], [4.0, 3.0]))
    CG = coarse_grid_as_grid(generate_coarse_grid(G, partition_ui(G, [2, 3])))

    def test_agrees_with_fine_to_coarse_sign():
        nsub, sub = sub_faces(G, CG)
        sgn, coarse_f = sign_of_fine_faces_on_coarse_faces(G, CG, nsub, sub)
        assert np.array_equal(sgn, fine_to_coarse_sign(CG.coarse_grid))
        assert coarse_f.size == sub.size and coarse_f.max() == CG.faces.num - 1
        print("ok test_agrees_with_fine_to_coarse_sign")

    test_agrees_with_fine_to_coarse_sign()
    print("\nAll tests passed!")
