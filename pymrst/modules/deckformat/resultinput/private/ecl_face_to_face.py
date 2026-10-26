"""MRST ``deckformat/resultinput/private/eclFaceToFace.m``.

Where each face's flux sits in an ECLIPSE restart: the faces between
Cartesian neighbours in I, J and K (`iI`, `iJ`, `iK`, as masks over faces),
the active cell whose `+` flux array holds each (`iIe`, `iJe`, `iKe`), and
the sign that turns that flux into the face's orientation (`sI`, `sJ`,
`sK`); for the non-neighbour connections `NNC` (pairs of 1-based global
cell numbers) the faces (`iN`), their NNC rows (`iNe`) and signs (`sN`).
Warns, as MRST does, when connections and faces do not match.

Faces, cells and NNC rows are 0-based here; a boundary neighbour is -1.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np


def ecl_face_to_face(G: Any, NNC: Optional[np.ndarray] = None, *, neighbors: Any = None,
                     checkConsistency: bool = True) -> SimpleNamespace:
    """MRST `ix = eclFaceToFace(G, NNC, 'neighbors', N, 'checkConsistency', true)`."""
    if NNC is None:
        warnings.warn("No NNC given ...")
    nc = G.cells.num
    dims = np.asarray(G.cartDims, dtype=np.int64).ravel()
    N = np.array(G.faces.neighbors if neighbors is None else neighbors, dtype=np.int64)
    is_pos = N[:, 1] > N[:, 0]
    N[~is_pos] = N[~is_pos][:, ::-1]
    ie = np.all(N >= 0, axis=1)
    imap = np.asarray(G.cells.indexMap, dtype=np.int64).ravel()
    Ng = np.zeros_like(N)
    Ng[ie] = imap[N[ie]]
    diff = Ng[:, 1] - Ng[:, 0]
    ix = SimpleNamespace(iI=ie & (diff == 1), iJ=ie & (diff == dims[0]),
                         iK=ie & (diff == dims[0]*dims[1]))
    glob2act = np.full(int(np.prod(dims)), -1, dtype=np.int64)
    glob2act[imap] = np.arange(nc)
    ix.iIe = glob2act[Ng[ix.iI, 0]]
    ix.iJe = glob2act[Ng[ix.iJ, 0]]
    ix.iKe = glob2act[Ng[ix.iK, 0]]
    ix.sI = 2*is_pos[ix.iI] - 1
    ix.sJ = 2*is_pos[ix.iJ] - 1
    ix.sK = 2*is_pos[ix.iK] - 1
    if NNC is not None and np.size(NNC):
        nnc = glob2act[np.asarray(NNC, dtype=np.int64) - 1]
        is_pos_n = nnc[:, 1] > nnc[:, 0]
        nnc[~is_pos_n] = nnc[~is_pos_n][:, ::-1]
        assert np.all(nnc >= 0), "Some NNC-cells are not active in grid"
        lookup = {(a, b): k for k, (a, b) in enumerate(nnc.tolist())}
        hits = [(f, lookup.get((int(N[f, 0]), int(N[f, 1]))))
                for f in np.flatnonzero(ie)]
        hits = [(f, k) for f, k in hits if k is not None]
        ix.iN = np.array([f for f, _ in hits], dtype=np.int64)
        ix.iNe = np.array([k for _, k in hits], dtype=np.int64)
        ix.sN = (2*is_pos[ix.iN] - 1)*(2*is_pos_n[ix.iNe] - 1)
    if checkConsistency:
        _check_consistency(ix, NNC, ie)
    return ix


eclFaceToFace = ecl_face_to_face


def _check_consistency(ix: SimpleNamespace, NNC: Any, ie: np.ndarray) -> None:
    if NNC is None or not np.size(NNC):
        return
    n_fail = np.shape(NNC)[0] - ix.iN.size
    if n_fail != 0:
        warnings.warn(f"Non compatible grid! Found {n_fail} NNCs not occuring in grid.")
    ii = ix.iI | ix.iJ | ix.iK
    ii[ix.iN] = True
    n_fail = int(np.count_nonzero(~ii[ie]))
    if n_fail > 0:
        warnings.warn(f"Non compatible grid! Found {n_fail} interior grid-faces not "
                      f"occuring in output.")


__all__ = ["ecl_face_to_face", "eclFaceToFace"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid

    def test_cartesian_faces_find_their_flux_arrays():
        G = cart_grid([2, 2, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ix = ecl_face_to_face(G)
        assert ix.iI.sum() == 2 and ix.iJ.sum() == 2 and ix.iK.sum() == 0
        np.testing.assert_array_equal(np.sort(ix.iIe), [0, 2])
        assert np.all(ix.sI == 1)
        print("ok test_cartesian_faces_find_their_flux_arrays")

    def test_an_nnc_matches_its_face():
        G = cart_grid([2, 1, 1])
        ix = ecl_face_to_face(G, np.array([[2, 1]]))
        face = int(np.flatnonzero(ix.iI)[0])
        np.testing.assert_array_equal(ix.iN, [face])
        assert ix.sN[0] == -1, "the NNC is written from cell 2 to cell 1"
        print("ok test_an_nnc_matches_its_face")

    test_cartesian_faces_find_their_flux_arrays()
    test_an_nnc_matches_its_face()
    print("\nAll tests passed!")
