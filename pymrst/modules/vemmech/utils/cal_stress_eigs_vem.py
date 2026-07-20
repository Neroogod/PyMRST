"""MRST ``calStressEigsVEM.m`` principal-stress recovery."""

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.utils.eigs import multi_symm_eig


def _stress_tensor_from_voigt(g: Grid, stress: np.ndarray) -> np.ndarray:
    stress = np.asarray(stress, dtype=float)
    if g.griddim == 2:
        return np.stack(
            [stress[:, 0], stress[:, 2], stress[:, 2], stress[:, 1]],
            axis=1,
        ).reshape(-1, 2, 2)
    return np.stack(
        [
            stress[:, 0], stress[:, 5], stress[:, 4],
            stress[:, 5], stress[:, 1], stress[:, 3],
            stress[:, 4], stress[:, 3], stress[:, 2],
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def cal_stress_eigs_vem(
    g: Grid,
    stress: np.ndarray,
    use_c: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stress eigenvalues/eigenvectors cell by cell."""
    tensors = _stress_tensor_from_voigt(g, stress)
    dim = g.griddim
    if use_c:
        flat = np.concatenate([t.reshape(-1, order="F") for t in tensors])
        vals, vecs = multi_symm_eig(flat, np.full(g.cells.num, dim), vectors=True)
        return vals.reshape(g.cells.num, dim), vecs.reshape(g.cells.num, dim * dim)
    sigm = np.empty((g.cells.num, dim))
    evec = np.empty((g.cells.num, dim * dim))
    for c, tensor in enumerate(tensors):
        vals, vecs = np.linalg.eig(tensor)
        sigm[c] = vals.real
        evec[c] = vecs.real.reshape(-1, order="F")
    return sigm, evec

__all__ = ["cal_stress_eigs_vem"]
