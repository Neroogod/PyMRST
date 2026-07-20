"""Batched eigenvalue helpers matching MRST ``multiEig``/``multiSymmEig``."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _block_sizes(a: np.ndarray, sz=None) -> np.ndarray:
    if sz is not None:
        out = np.asarray(sz, dtype=np.int64).ravel()
        if np.any(out <= 0):
            raise ValueError("matrix block sizes must be positive")
        return out
    n = int(round(np.sqrt(a.size)))
    if n * n != a.size:
        raise ValueError("sz is required when A does not contain one square block")
    return np.array([n], dtype=np.int64)


def _blocks(a, sz=None) -> Iterable[np.ndarray]:
    flat = np.asarray(a).ravel(order="F")
    sizes = _block_sizes(flat, sz)
    need = int(np.sum(sizes * sizes))
    if flat.size != need:
        raise ValueError(f"A has {flat.size} entries, expected {need}")
    off = 0
    for n in sizes:
        n = int(n)
        block = flat[off:off + n * n].reshape((n, n), order="F")
        off += n * n
        yield block


def multi_eig(a, sz=None, vectors: bool = False, left: bool = False):
    """Solve a sequence of small dense eigenvalue problems.

    This is the Python equivalent of MRST ``multiEig``. With ``vectors=False``
    it returns the concatenated eigenvalues. With ``vectors=True`` it also
    returns concatenated right eigenvectors. ``left=True`` additionally returns
    left eigenvectors, computed as right eigenvectors of ``A.T``.
    """
    vals, right, left_vecs = [], [], []
    for block in _blocks(a, sz):
        if vectors:
            d, v = np.linalg.eig(block)
            right.append(v.reshape(-1, order="F"))
            if left:
                _, w = np.linalg.eig(block.T)
                left_vecs.append(w.reshape(-1, order="F"))
        else:
            d = np.linalg.eigvals(block)
        vals.append(d)
    dcat = np.concatenate(vals)
    if not vectors:
        return dcat
    vcat = np.concatenate(right)
    if not left:
        return dcat, vcat
    return dcat, vcat, np.concatenate(left_vecs)


def multi_symm_eig(a, sz=None, vectors: bool = False):
    """Solve a sequence of small symmetric dense eigenvalue problems.

    Equivalent to MRST ``multiSymmEig`` for the returned values/layout. NumPy's
    ``eigh`` returns eigenvalues in ascending order, matching LAPACK's symmetric
    eigensolver convention.
    """
    vals, vecs = [], []
    for block in _blocks(a, sz):
        if vectors:
            d, v = np.linalg.eigh(block)
            vecs.append(v.reshape(-1, order="F"))
        else:
            d = np.linalg.eigvalsh(block)
        vals.append(d)
    dcat = np.concatenate(vals)
    if not vectors:
        return dcat
    return dcat, np.concatenate(vecs)


__all__ = ["multi_eig", "multi_symm_eig"]


if __name__ == "__main__":

    def test_multi_symm_eig_two_blocks():
        a = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = np.diag([3.0, 5.0, 7.0])
        flat = np.concatenate([a.reshape(-1, order="F"),
                               b.reshape(-1, order="F")])
        d, v = multi_symm_eig(flat, [2, 3], vectors=True)
        assert np.allclose(d[:2], [1.0, 3.0])
        assert np.allclose(d[2:], [3.0, 5.0, 7.0])
        assert v.size == 4 + 9
        print("ok test_multi_symm_eig_two_blocks")

    def test_multi_eig_general():
        a = np.array([[0.0, -1.0], [1.0, 0.0]])
        d = multi_eig(a)
        assert np.allclose(np.sort_complex(d), [-1j, 1j])
        print("ok test_multi_eig_general")

    test_multi_symm_eig_two_blocks()
    test_multi_eig_general()
    print("\nAll tests passed!")
