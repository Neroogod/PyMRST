"""Port of MRST ``natvars/ReducedLinearizedSystem.m``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class ReducedLinearizedSystem:
    """Linearized system with Schur-complement elimination."""

    A: sp.spmatrix | np.ndarray | None = None
    b: np.ndarray | None = None
    keepNum: int = 0
    reorder: np.ndarray | None = None
    B: sp.spmatrix | None = field(default=None, init=False)
    C: sp.spmatrix | None = field(default=None, init=False)
    D: sp.spmatrix | None = field(default=None, init=False)
    E: sp.spmatrix | None = field(default=None, init=False)
    f: np.ndarray | None = field(default=None, init=False)
    h: np.ndarray | None = field(default=None, init=False)
    E_U: sp.spmatrix | None = field(default=None, init=False)
    E_L: sp.spmatrix | None = field(default=None, init=False)
    _permutation: np.ndarray | None = field(default=None, init=False)

    @property
    def keep_num(self) -> int:
        return self.keepNum

    @keep_num.setter
    def keep_num(self, value: int) -> None:
        self.keepNum = int(value)

    def assembleSystem(self):
        """Assemble and reduce the linear system."""
        if self.A is None:
            return self
        A = sp.csr_matrix(self.A)
        b = np.asarray(self.b, dtype=float).reshape(-1)
        if self.reorder is not None:
            reorder = np.asarray(self.reorder, dtype=int).reshape(-1)
            A = A[:, reorder]
            self._permutation = reorder
        n = A.shape[1]
        nk = int(self.keepNum)
        if nk <= 0 or nk >= n:
            self.A = A
            self.b = b
            self.B = A
            self.f = b
            return self

        self.B = A[:nk, :nk].tocsr()
        self.C = A[:nk, nk:].tocsr()
        self.D = A[nk:, :nk].tocsr()
        self.E = A[nk:, nk:].tocsr()
        self.f = b[:nk]
        self.h = b[nk:]
        if self.E.shape[0] == 0:
            self.A = self.B
            self.b = self.f
            return self
        solved_D = spla.spsolve(self.E, self.D.toarray())
        solved_h = spla.spsolve(self.E, self.h)
        self.A = (self.B - self.C @ sp.csr_matrix(solved_D)).tocsr()
        self.b = self.f - np.asarray(self.C @ solved_h).reshape(-1)
        self.E_U = self.E
        self.E_L = sp.eye(self.E.shape[0], format="csr")
        return self

    assemble_system = assembleSystem

    def processResultAfterSolve(self, p, report=None):
        """Recover eliminated variables after solving the reduced system."""
        p = np.asarray(p, dtype=float).reshape(-1)
        if self.E is None or self.E.shape[0] == 0:
            return p, report
        s = spla.spsolve(self.E, self.h - self.D @ p)
        out = np.concatenate([p, np.asarray(s, dtype=float).reshape(-1)])
        if self._permutation is not None:
            backmap = np.zeros_like(self._permutation)
            backmap[self._permutation] = np.arange(self._permutation.size)
            out = out[backmap]
        return out, report

    process_result_after_solve = processResultAfterSolve


__all__ = ["ReducedLinearizedSystem"]
