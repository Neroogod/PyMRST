"""Linear-solver modes for the mimetic hybrid system (MRST `mimetic/utils`).

The mimetic (mixed hybrid) discretisation assembles a saddle-point system in the
half-face fluxes `v`, cell pressures `p` and face-pressure Lagrange multipliers
`lam`:

    B v = f + C p - D lam ,   C' v = g ,   D' v = h .

`incompMimetic` offers three equivalent ways to solve it, ported here:

- `schur_complement_symm` - eliminate `v` (via `BI = B^-1`) and then the cell
  pressures (whose block `L = diag(C'BI C)` is diagonal), leaving a small SPD
  Schur system for `lam`. The default "hybrid" solver.
- `mixed_symm` - project to faces with the signed face map `Do` and solve the
  reduced mixed saddle-point system `[[Bm,Cm,Dm],[Cm',0,0],[Dm',0,0]]` directly.
- `tpf_symm` - the two-point-flux specialisation: when `Bm = Do'B Do` is diagonal
  (a two-point inner product) the mixed system collapses to a symmetric
  cell/face system.

`schur_complement_symm_fault` is the Schur solver with an extra fault-coupling
term `E` subtracted from the Schur matrix.
"""

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def _dense(a) -> np.ndarray:
    return a.toarray() if sp.issparse(a) else np.asarray(a, dtype=float)


def schur_complement_symm(BI, C, D, f, g, h, regularize: bool = False
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the hybrid system by Schur complement. Returns
    `(flux, press, lam, S)` - half-face fluxes, cell pressures, face pressures,
    and the Schur matrix."""
    BI, C, D = sp.csr_array(BI), sp.csr_array(C), sp.csr_array(D)
    f = np.asarray(f, dtype=float).ravel()
    g = np.asarray(g, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()
    ncell = g.size

    Df = sp.hstack([D, sp.csr_array(f[:, None])]).tocsr()
    BIDf = _dense(BI @ Df)                              # (nhf, nface+1)
    CtBIDf = _dense(C.T) @ BIDf
    DtBIDf = _dense(D.T) @ BIDf

    M = CtBIDf[:, :-1]                                  # C' BI D
    Mt = _dense(D.T @ BI @ C)                           # D' BI C == M'
    S = DtBIDf[:, :-1]                                  # D' BI D
    g = g - CtBIDf[:, -1]
    h = h - DtBIDf[:, -1]

    L = np.asarray((C.T @ BI @ C).diagonal(), dtype=float)
    if regularize:
        L[0] = 2.0 * L[0]
    LIMg = np.column_stack([M, g]) / L[:, None]         # L^-1 [M, g]
    MtLIMg = Mt @ LIMg

    S = S - MtLIMg[:, :-1]
    h = MtLIMg[:, -1] - h

    lam = np.linalg.solve(S, h)
    press = LIMg[:, -1] + LIMg[:, :-1] @ lam
    flux = BIDf[:, -1] - BIDf[:, :-1] @ lam + _dense(BI @ C) @ press
    return flux, press, lam, S


def schur_complement_symm_fault(BI, C, D, E, f, g, h, regularize: bool = False):
    """Schur-complement solver with a fault-coupling term `E` subtracted from the
    Schur matrix (MRST `schurComplementSymmFault`). Returns `(flux, press, lam,
    S, h)`."""
    BI, C, D = sp.csr_array(BI), sp.csr_array(C), sp.csr_array(D)
    f = np.asarray(f, dtype=float).ravel()
    g = np.asarray(g, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()

    Df = sp.hstack([D, sp.csr_array(f[:, None])]).tocsr()
    BIDf = _dense(BI @ Df)
    CtBIDf = _dense(C.T) @ BIDf
    DtBIDf = _dense(D.T) @ BIDf

    M = CtBIDf[:, :-1]
    S = DtBIDf[:, :-1]
    g = g - CtBIDf[:, -1]
    h = h - DtBIDf[:, -1]

    L = np.asarray((C.T @ BI @ C).diagonal(), dtype=float)
    if regularize:
        L[0] = 2.0 * L[0]
    LIMg = np.column_stack([M, g]) / L[:, None]
    MtLIMg = M.T @ LIMg

    S = S - MtLIMg[:, :-1] - _dense(E)                  # fault coupling
    h = MtLIMg[:, -1] - h

    lam = np.linalg.solve(S, h)
    press = LIMg[:, -1] + LIMg[:, :-1] @ lam
    flux = BIDf[:, -1] - BIDf[:, :-1] @ lam + _dense(BI @ C) @ press
    return flux, press, lam, S, h


def mixed_symm(B, C, D, f, g, h, Do, regularize: bool = False,
               mixed_b: bool = False):
    """Solve the reduced mixed saddle-point system. Returns `(face_flux, press,
    lam_n, A)`."""
    B, C, D, Do = (sp.csr_array(x) for x in (B, C, D, Do))
    f = np.asarray(f, dtype=float).ravel()
    numC, numF, numNF = C.shape[1], Do.shape[1], D.shape[1]

    if not mixed_b:
        Bm = (Do.T @ B @ Do).tocsr()
        fm = _dense(Do.T) @ f
    else:
        Bm, fm = B.tocsr(), f
    Cm = (Do.T @ C).tocsr()
    Dm = (Do.T @ D).tocsr()

    A = sp.bmat([[Bm, Cm, Dm],
                 [Cm.T, None, None],
                 [Dm.T, None, None]], format="lil")
    if regularize:
        A[numF, numF] = 1.0
    rhs = np.r_[fm, np.asarray(g).ravel(), np.asarray(h).ravel()]
    x = spsolve(A.tocsc(), rhs)

    face_flux = x[:numF]
    press = -x[numF:numF + numC]
    lam_n = x[-numNF:]
    return face_flux, press, lam_n, A.tocsr()


def tpf_symm(B, C, D, f, g, h, Do, regularize: bool = False,
             mixed_b: bool = False):
    """Two-point-flux specialisation (assumes `Bm = Do'B Do` diagonal). Returns
    `(face_flux, press, lam_n, A)`."""
    B, C, D, Do = (sp.csr_array(x) for x in (B, C, D, Do))
    f = np.asarray(f, dtype=float).ravel()
    numC, numF, numNF = C.shape[1], Do.shape[1], D.shape[1]

    Bm = (Do.T @ B @ Do) if not mixed_b else B
    CmDm = (Do.T @ sp.hstack([C, D])).tocsr()          # (numF, numC+numNF)
    fm = _dense(Do.T) @ f

    inv_bm = 1.0 / np.asarray(sp.csr_array(Bm).diagonal(), dtype=float)
    CmDmfm = np.column_stack([_dense(CmDm), fm])
    BiCmDmfm = inv_bm[:, None] * CmDmfm                 # Bm^-1 [CmDm, fm]

    A = _dense(CmDm.T) @ BiCmDmfm[:, :-1]
    rhs = np.r_[np.asarray(g).ravel(), np.asarray(h).ravel()] \
        - _dense(CmDm.T) @ BiCmDmfm[:, -1]
    if regularize:
        A[0, 0] = 2.0 * A[0, 0]
    x = np.linalg.solve(A, rhs)

    face_flux = BiCmDmfm @ np.r_[x, 1.0]
    press = x[:numC]
    lam_n = x[-numNF:]
    return face_flux, press, lam_n, A


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.mimetic.compute_mimetic_ip import compute_mimetic_ip
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY

    def _hybrid_system(g, rock, inner_product="ip_simple"):
        """Build (B, BI, C, D, Do) for a grid's mimetic hybrid system."""
        system = compute_mimetic_ip(g, rock, inner_product=inner_product)
        nhf = len(g.cells.faces)
        nc, nf = g.cells.num, g.faces.num
        cellno = np.repeat(np.arange(nc), np.diff(g.cells.facePos))
        faces = np.asarray(g.cells.faces, dtype=np.int64)
        neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
        C = sp.csr_array((np.ones(nhf), (np.arange(nhf), cellno)), shape=(nhf, nc))
        D = sp.csr_array((np.ones(nhf), (np.arange(nhf), faces)), shape=(nhf, nf))
        sgn = 2.0 * (cellno == neigh[faces, 0]) - 1.0
        Do = sp.csr_array((sgn, (np.arange(nhf), faces)), shape=(nhf, nf))
        if system.BI is not None:
            BI = sp.csr_array(system.BI)
            B = sp.csr_array(np.linalg.inv(BI.toarray())) if nhf < 60 else None
        else:
            B = sp.csr_array(system.B)
            blocks = [np.linalg.inv(B[s:e, s:e].toarray())
                      for s, e in zip(g.cells.facePos[:-1], g.cells.facePos[1:])]
            BI = sp.block_diag(blocks, format="csr")
        return B, BI, C, D, Do

    def _dirichlet_problem(g, C, D, Do, a=(3.0, 2.0, 0.0), b=5.0):
        """A well-posed Dirichlet problem: boundary faces fixed to a linear field,
        interior faces free. Returns the reduced (D_int, Do_int, f, g, h) and the
        interior-face mask."""
        fc = np.asarray(g.faces.centroids)
        p_bc = fc[:, 0] * a[0] + fc[:, 1] * a[1] + fc[:, 2] * a[2] + b
        neigh = np.asarray(g.faces.neighbors)
        interior = np.all(neigh >= 0, axis=1)
        Db = D[:, ~interior]
        Di = D[:, interior]
        Doi = Do[:, interior]
        f = -np.asarray(Db @ p_bc[~interior]).ravel()      # known lam -> rhs
        gg = np.zeros(C.shape[1])
        h = np.zeros(int(interior.sum()))
        return Di, Doi, f, gg, h, interior

    def _linear(g, a=(3.0, 2.0, 0.0), b=5.0):
        cc = np.asarray(g.cells.centroids)
        return cc[:, 0] * a[0] + cc[:, 1] * a[1] + cc[:, 2] * a[2] + b

    def test_schur_reproduces_linear_field():
        """ip_simple mimetic is exact for a linear field: with the boundary fixed
        to it, the Schur solve recovers it in the cells."""
        g = compute_geometry(cart_grid([4, 4, 1], [4.0, 4.0, 1.0]))
        rock = make_rock(g, 1.0 * DARCY, 0.2)
        B, BI, C, D, Do = _hybrid_system(g, rock)
        Di, Doi, f, gg, h, _ = _dirichlet_problem(g, C, D, Do)
        _, p, _, _ = schur_complement_symm(BI, C, Di, f, gg, h)
        err = np.max(np.abs(p - _linear(g)))
        assert err < 1e-9, err
        print(f"ok test_schur_reproduces_linear_field (max err {err:.2e})")

    def test_mixed_symm_solves_its_saddle_point_exactly():
        """mixed_symm returns the exact solution of `A x = [f;g;h]` with the right
        sign conventions - checked by reconstructing `x = [flux; -press; lam]` on a
        non-singular synthetic system (SPD B, full-rank constraints)."""
        rng = np.random.default_rng(0)
        numF, numC, numNF = 7, 3, 2
        M = rng.standard_normal((numF, numF))
        B = M @ M.T + numF * np.eye(numF)              # SPD
        C = rng.standard_normal((numF, numC))
        D = rng.standard_normal((numF, numNF))
        Do = sp.eye_array(numF)                        # Bm == B, fm == f
        f = rng.standard_normal(numF)
        gg = rng.standard_normal(numC)
        h = rng.standard_normal(numNF)
        flux, press, lam, A = mixed_symm(B, C, D, f, gg, h, Do)
        x = np.r_[flux, -press, lam]
        assert np.allclose(_dense(A) @ x, np.r_[f, gg, h], atol=1e-8)
        print("ok test_mixed_symm_solves_its_saddle_point_exactly")

    def test_fault_solver_reduces_to_schur_when_E_is_zero():
        g = compute_geometry(cart_grid([3, 2, 1], [3.0, 2.0, 1.0]))
        rock = make_rock(g, 1.0 * DARCY, 0.2)
        B, BI, C, D, Do = _hybrid_system(g, rock)
        Di, Doi, f, gg, h, interior = _dirichlet_problem(g, C, D, Do)
        E = sp.csr_array((int(interior.sum()), int(interior.sum())))
        fl0, p0, lam0, _ = schur_complement_symm(BI, C, Di, f, gg, h)
        flf, pf, lamf, _, _ = schur_complement_symm_fault(BI, C, Di, E, f, gg, h)
        assert np.allclose(p0, pf) and np.allclose(fl0, flf)
        print("ok test_fault_solver_reduces_to_schur_when_E_is_zero")

    def test_tpf_symm_matches_mixed_for_diagonal_bm():
        """When `Bm = Do'B Do` is diagonal, tpf_symm and mixed_symm solve the same
        system and agree exactly on flux and pressure."""
        rng = np.random.default_rng(1)
        numF, numC, numNF = 7, 3, 2
        B = np.diag(rng.uniform(1.0, 3.0, numF))       # diagonal SPD
        C = rng.standard_normal((numF, numC))
        D = rng.standard_normal((numF, numNF))
        Do = sp.eye_array(numF)
        f = rng.standard_normal(numF)
        gg = rng.standard_normal(numC)
        h = rng.standard_normal(numNF)
        ff_t, p_t, lam_t, _ = tpf_symm(B, C, D, f, gg, h, Do)
        ff_m, p_m, lam_m, _ = mixed_symm(B, C, D, f, gg, h, Do)
        assert np.allclose(p_t, p_m, atol=1e-8), np.abs(p_t - p_m).max()
        assert np.allclose(ff_t, ff_m, atol=1e-8)
        # tpfSymm and mixedSymm use opposite multiplier-sign conventions (as in
        # MRST): the physical pressure and flux agree, lam differs by a sign
        assert np.allclose(lam_t, -lam_m, atol=1e-8)
        print("ok test_tpf_symm_matches_mixed_for_diagonal_bm")

    test_schur_reproduces_linear_field()
    test_mixed_symm_solves_its_saddle_point_exactly()
    test_fault_solver_reduces_to_schur_when_E_is_zero()
    test_tpf_symm_matches_mixed_for_diagonal_bm()
    print("\nAll tests passed!")
