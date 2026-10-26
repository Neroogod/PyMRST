"""ADI operators and equation assembly helpers."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import _sparsetools

from pymrst.utils.adi.adi import ADI, _diag_mul, _fast_csr
from pymrst.utils.adi.value import value


def _is_generic_ad(x) -> bool:
    return x.__class__.__name__ == "GenericAD"


def _fast_matmul(mat: sp.csr_array, jac: sp.csr_array) -> sp.csr_array:
    """`mat @ jac` for two CSR matrices, skipping scipy's redundant
    constructor validation on the result.

    Mirrors `scipy.sparse._compressed._cs_matrix._matmul_sparse` exactly -
    same `csr_matmat_maxnnz`/`csr_matmat` C routines scipy's own `@`
    dispatches to, computing the identical `data`/`indices`/`indptr` - but
    wraps the (already scipy-computed, already-canonical) result via
    `_fast_csr` instead of the validating constructor `mat @ jac` would
    otherwise go through. `mat` (a fixed discretization operator) and
    `jac` (a Jacobian block) are both plain float64/int64 CSR here, so this
    skips the general-dtype-upgrade bookkeeping `_matmul_sparse` also does
    for inputs that might not match - not applicable to this module's own
    invariants. Falls back to the ordinary `@` if scipy's internal API this
    relies on is ever unavailable, rather than failing hard.

    `csr_matmat_maxnnz` is an *upper bound* on the result's nonzero count,
    not the exact one - when a row of `mat` has two entries that route
    through `jac` into the same output column, `csr_matmat` sums them into
    one stored entry, so `indptr[-1]` (the true count) can come out smaller
    than `nnz` (the allocated size). Scipy's own validated constructor path
    trims `data`/`indices` down via `prune()`, a step `_fast_csr` exists
    specifically to skip; skipping it *without* trimming first left stale,
    over-allocated tail entries in `data`/`indices` that a later op read as
    real nonzeros. `data`/`indices` are sliced to `indptr[-1]` below before
    handoff to `_fast_csr`, which is the one attribute of "already-valid
    CSR" `csr_matmat`'s raw output does not satisfy on its own.
    """
    try:
        M, N = mat.shape[0], jac.shape[1]
        mat_indptr = mat.indptr.astype(np.int64, copy=False)
        mat_indices = mat.indices.astype(np.int64, copy=False)
        jac_indptr = jac.indptr.astype(np.int64, copy=False)
        jac_indices = jac.indices.astype(np.int64, copy=False)
        max_nnz = _sparsetools.csr_matmat_maxnnz(
            M, N, mat_indptr, mat_indices, jac_indptr, jac_indices)
        indptr = np.empty(M + 1, dtype=np.int64)
        indices = np.empty(max_nnz, dtype=np.int64)
        data = np.empty(max_nnz, dtype=np.float64)
        _sparsetools.csr_matmat(
            M, N, mat_indptr, mat_indices, mat.data,
            jac_indptr, jac_indices, jac.data, indptr, indices, data)
        true_nnz = int(indptr[-1])
        return _fast_csr(data[:true_nnz], indices[:true_nnz], indptr, (M, N))
    except (AttributeError, TypeError):
        return (mat @ jac).tocsr()


def ad_matmul(mat, x):
    """Return matrix product `mat @ x` for ADI or plain arrays."""
    if _is_generic_ad(x):
        from pymrst.modules.ad_core.backends.diagonal.generic_ad import diagonal_ad_matmul

        return diagonal_ad_matmul(mat, x)
    if isinstance(x, ADI):
        mat_csr = mat.tocsr() if sp.issparse(mat) else sp.csr_array(mat)
        return ADI(mat @ x.val, [_fast_matmul(mat_csr, j) for j in x.jac])
    return mat @ x


def ad_exp(x):
    if _is_generic_ad(x):
        from pymrst.modules.ad_core.backends.diagonal.generic_ad import GenericAD

        exp_val = np.exp(x.val)
        return GenericAD(exp_val, [j.left_scale(exp_val) for j in x.jac])
    if isinstance(x, ADI):
        exp_val = np.exp(x.val)
        return ADI(exp_val, [_diag_mul(exp_val, j) for j in x.jac])
    return np.exp(x)


def ad_log(x):
    if _is_generic_ad(x):
        from pymrst.modules.ad_core.backends.diagonal.generic_ad import GenericAD

        return GenericAD(np.log(x.val), [j.left_scale(1.0 / x.val) for j in x.jac])
    if isinstance(x, ADI):
        return ADI(np.log(x.val), [_diag_mul(1.0 / x.val, j) for j in x.jac])
    return np.log(x)


def ad_abs(x):
    if _is_generic_ad(x):
        from pymrst.modules.ad_core.backends.diagonal.generic_ad import GenericAD

        return GenericAD(np.abs(x.val), [j.left_scale(np.sign(x.val)) for j in x.jac])
    if isinstance(x, ADI):
        return ADI(np.abs(x.val), [_diag_mul(np.sign(x.val), j) for j in x.jac])
    return np.abs(x)


def ad_max(u, v):
    """Return element-wise maximum, differentiating through the winning branch."""
    if _is_generic_ad(u) or _is_generic_ad(v):
        from pymrst.modules.ad_core.backends.diagonal.generic_ad import GenericAD

        if not _is_generic_ad(u):
            values = np.broadcast_to(np.atleast_1d(u).astype(float), v.val.shape)
            pick_v = ~(values > v.val)
            return GenericAD(np.maximum(values, v.val),
                             [j.left_scale(pick_v.astype(float)) for j in v.jac])
        if not _is_generic_ad(v):
            return ad_max(v, u)
        aligned_u, aligned_v = GenericAD._align(u, v)
        pick_u = aligned_u.val > aligned_v.val
        jac = [
            a.left_scale(pick_u.astype(float)) + b.left_scale((~pick_u).astype(float))
            for a, b in zip(aligned_u.jac, aligned_v.jac)
        ]
        return GenericAD(np.maximum(aligned_u.val, aligned_v.val), jac)
    if not isinstance(u, ADI) and not isinstance(v, ADI):
        return np.maximum(np.asarray(u), np.asarray(v))
    if not isinstance(u, ADI) and isinstance(v, ADI):
        values = np.broadcast_to(np.atleast_1d(u).astype(float), v.val.shape)
        pick_v = ~(values > v.val)
        return ADI(np.maximum(values, v.val), [_diag_mul(pick_v.astype(float), j) for j in v.jac])
    if isinstance(u, ADI) and not isinstance(v, ADI):
        return ad_max(v, u)
    aligned_u, aligned_v = ADI._align(u, v)
    pick_u = aligned_u.val > aligned_v.val
    jac = [
        _diag_mul(pick_u.astype(float), a) + _diag_mul((~pick_u).astype(float), b)
        for a, b in zip(aligned_u.jac, aligned_v.jac)
    ]
    return ADI(np.maximum(aligned_u.val, aligned_v.val), jac)


def ad_min(u, v):
    """Return element-wise minimum, differentiating through the winning branch."""
    neg_u = -u if isinstance(u, ADI) or _is_generic_ad(u) else -np.asarray(u)
    neg_v = -v if isinstance(v, ADI) or _is_generic_ad(v) else -np.asarray(v)
    return -ad_max(neg_u, neg_v)


def ad_sum(x):
    """Sum to a scalar ADI while preserving Jacobian blocks."""
    if _is_generic_ad(x):
        from pymrst.modules.ad_core.backends.diagonal.generic_ad import diagonal_sum

        return diagonal_sum(x)
    if isinstance(x, ADI):
        ones = sp.csr_array(np.ones((1, len(x))))
        return ADI(np.array([x.val.sum()]), [ones @ j for j in x.jac])
    return np.sum(x)


def vertcat_adi(*parts):
    """Stack ADIs vertically, keeping their block structure."""
    if any(_is_generic_ad(part) for part in parts):
        from pymrst.modules.ad_core.backends.diagonal.diagonal_jacobian import DiagonalJacobian
        from pymrst.modules.ad_core.backends.diagonal.generic_ad import GenericAD
        from pymrst.modules.ad_core.backends.diagonal.row_major_diagonal_jacobian import (
            RowMajorDiagonalJacobian,
        )

        ads = [part for part in parts if _is_generic_ad(part)]
        sizes = [j.shape[1] for j in ads[0].jac]
        block_cls = RowMajorDiagonalJacobian if isinstance(ads[0].jac[0], RowMajorDiagonalJacobian) else DiagonalJacobian
        full = []
        for part in parts:
            if _is_generic_ad(part):
                full.append(part)
            else:
                val = np.atleast_1d(np.asarray(part, dtype=np.float64)).ravel()
                full.append(GenericAD(val, [block_cls.zeros(val.size, size) for size in sizes]))
        val = np.concatenate([part.val for part in full])
        # `block_cls.vertcat` stacks in its own native representation - a
        # triplet/row-major append with a row-index shift, not a promote to
        # scipy sparse and back. That round-trip was the whole reason
        # DiagonalJacobian's `vertcat`/`cat` read as unported: the capability
        # was never missing, just not used on the path that needed it, which
        # is every equation-block assembly in the black-oil and compositional
        # models (combine_equations goes through here).
        jac = [block_cls.vertcat([part.jac[k] for part in full])
              for k in range(len(sizes))]
        return GenericAD(val, jac)
    ads = [part for part in parts if isinstance(part, ADI)]
    if not ads:
        return np.concatenate(
            [np.atleast_1d(np.asarray(part, dtype=np.float64)).ravel() for part in parts]
        )
    sizes = [j.shape[1] for j in ads[0].jac]
    full = [_as_adi(part, sizes) for part in parts]
    val = np.concatenate([part.val for part in full])
    jac = [sp.vstack([part.jac[k] for part in full], format="csr") for k in range(len(sizes))]
    return ADI(val, jac)


def combine_equations(*eqs) -> ADI:
    """Stack equations and assemble one full Jacobian block."""
    if any(_is_generic_ad(eq) for eq in eqs):
        from pymrst.modules.ad_core.backends.diagonal.generic_ad import combine_diagonal_equations

        return combine_diagonal_equations(*eqs)
    stacked = vertcat_adi(*eqs)
    return ADI(stacked.val, [sp.hstack(stacked.jac, format="csr")])


def _as_adi(part, sizes: list[int]) -> ADI:
    if isinstance(part, ADI):
        return part
    val = np.atleast_1d(np.asarray(part, dtype=np.float64)).ravel()
    return ADI(val, [sp.csr_array((val.size, size)) for size in sizes])


if __name__ == "__main__":
    from pymrst.utils.adi.init_variables_adi import init_variables_adi

    def test_matmul_div_sum() -> None:
        (x,) = init_variables_adi(np.array([1.0, 2.0, 4.0]))
        mat = sp.csr_array(np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]))
        y = ad_matmul(mat, x**2)
        assert np.allclose(y.val, [1.0 - 4.0, 4.0 - 16.0])
        assert np.allclose(y.jac[0].toarray(), [[2.0, -4.0, 0.0], [0.0, 4.0, -8.0]])
        summed = ad_sum(1.0 / x)
        assert np.isclose(summed.val[0], 1.75)
        print("ok test_matmul_div_sum")

    def test_combine_equations() -> None:
        p, b = init_variables_adi(np.array([1.0, 2.0]), 5.0)
        eq = combine_equations(p * 3.0 + b, b * 2.0)
        assert len(eq.jac) == 1
        assert eq.jac[0].shape == (3, 3)
        assert np.allclose(eq.jac[0].toarray(), [[3, 0, 1], [0, 3, 1], [0, 0, 2]])
        assert np.allclose(eq.val, [8.0, 11.0, 10.0])
        print("ok test_combine_equations")

    def test_max_min() -> None:
        (x,) = init_variables_adi(np.array([-2.0, 3.0]))
        maxed = ad_max(0.0, x)
        assert np.allclose(maxed.val, [0.0, 3.0])
        assert np.allclose(maxed.jac[0].toarray(), [[0, 0], [0, 1]])
        mined = ad_min(x, 1.0)
        assert np.allclose(mined.val, [-2.0, 1.0])
        assert np.allclose(mined.jac[0].toarray(), [[1, 0], [0, 0]])
        print("ok test_max_min")

    def test_fast_matmul_matches_scipy_reference() -> None:
        """`_fast_matmul` reimplements `mat @ jac`'s C-level computation
        directly (mirroring scipy's own `_matmul_sparse`) rather than
        calling scipy's `@` - so this checks its output against plain
        `mat @ jac` across several random, differently-shaped matrix
        pairs, not just the one small case `test_matmul_div_sum` already
        exercises through `ad_matmul`."""
        rng = np.random.default_rng(4)
        worst = 0.0
        for _ in range(15):
            m, k, n = (int(rng.integers(1, 25)) for _ in range(3))
            mat = sp.random_array((m, k), density=0.4, random_state=rng, format="csr")
            jac = sp.random_array((k, n), density=0.4, random_state=rng, format="csr")
            got = _fast_matmul(mat, jac)
            ref = (mat @ jac).tocsr()
            assert got.shape == ref.shape, (got.shape, ref.shape)
            assert got.indptr[-1] == len(got.data) == len(got.indices), (
                "got.data/indices must be trimmed to the true nnz, not left at "
                "csr_matmat_maxnnz's upper-bound allocation size")
            worst = max(worst, float(np.max(np.abs(got.toarray() - ref.toarray()))))
        assert worst < 1e-12, f"fast matmul should match scipy exactly, worst diff {worst:.2e}"
        print(f"ok test_fast_matmul_matches_scipy_reference (15 random cases, worst diff {worst:.2e})")

    def test_fast_matmul_trims_when_maxnnz_overestimates() -> None:
        """Regression test for a real bug found wiring this into
        `three_phase_black_oil_model.py`: `csr_matmat_maxnnz`'s symbolic
        pass counts a `(row, col)` output position once it sees *any* path
        into it, before any arithmetic happens - it cannot know that two
        opposite-signed contributions to that same position will cancel to
        *exactly* zero. `csr_matmat`'s numeric pass then correctly drops
        that exact-zero entry, so the true nnz (`indptr[-1]`) comes out
        smaller than `csr_matmat_maxnnz`'s allocated size. Scipy's own
        validated constructor path trims `data`/`indices` down via
        `prune()` - the exact step `_fast_csr` skips - so `_fast_matmul`
        must trim them itself first. Before that fix, the stale,
        uninitialized tail of the over-allocated `data`/`indices` arrays
        was read downstream as real nonzero entries:
        `three_phase_black_oil_model.py`'s SPE1 self-test crashed in
        `_diag_mul` with a shape mismatch between `jac.data` and
        `np.diff(jac.indptr)`, confirmed (by reverting this trim and
        rerunning) to reproduce that exact failure.

        This constructs the minimal case directly: `mat`'s one row has two
        entries of opposite sign, both landing (through `jac`) on output
        column 0, cancelling exactly - the precise pattern that makes
        `csr_matmat_maxnnz` overcount.
        """
        mat = sp.csr_array(np.array([[1.0, -1.0], [0.0, 0.0]]))
        jac = sp.csr_array(np.array([[1.0, 0.0], [1.0, 0.0]]))
        got = _fast_matmul(mat, jac)
        ref = (mat @ jac).tocsr()
        assert got.indptr[-1] == len(got.data), (
            f"true nnz {got.indptr[-1]} must match allocated data length {len(got.data)}")
        assert np.allclose(got.toarray(), ref.toarray())
        assert np.allclose(got.toarray(), [[0.0, 0.0], [0.0, 0.0]])
        print("ok test_fast_matmul_trims_when_maxnnz_overestimates")

    def test_ad_matmul_end_to_end_through_the_fast_path() -> None:
        """`ad_matmul` itself, through `_fast_matmul`, against a
        finite-difference reference - not just `_fast_matmul` in
        isolation."""
        x0 = np.array([1.0, 2.0, 4.0])
        mat = sp.csr_array(np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [2.0, 0.0, 3.0]]))

        def f(x):
            return mat @ (x ** 2)

        (x,) = init_variables_adi(x0)
        y = ad_matmul(mat, x ** 2)
        h = 1e-6
        numeric = np.zeros((3, 3))
        for j in range(3):
            step = np.zeros(3)
            step[j] = h
            numeric[:, j] = (f(x0 + step) - f(x0 - step)) / (2 * h)
        assert np.allclose(y.jac[0].toarray(), numeric, atol=1e-5)
        print("ok test_ad_matmul_end_to_end_through_the_fast_path")

    test_matmul_div_sum()
    test_combine_equations()
    test_max_min()
    test_fast_matmul_matches_scipy_reference()
    test_fast_matmul_trims_when_maxnnz_overestimates()
    test_ad_matmul_end_to_end_through_the_fast_path()
    print("\nAll tests passed!")
