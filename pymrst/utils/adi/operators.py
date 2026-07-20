"""ADI operators and equation assembly helpers."""

import numpy as np
import scipy.sparse as sp

from pymrst.utils.adi.adi import ADI, _diag_mul
from pymrst.utils.adi.value import value


def _is_generic_ad(x) -> bool:
    return x.__class__.__name__ == "GenericAD"


def ad_matmul(mat, x):
    """Return matrix product `mat @ x` for ADI or plain arrays."""
    if _is_generic_ad(x):
        from pymrst.modules.ad_core.backends.diagonal.generic_ad import diagonal_ad_matmul

        return diagonal_ad_matmul(mat, x)
    if isinstance(x, ADI):
        return ADI(mat @ x.val, [(mat @ j).tocsr() for j in x.jac])
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
        jac = []
        for k, size in enumerate(sizes):
            jac.append(block_cls.from_sparse(
                sp.vstack([part.jac[k].to_sparse() for part in full], format="csr")
            ))
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

    test_matmul_div_sum()
    test_combine_equations()
    test_max_min()
    print("\nAll tests passed!")
