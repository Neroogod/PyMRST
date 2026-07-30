import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    critical_point_chop,
    get_cell_major_reordering,
    split_matrix_for_reduction,
)


def test_critical_point_chop_modes_and_directions():
    x0 = np.array([-2.0, -0.25, 2.0, 0.25, -2.0])
    trial = np.array([2.0, 2.0, -2.0, -2.0, 2.0])
    both = critical_point_chop(x0, trial, 0.0, 1.0, "both")
    assert np.allclose(both, [-0.5, 1.5, 0.5, -1.5, -0.5])
    above = critical_point_chop(x0, trial, 0.0, 1.0, "above", "increasing")
    assert np.allclose(above, [1.0, 1.0, -2.0, -2.0, 1.0])


def test_split_matrix_extracts_schur_blocks_and_factors_e():
    A = sp.csr_matrix(np.arange(1.0, 26.0).reshape(5, 5) + np.eye(5) * 20)
    b = np.arange(10.0).reshape(5, 2)
    system = split_matrix_for_reduction(A, b, 2, "sparse")
    assert np.array_equal(system.B.toarray(), A[:2, :2].toarray())
    assert np.array_equal(system.C.toarray(), A[:2, 2:].toarray())
    assert np.array_equal(system.D.toarray(), A[2:, :2].toarray())
    assert np.array_equal(system.E.toarray(), A[2:, 2:].toarray())
    assert np.array_equal(system.f, b[:2])
    assert np.array_equal(system.h, b[2:])
    assert system.factor is not None


def test_cell_major_reordering_with_extra_dofs_and_custom_orders():
    assert np.array_equal(
        get_cell_major_reordering(3, 2), [0, 3, 1, 4, 2, 5]
    )
    order = get_cell_major_reordering(
        3, 2, ndof=8, equation_ordering=[1, 0], cell_ordering=[2, 0, 1]
    )
    assert np.array_equal(order, [5, 2, 3, 0, 4, 1, 6, 7])
