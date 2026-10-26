"""The AD backend actually dispatches now (Phase 2).

The original audit of `ad_core` called this class a facade, and it was: it
recorded *which* backend had been asked for and supplied none. MRST's
`AutoDiffBackend` defines what an AD variable **is** - `initVariablesAD` and
`convertToAD` - and neither existed here, so consumers picked an
implementation with `isinstance` checks at the call site and the backend object
was a label with a `.available` flag.

These tests are about the thing that makes a backend a backend: ask two
different backends for AD variables and they must produce *different
representations of the same derivatives*. A label test would pass on the
facade; this cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.modules.ad_core.backends.auto_diff_backend import AutoDiffBackend
from pymrst.modules.ad_core.backends.diagonal import DiagonalAutoDiffBackend
from pymrst.modules.ad_core.backends.sparse.sparse_auto_diff_backend import (
    SparseAutoDiffBackend,
)


def _jacobian(variable):
    """The full Jacobian, however the backend chooses to store it.

    The two backends genuinely differ here, and comparing `jac[0]` to `jac[0]`
    would be comparing different things: sparse ADI keeps one block per primary
    variable (n-by-n each), while the diagonal backend keeps a single combined
    block (n-by-n*nvar). Stacking the blocks horizontally gives the same
    matrix from both, which is the whole point - the representation differs and
    the derivatives do not.
    """
    def dense(block):
        if hasattr(block, "todense"):
            return np.asarray(block.todense())
        return np.asarray(block.to_sparse().todense())

    return np.hstack([dense(b) for b in variable.jac])


# -- the interface exists at all -------------------------------------------

def test_a_backend_seeds_ad_variables():
    """The method whose absence made this class a label."""
    x, y = SparseAutoDiffBackend().init_variables_ad(np.array([1.0, 2.0]),
                                                      np.array([3.0, 4.0]))
    np.testing.assert_allclose(np.asarray(x.val if hasattr(x, "val") else x),
                               [1.0, 2.0])
    assert len(x.jac) == 2, "one Jacobian block per primary variable"


def test_seeding_gives_the_identity_derivative():
    """dx/dx = I. If this is wrong every derivative downstream is wrong by the
    same factor, which is a uniquely hard thing to notice."""
    (x,) = SparseAutoDiffBackend().init_variables_ad(np.array([5.0, 6.0, 7.0]))
    np.testing.assert_allclose(_jacobian(x), np.eye(3))


def test_the_two_backends_really_do_store_derivatives_differently():
    """Guards the comparison above from becoming vacuous.

    If both backends ever produced the same block layout, the agreement tests
    would still pass while proving nothing about dispatch.
    """
    values = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    sparse = SparseAutoDiffBackend().init_variables_ad(*values)[0]
    diagonal = DiagonalAutoDiffBackend(use_mex=False).init_variables_ad(*values)[0]
    assert len(sparse.jac) != len(diagonal.jac), (
        "sparse keeps one block per variable, diagonal keeps one combined "
        "block; if that stops being true this comparison is vacuous")


# -- the claim worth making -------------------------------------------------

def test_the_backends_disagree_on_representation_and_agree_on_derivatives():
    """Two backends, one answer.

    This is what a backend abstraction is *for*, and it is exactly what a
    label-only class cannot deliver. The representations differ - sparse ADI
    against diagonal `GenericAD` - and the derivatives must not.
    """
    values = (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))

    sparse_x, sparse_y = SparseAutoDiffBackend().init_variables_ad(*values)
    diag_x, diag_y = DiagonalAutoDiffBackend(
        use_mex=False).init_variables_ad(*values)

    assert type(sparse_x).__name__ != type(diag_x).__name__, (
        "the two backends must produce different representations, or this "
        "test proves nothing")

    for sparse, diagonal in ((sparse_x, diag_x), (sparse_y, diag_y)):
        np.testing.assert_allclose(_jacobian(sparse), _jacobian(diagonal))


def test_both_diagonal_layouts_give_the_same_derivatives():
    """Row-major is a storage order, not a different answer.

    Agreement alone is not enough to prove the flag is honoured - swapping the
    two builders keeps them agreeing, and an earlier version of this test
    passed against exactly that mutation. So the *representation* is checked
    too: the flag must select a different Jacobian class.
    """
    values = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    column = DiagonalAutoDiffBackend(use_mex=False).init_variables_ad(*values)
    row = DiagonalAutoDiffBackend(use_mex=False,
                                   row_major=True).init_variables_ad(*values)

    assert type(column[0].jac[0]).__name__ == "DiagonalJacobian"
    assert type(row[0].jac[0]).__name__ == "RowMajorDiagonalJacobian"

    for a, b in zip(column, row):
        np.testing.assert_allclose(_jacobian(a), _jacobian(b))


def test_an_expression_differentiates_the_same_under_either_backend():
    """The end-to-end claim: same arithmetic, same derivative."""
    def expression(x, y):
        return x*y + x*x

    values = (np.array([2.0, 3.0]), np.array([5.0, 7.0]))
    sx, sy = SparseAutoDiffBackend().init_variables_ad(*values)
    dx, dy = DiagonalAutoDiffBackend(use_mex=False).init_variables_ad(*values)

    np.testing.assert_allclose(_jacobian(expression(sx, sy)),
                               _jacobian(expression(dx, dy)))
    # d/dx (xy + x^2) = y + 2x, on the diagonal
    np.testing.assert_allclose(np.diag(_jacobian(expression(sx, sy))),
                               values[1] + 2*values[0])


# -- converting a constant --------------------------------------------------

def test_a_constant_lifted_to_ad_carries_zero_derivatives():
    """Without `convertToAD` a constant meeting an AD expression has no
    derivative blocks, and the result silently loses the derivatives of
    whatever it was combined with."""
    backend = SparseAutoDiffBackend()
    (x,) = backend.init_variables_ad(np.array([1.0, 2.0]))
    lifted = backend.convert_to_ad(np.array([7.0, 8.0]), x)

    np.testing.assert_allclose(np.asarray(lifted.val), [7.0, 8.0])
    np.testing.assert_allclose(_jacobian(lifted), np.zeros((2, 2)))


def test_a_lifted_constant_combines_without_losing_derivatives():
    backend = SparseAutoDiffBackend()
    (x,) = backend.init_variables_ad(np.array([3.0, 4.0]))
    total = x + backend.convert_to_ad(np.array([1.0, 1.0]), x)
    np.testing.assert_allclose(_jacobian(total), np.eye(2))


# -- descriptions -----------------------------------------------------------

@pytest.mark.parametrize("backend,expected", [
    (AutoDiffBackend(), "Standard ADI (sparse jacobian)"),
    (SparseAutoDiffBackend(), "Sparse"),
    (SparseAutoDiffBackend(use_blocks=False), "Sparse (single matrix)"),
    (DiagonalAutoDiffBackend(use_mex=False), "Diagonal"),
    (DiagonalAutoDiffBackend(use_mex=False, row_major=True),
     "Diagonal-RowMajor"),
])
def test_each_backend_describes_itself(backend, expected):
    assert backend.get_backend_description() == expected


def test_updating_discrete_operators_is_a_no_op_that_returns_the_model():
    """PyMRST's operators are backend-agnostic. The hook exists so a caller
    written against MRST does not have to special-case its absence."""
    model = object()
    for backend in (AutoDiffBackend(), SparseAutoDiffBackend(),
                    DiagonalAutoDiffBackend(use_mex=False)):
        assert backend.update_discrete_operators(model) is model


def test_the_mrst_spellings_resolve():
    for cls in (AutoDiffBackend, SparseAutoDiffBackend,
                DiagonalAutoDiffBackend):
        assert cls.initVariablesAD is cls.init_variables_ad
        assert cls.convertToAD is cls.convert_to_ad
        assert cls.getBackendDescription is cls.get_backend_description
