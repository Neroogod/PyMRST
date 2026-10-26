"""MRST `GenericAD` static Jacobian helpers (Phase 2).

The chain rule, factored out of the operators. Each takes a *list* of Jacobian
blocks - one per primary variable - and returns a new list.

The operators on `GenericAD` already implement this arithmetic inline, which
gives every test here a free oracle: the helper and the operator must agree.
That is a stronger check than comparing against a hand-written matrix, because
the operator path is the one the simulator actually runs.

One deliberate divergence from MRST is pinned below, in
`test_minus_jac_subtracts_in_the_order_asked_for`.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal import DiagonalAutoDiffBackend
from pymrst.modules.ad_core.backends.diagonal.generic_ad import GenericAD


def _vars(a=(2.0, 3.0), b=(5.0, 7.0)):
    return DiagonalAutoDiffBackend(use_mex=False).init_variables_ad(
        np.array(a), np.array(b))


def _dense(block):
    if hasattr(block, "to_sparse"):
        block = block.to_sparse()
    return np.asarray(block.toarray() if hasattr(block, "toarray")
                      else block.todense())


# -- against the operators, which are the oracle ---------------------------

def test_the_product_rule_matches_the_multiply_operator():
    """`d(uv) = u dv + v du`, checked against the path the simulator runs."""
    x, y = _vars()
    helper = GenericAD.times_jac(x.val, y.val, x.jac, y.jac)
    np.testing.assert_allclose(_dense(helper[0]), _dense((x*y).jac[0]))

    # and against the closed form: d/dx = y, d/dy = x
    d = _dense(helper[0])
    np.testing.assert_allclose(d.diagonal()[:2], y.val)
    np.testing.assert_allclose([d[0, 2], d[1, 3]], x.val)


def test_addition_matches_the_add_operator():
    x, y = _vars()
    np.testing.assert_allclose(_dense(GenericAD.plus_jac(x.jac, y.jac)[0]),
                               _dense((x + y).jac[0]))


def test_subtraction_matches_the_subtract_operator():
    x, y = _vars()
    np.testing.assert_allclose(_dense(GenericAD.minus_jac(x.jac, y.jac)[0]),
                               _dense((x - y).jac[0]))


# -- the divergence, pinned -------------------------------------------------

def test_minus_jac_subtracts_in_the_order_asked_for():
    """A deliberate divergence from MRST, recorded rather than hidden.

    MRST's `minusJac` handles the `nv2 == 1` case by recursing as
    `minusJac(J2, J1)`, which computes `J2 - J1` - the negation of what was
    asked. The `nv1 == 1` branch immediately above does not have that shape, so
    MRST's two scalar cases disagree in sign with each other.

    This implements the subtraction that was requested in both directions. The
    test asserts antisymmetry, which is what fails if the MRST form is
    restored. Not yet checked against MATLAB; that is how to settle it.
    """
    x, _ = _vars()
    scalar = GenericAD(np.array([1.0]),
                       [sp.csr_array(np.ones((1, x.jac[0].shape[1])))])

    forward = GenericAD.minus_jac(x.jac, scalar.jac)
    backward = GenericAD.minus_jac(scalar.jac, x.jac)
    np.testing.assert_allclose(_dense(forward[0]), -_dense(backward[0]),
                               err_msg="a - b must be -(b - a)")


# -- broadcasting -----------------------------------------------------------

def test_a_one_row_operand_broadcasts_to_the_other():
    x, _ = _vars()
    width = x.jac[0].shape[1]
    scalar = [sp.csr_array(np.ones((1, width)))]

    result = GenericAD.plus_jac(scalar, x.jac)
    assert result[0].shape == (2, width)
    np.testing.assert_allclose(_dense(result[0]),
                               np.ones((2, width)) + _dense(x.jac[0]))


def test_broadcasting_works_from_either_side():
    x, _ = _vars()
    width = x.jac[0].shape[1]
    scalar = [sp.csr_array(np.ones((1, width)))]
    np.testing.assert_allclose(_dense(GenericAD.plus_jac(scalar, x.jac)[0]),
                               _dense(GenericAD.plus_jac(x.jac, scalar)[0]))


def test_two_incompatible_shapes_are_refused():
    """Neither operand is scalar, so there is no broadcast that makes sense."""
    a = [sp.csr_array(np.ones((2, 4)))]
    b = [sp.csr_array(np.ones((3, 4)))]
    with pytest.raises(ValueError, match="one must be"):
        GenericAD.plus_jac(a, b)


# -- the zero-term shortcut -------------------------------------------------

def test_a_zero_value_drops_its_term_from_the_product_rule():
    """Not an optimisation detail: forming `0 * J` on a large sparse block
    costs more than testing for it, which is why MRST branches."""
    x, y = _vars()
    zero = np.zeros_like(x.val)

    only_second = GenericAD.times_jac(zero, y.val, x.jac, y.jac)
    np.testing.assert_allclose(_dense(only_second[0]),
                               _dense(GenericAD.times_jac(
                                   zero, y.val, x.jac, y.jac)[0]))
    # with v1 == 0 the surviving term is v2 * dJ1
    from pymrst.modules.ad_core.backends.diagonal.utils import diag_mult
    np.testing.assert_allclose(
        _dense(only_second[0]), _dense(diag_mult(y.val, x.jac[0].to_sparse())))


# -- structure ---------------------------------------------------------------

def test_a_linear_operator_multiplies_every_block():
    x, _ = _vars()
    operator = sp.csr_array(np.array([[1.0, 0.0], [0.0, 2.0]]))
    result = GenericAD.mtimes_jac(operator, x.jac)
    np.testing.assert_allclose(_dense(result[0]),
                               operator @ _dense(x.jac[0]))


def test_num_vars_reports_each_block_s_width():
    x, _ = _vars()
    assert x.get_num_vars() == [x.jac[0].shape[1]]


def test_casting_returns_blocks_because_the_container_normalises_them():
    """`GenericAD.__init__` converts whatever it is handed back into
    `DiagonalJacobian`, so unlike MRST this cannot return an AD object holding
    sparse blocks. The blocks are what a caller wants anyway."""
    x, _ = _vars()
    blocks = x.cast_jac_to_sparse()
    assert sp.issparse(blocks[0])
    np.testing.assert_allclose(np.asarray(blocks[0].toarray()),
                               _dense(x.jac[0]))


def test_convert_double_keeps_the_variable_space_and_drops_the_derivatives():
    """A constant in the same variable space - which is what makes it
    combinable with the original without losing its derivatives."""
    x, _ = _vars()
    constant = x.convert_double(np.array([9.0, 9.0]))

    np.testing.assert_allclose(constant.val, [9.0, 9.0])
    assert not np.any(_dense(constant.jac[0])), "a constant has no derivatives"
    assert constant.jac[0].shape == x.jac[0].shape

    total = x + constant
    np.testing.assert_allclose(_dense(total.jac[0]), _dense(x.jac[0]))
