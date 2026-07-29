"""MRST-like ``GenericAD`` variable for the diagonal backend."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal.diagonal_jacobian import DiagonalJacobian
from pymrst.modules.ad_core.backends.diagonal.row_major_diagonal_jacobian import (
    RowMajorDiagonalJacobian,
)
from pymrst.modules.ad_core.backends.diagonal.kernels import HAS_NUMBA_KERNELS, vstack_row_major, vstack_triplet
from pymrst.utils.adi.adi import ADI
from pymrst.utils.adi.value import value

Numeric = Union[float, int, np.ndarray]


class GenericAD:
    """Value vector plus diagonal-backend Jacobian blocks.

    This intentionally mirrors the sparse ``ADI`` interface used elsewhere in
    PyMRST: ``val`` stores primary values and ``jac`` is a block list. The block
    objects stay compact until ``to_adi`` or ``to_sparse`` is requested.
    """

    __slots__ = ("val", "jac")
    __array_ufunc__ = None

    def __init__(self, val: np.ndarray, jac: Sequence):
        self.val = np.atleast_1d(np.asarray(val, dtype=np.float64)).ravel()
        self.jac = [
            block if _is_diagonal_block(block) else DiagonalJacobian.from_sparse(block)
            for block in jac
        ]

    def __len__(self) -> int:
        return self.val.size

    def __repr__(self) -> str:
        cols = [j.shape[1] for j in self.jac]
        return f"GenericAD(n={len(self)}, blocks={cols})"

    def copy(self) -> "GenericAD":
        return GenericAD(self.val.copy(), [j.copy() for j in self.jac])

    def to_adi(self) -> ADI:
        return ADI(self.val.copy(), [j.to_sparse() for j in self.jac])

    def _expand(self, n: int) -> "GenericAD":
        if len(self) != 1:
            raise ValueError("only scalar GenericAD can broadcast")
        return GenericAD(np.full(n, self.val[0]), [j.row_repeat(n) for j in self.jac])

    @staticmethod
    def _align(u: "GenericAD", v: "GenericAD") -> Tuple["GenericAD", "GenericAD"]:
        if len(u) == len(v):
            return u, v
        if len(u) == 1:
            return u._expand(len(v)), v
        if len(v) == 1:
            return u, v._expand(len(u))
        raise ValueError(f"incompatible GenericAD lengths {len(u)}, {len(v)}")

    def __add__(self, other) -> "GenericAD":
        if isinstance(other, ADI) and not isinstance(other, GenericAD):
            return self.to_adi() + other
        if isinstance(other, GenericAD):
            u, v = self._align(self, other)
            return GenericAD(u.val + v.val, [a + b for a, b in zip(u.jac, v.jac)])
        values = np.atleast_1d(np.asarray(other, dtype=np.float64)).ravel()
        u = self if values.size == 1 or len(self) > 1 else self._expand(values.size)
        return GenericAD(u.val + values, [j.copy() for j in u.jac])

    __radd__ = __add__

    def __neg__(self) -> "GenericAD":
        return GenericAD(-self.val, [-j for j in self.jac])

    def __sub__(self, other) -> "GenericAD":
        return self + (-other if isinstance(other, GenericAD) else -np.asarray(other))

    def __rsub__(self, other) -> "GenericAD":
        return (-self) + other

    def __mul__(self, other) -> "GenericAD":
        if isinstance(other, ADI) and not isinstance(other, GenericAD):
            return self.to_adi() * other
        if isinstance(other, GenericAD):
            u, v = self._align(self, other)
            jac = [
                a.left_scale(v.val) + b.left_scale(u.val)
                for a, b in zip(u.jac, v.jac)
            ]
            return GenericAD(u.val * v.val, jac)
        values = np.atleast_1d(np.asarray(other, dtype=np.float64)).ravel()
        if values.size == 1:
            return GenericAD(self.val * values[0], [j * values[0] for j in self.jac])
        u = self if len(self) > 1 else self._expand(values.size)
        return GenericAD(u.val * values, [j.left_scale(values) for j in u.jac])

    __rmul__ = __mul__

    def __truediv__(self, other) -> "GenericAD":
        if isinstance(other, GenericAD):
            return self * other ** -1.0
        return self * (1.0 / np.asarray(other, dtype=np.float64))

    def __rtruediv__(self, other) -> "GenericAD":
        return self ** -1.0 * other

    def __pow__(self, k) -> "GenericAD":
        power = float(k)
        derivative = power * self.val ** (power - 1.0)
        return GenericAD(self.val**power, [j.left_scale(derivative) for j in self.jac])

    def __lt__(self, other):
        return self.val < value(other)

    def __le__(self, other):
        return self.val <= value(other)

    def __gt__(self, other):
        return self.val > value(other)

    def __ge__(self, other):
        return self.val >= value(other)

    def __getitem__(self, idx) -> "GenericAD":
        selected = np.atleast_1d(np.arange(len(self))[idx])
        return GenericAD(self.val[selected], [j.take_rows(selected) for j in self.jac])

    def __setitem__(self, idx, val) -> None:
        selected = np.atleast_1d(np.arange(len(self))[idx])
        keep = np.ones(len(self))
        keep[selected] = 0.0
        if isinstance(val, GenericAD):
            new_val = val if len(val) == selected.size else val._expand(selected.size)
            projector = sp.csr_array(
                (np.ones(selected.size), (selected, np.arange(selected.size))),
                shape=(len(self), selected.size),
            )
            self.jac = [
                old.left_scale(keep) + new.matmul_left(projector)
                for old, new in zip(self.jac, new_val.jac)
            ]
            self.val[selected] = new_val.val
        elif isinstance(val, ADI):
            converted = GenericAD(value(val), [DiagonalJacobian.from_sparse(j) for j in val.jac])
            self[idx] = converted
        else:
            self.jac = [old.left_scale(keep) for old in self.jac]
            self.val[selected] = val


def _is_diagonal_block(block) -> bool:
    return isinstance(block, (DiagonalJacobian, RowMajorDiagonalJacobian))


def _block_type(row_major: bool):
    return RowMajorDiagonalJacobian if row_major else DiagonalJacobian


def _stack_block_column(blocks: list):
    if isinstance(blocks[0], RowMajorDiagonalJacobian):
        ncol = blocks[0].shape[1]
        for block in blocks:
            if block.shape[1] != ncol:
                raise ValueError("all Jacobian blocks must have the same column count")
        if HAS_NUMBA_KERNELS and vstack_row_major is not None:
            data, col, indptr, nrow = vstack_row_major(
                [block.data for block in blocks],
                [block.col for block in blocks],
                [block.indptr for block in blocks],
                [block.shape[0] for block in blocks],
            )
            return RowMajorDiagonalJacobian(data, col, indptr, (nrow, ncol))
        data = []
        col = []
        indptr = [0]
        nnz = 0
        nrow = 0
        for block in blocks:
            data.append(block.data)
            col.append(block.col)
            counts = np.diff(block.indptr)
            nnz_offsets = nnz + np.cumsum(counts)
            indptr.extend(nnz_offsets.tolist())
            nnz += block.nnz
            nrow += block.shape[0]
        return RowMajorDiagonalJacobian(
            np.concatenate(data) if data else np.array([], dtype=float),
            np.concatenate(col) if col else np.array([], dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
            (nrow, ncol),
        )

    ncol = blocks[0].shape[1]
    data, row, col = [], [], []
    offset = 0
    for block in blocks:
        if block.shape[1] != ncol:
            raise ValueError("all Jacobian blocks must have the same column count")
    if HAS_NUMBA_KERNELS and vstack_triplet is not None:
        data, row, col, nrow = vstack_triplet(
            [block.data for block in blocks],
            [block.row for block in blocks],
            [block.col for block in blocks],
            [block.shape[0] for block in blocks],
        )
        return DiagonalJacobian(data, row, col, (nrow, ncol))
    for block in blocks:
        data.append(block.data)
        row.append(block.row + offset)
        col.append(block.col)
        offset += block.shape[0]
    return DiagonalJacobian(
        np.concatenate(data) if data else np.array([], dtype=float),
        np.concatenate(row) if row else np.array([], dtype=np.int64),
        np.concatenate(col) if col else np.array([], dtype=np.int64),
        (offset, ncol),
    )


def init_variables_diagonal(*values: Numeric, row_major: bool = False) -> tuple[GenericAD, ...]:
    """Declare diagonal-backend primary variables with identity blocks."""
    vals = [np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel() for v in values]
    sizes = [v.size for v in vals]
    block_cls = _block_type(row_major)
    out = []
    for i, val in enumerate(vals):
        blocks = [
            block_cls.identity(val.size) if i == j else block_cls.zeros(val.size, m)
            for j, m in enumerate(sizes)
        ]
        out.append(GenericAD(val, blocks))
    return tuple(out)


def init_variables_diagonal_grouped(
    *values: Numeric,
    types=None,
    row_major: bool = False,
) -> tuple[GenericAD, ...]:
    """MRST grouped diagonal initializer.

    Variables with the same type share one Jacobian block. By default MRST
    groups consecutive variables of equal size; an explicit ``types`` vector
    can select other groups, provided every variable in a group has equal size.
    """
    vals = [np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel() for v in values]
    sizes = np.asarray([v.size for v in vals], dtype=np.int64)
    if types is None:
        labels = np.ones(len(vals), dtype=np.int64)
        for i in range(1, len(vals)):
            labels[i] = labels[i - 1] + (sizes[i] != sizes[i - 1])
    else:
        labels = np.asarray(types, dtype=np.int64).ravel()
        if labels.size != len(vals):
            raise ValueError("types must contain one entry per value")
    unique = list(dict.fromkeys(labels.tolist()))
    block_cls = _block_type(row_major)
    group_columns = {}
    for label in unique:
        members = np.flatnonzero(labels == label)
        member_sizes = sizes[members]
        if not np.all(member_sizes == member_sizes[0]):
            raise ValueError("variables in one diagonal type must have equal size")
        group_columns[label] = int(member_sizes[0] * members.size)

    out = []
    for variable, val in enumerate(vals):
        blocks = []
        for label in unique:
            ncols = group_columns[label]
            if labels[variable] != label:
                blocks.append(block_cls.zeros(val.size, ncols))
                continue
            members = np.flatnonzero(labels == label)
            position = int(np.flatnonzero(members == variable)[0])
            row = np.arange(val.size, dtype=np.int64)
            col = position * val.size + row
            matrix = sp.csr_array(
                (np.ones(val.size), (row, col)), shape=(val.size, ncols)
            )
            blocks.append(block_cls.from_sparse(matrix))
        out.append(GenericAD(val, blocks))
    return tuple(out)


def init_variables_one_block(*values: Numeric) -> tuple[GenericAD, ...]:
    """MRST ``initVariablesAD_oneBlock`` sparse global-block initializer."""
    vals = [np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel() for v in values]
    sizes = [v.size for v in vals]
    total = int(sum(sizes))
    offset = 0
    out = []
    for val in vals:
        row = np.arange(val.size, dtype=np.int64)
        col = offset + row
        jac = sp.csr_array(
            (np.ones(val.size), (row, col)), shape=(val.size, total)
        )
        out.append(GenericAD(val, [jac]))
        offset += val.size
    return tuple(out)


def diagonal_ad_matmul(mat, x):
    """Matrix product for ``GenericAD`` values."""
    if isinstance(x, GenericAD):
        return GenericAD(mat @ x.val, [j.matmul_left(mat) for j in x.jac])
    return mat @ x


def diagonal_sum(x):
    """Sum a diagonal AD vector to a scalar."""
    if isinstance(x, GenericAD):
        import scipy.sparse as sp

        ones = sp.csr_array(np.ones((1, len(x))))
        return diagonal_ad_matmul(ones, x)
    return np.sum(x)


def combine_diagonal_equations(*eqs) -> ADI:
    """Stack diagonal-backend equations and assemble sparse Jacobian for solve."""
    parts = []
    ads = [eq for eq in eqs if isinstance(eq, GenericAD)]
    if not ads:
        return ADI(np.concatenate([np.atleast_1d(eq).ravel() for eq in eqs]), [])
    sizes = [j.shape[1] for j in ads[0].jac]
    block_cls = type(ads[0].jac[0])
    for eq in eqs:
        if isinstance(eq, GenericAD):
            parts.append(eq)
        else:
            val = np.atleast_1d(np.asarray(eq, dtype=np.float64)).ravel()
            parts.append(GenericAD(val, [block_cls.zeros(val.size, size) for size in sizes]))
    val = np.concatenate([part.val for part in parts])
    jac = []
    for block_no in range(len(sizes)):
        jac.append(_stack_block_column([part.jac[block_no] for part in parts]).to_sparse())

    return ADI(val, [sp.hstack(jac, format="csr")])


if __name__ == "__main__":
    def test_generic_ad_arithmetic_matches_sparse_adi():
        from pymrst.utils.adi.init_variables_adi import init_variables_adi

        (xd,) = init_variables_diagonal(np.array([1.0, 2.0, 4.0]))
        (xs,) = init_variables_adi(np.array([1.0, 2.0, 4.0]))
        yd = (xd * xd + 2.0) / xd
        ys = (xs * xs + 2.0) / xs
        assert np.allclose(yd.val, ys.val)
        assert np.allclose(yd.to_adi().jac[0].toarray(), ys.jac[0].toarray())
        print("ok test_generic_ad_arithmetic_matches_sparse_adi")

    def test_combine_diagonal_equations():
        x, b = init_variables_diagonal(np.array([1.0, 2.0]), 5.0)
        eq = combine_diagonal_equations(x * 3.0 + b, b * 2.0)
        assert np.allclose(eq.val, [8.0, 11.0, 10.0])
        assert np.allclose(eq.jac[0].toarray(), [[3, 0, 1], [0, 3, 1], [0, 0, 2]])
        print("ok test_combine_diagonal_equations")

    def test_setitem_matches_sparse_adi():
        x, b = init_variables_diagonal(np.array([1.0, 2.0, 3.0]), 5.0)
        y = x * 2.0
        y[[0, 2]] = b + np.array([1.0, 2.0])
        assert np.allclose(y.val, [6.0, 4.0, 7.0])
        assert np.allclose(y.to_adi().jac[0].toarray(), [[0, 0, 0], [0, 2, 0], [0, 0, 0]])
        assert np.allclose(y.to_adi().jac[1].toarray(), [[1], [0], [1]])
        print("ok test_setitem_matches_sparse_adi")

    def test_row_major_generic_ad_matches_standard_diagonal():
        (xd,) = init_variables_diagonal(np.array([1.0, 2.0, 4.0]))
        (xr,) = init_variables_diagonal(np.array([1.0, 2.0, 4.0]), row_major=True)
        yd = (xd * xd + 2.0) / xd
        yr = (xr * xr + 2.0) / xr
        assert isinstance(yr.jac[0], RowMajorDiagonalJacobian)
        assert np.allclose(yr.val, yd.val)
        assert np.allclose(yr.to_adi().jac[0].toarray(), yd.to_adi().jac[0].toarray())
        print("ok test_row_major_generic_ad_matches_standard_diagonal")

    test_generic_ad_arithmetic_matches_sparse_adi()
    test_combine_diagonal_equations()
    test_setitem_matches_sparse_adi()
    test_row_major_generic_ad_matches_standard_diagonal()
    print("\nAll tests passed!")
