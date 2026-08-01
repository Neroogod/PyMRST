"""Eager subset of MRST ``vemmech/ad_version`` tensor utilities.

MRST's AD VEM implementation uses ``SparseMultiArray`` as a lazy sparse tensor
class. This module ports the eager numeric/ADI-coefficient subset needed by the
first AD VEM helpers and calibration utilities. Indices are stored 1-based to
match MRST.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.utils.adi import ADI, ad_matmul, ad_sum, value, vertcat_adi


def _names(indexnames) -> list[str]:
    if isinstance(indexnames, str):
        return [indexnames]
    return list(indexnames)


def _compute_1d_index(ixs: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    ixs = np.asarray(ixs, dtype=np.int64)
    if ixs.ndim == 1:
        ixs = ixs[:, None]
    shape = np.asarray(shape, dtype=np.int64).ravel()
    stride = np.concatenate([[1], np.cumprod(shape[:-1])])
    return ((ixs - 1) * stride).sum(axis=1)


def _coef_array(coefs):
    if isinstance(coefs, ADI):
        return coefs
    return np.asarray(coefs, dtype=float).ravel()


def _coef_size(coefs) -> int:
    return len(coefs) if isinstance(coefs, ADI) else np.asarray(coefs).size


def _coef_take(coefs, idx):
    if isinstance(coefs, ADI):
        return coefs[idx]
    arr = np.atleast_1d(np.asarray(coefs, dtype=float)).ravel()
    return arr[idx]


def _coef_concat(parts):
    if any(isinstance(p, ADI) for p in parts):
        return vertcat_adi(*parts)
    return np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts])


def _coef_values(coefs):
    return np.atleast_1d(value(coefs) if isinstance(coefs, ADI) else np.asarray(coefs, dtype=float)).ravel()


@dataclass
class SparseMultiArray:
    """Small eager counterpart of MRST ``SparseMultiArray``.

    ``ixs`` are 1-based multi-indices, one row per stored value. Duplicate
    entries are summed on construction and after arithmetic.
    """

    coefs: np.ndarray | ADI
    ixs: np.ndarray
    indexnames: list[str]
    keep_zeros: bool = False

    def __init__(self, coefs, ixs=None, indexnames=None, keep_zeros=False):
        self.keep_zeros = bool(keep_zeros)
        if ixs is None and indexnames is None:
            arr = _coef_array(coefs)
            if isinstance(arr, ADI) or np.asarray(arr).ndim == 0 or np.asarray(arr).size == 1:
                self.coefs = arr if isinstance(arr, ADI) else np.asarray([float(np.asarray(arr).ravel()[0])])
                self.ixs = np.zeros((1, 0), dtype=np.int64)
                self.indexnames = []
            else:
                raise ValueError("array construction requires indexnames")
        elif indexnames is None:
            names = _names(ixs)
            if isinstance(coefs, ADI):
                arr = coefs
                arr_val = coefs.val
            else:
                arr = np.asarray(coefs, dtype=float)
                arr_val = arr
            if arr_val.ndim == 1:
                nz = np.arange(arr_val.size) if self.keep_zeros else np.flatnonzero(arr_val != 0)
                self.coefs = _coef_take(arr, nz)
                self.ixs = (nz + 1)[:, None]
            elif arr_val.ndim == 2:
                if self.keep_zeros:
                    ii, jj = np.indices(arr_val.shape)
                    ii, jj = ii.ravel(), jj.ravel()
                else:
                    ii, jj = np.nonzero(arr_val != 0)
                self.coefs = _coef_take(arr, np.ravel_multi_index((ii, jj), arr_val.shape))
                self.ixs = np.column_stack([ii + 1, jj + 1])
            else:
                nz = np.array(np.indices(arr_val.shape)).reshape(arr_val.ndim, -1).T \
                    if self.keep_zeros else np.array(np.nonzero(arr_val != 0)).T
                self.coefs = _coef_take(arr, np.ravel_multi_index(tuple(nz.T), arr_val.shape))
                self.ixs = nz + 1
            self.indexnames = names
            self._coalesce()
        else:
            self.indexnames = _names(indexnames)
            ixarr = np.asarray(ixs, dtype=np.int64)
            if ixarr.ndim == 1:
                ixarr = ixarr[:, None]
            self.ixs = ixarr.reshape(-1, len(self.indexnames))
            if coefs is None or (not isinstance(coefs, ADI) and np.asarray(coefs).size == 0):
                self.coefs = np.ones(self.ixs.shape[0], dtype=float)
            else:
                vals = _coef_array(coefs)
                if _coef_size(vals) == 1 and self.ixs.shape[0] != 1:
                    if isinstance(vals, ADI):
                        vals = vals._expand(self.ixs.shape[0])
                    else:
                        vals = np.full(self.ixs.shape[0], vals[0], dtype=float)
                if _coef_size(vals) != self.ixs.shape[0]:
                    raise ValueError("number of coefficients must match index rows")
                self.coefs = vals
            self._coalesce()

    def copy(self) -> "SparseMultiArray":
        coefs = self.coefs.copy() if hasattr(self.coefs, "copy") else self.coefs
        return SparseMultiArray(coefs, self.ixs.copy(),
                                list(self.indexnames),
                                keep_zeros=self.keep_zeros)

    @property
    def shape(self) -> np.ndarray:
        if self.ixs.size == 0:
            return np.zeros(len(self.indexnames), dtype=np.int64)
        return self.ixs.max(axis=0)

    def _coalesce(self):
        if _coef_size(self.coefs) == 0:
            self.ixs = self.ixs.reshape(0, len(self.indexnames))
            return
        if len(self.indexnames) == 0:
            self.coefs = ad_sum(self.coefs) if isinstance(self.coefs, ADI) else np.asarray([np.asarray(self.coefs).sum()], dtype=float)
            self.ixs = np.zeros((1, 0), dtype=np.int64)
            return
        order = np.lexsort(self.ixs[:, ::-1].T)
        ixs = self.ixs[order]
        vals = _coef_take(self.coefs, order)
        keep = np.ones(_coef_size(vals), dtype=bool)
        keep[1:] = np.any(ixs[1:] != ixs[:-1], axis=1)
        starts = np.flatnonzero(keep)
        groups = np.cumsum(keep) - 1
        reducer = sp.csr_array(
            (np.ones(_coef_size(vals)), (groups, np.arange(_coef_size(vals)))),
            shape=(starts.size, _coef_size(vals)),
        )
        sums = ad_matmul(reducer, vals) if isinstance(vals, ADI) else reducer @ vals
        sum_values = _coef_values(sums)
        nz = np.ones(sum_values.size, dtype=bool) if self.keep_zeros else sum_values != 0
        self.ixs = ixs[starts][nz]
        self.coefs = _coef_take(sums, nz)

    def index_names(self) -> list[str]:
        return list(self.indexnames)

    def change_index_name(self, oldname: str, newname: str) -> "SparseMultiArray":
        out = self.copy()
        out.indexnames = [newname if n == oldname else n for n in out.indexnames]
        return out

    def sort_indices(self, ixset_order: Sequence[str]) -> "SparseMultiArray":
        order = [self.indexnames.index(n) for n in ixset_order]
        return SparseMultiArray(self.coefs, self.ixs[:, order],
                                list(ixset_order),
                                keep_zeros=self.keep_zeros)

    def to_indicator(self) -> "SparseMultiArray":
        return SparseMultiArray(np.ones(_coef_size(self.coefs)), self.ixs,
                                self.indexnames)

    def sub(self, ixname: str, ixval: int) -> "SparseMultiArray":
        col = self.indexnames.index(ixname)
        keep = self.ixs[:, col] == ixval
        names = [n for n in self.indexnames if n != ixname]
        ixs = np.delete(self.ixs[keep], col, axis=1)
        return SparseMultiArray(_coef_take(self.coefs, keep), ixs, names,
                                keep_zeros=self.keep_zeros)

    def __add__(self, other):
        other = _as_tensor(other)
        a, b = _compatible(self, other)
        return SparseMultiArray(_coef_concat([a.coefs, b.coefs]),
                                np.vstack([a.ixs, b.ixs]), a.indexnames,
                                keep_zeros=a.keep_zeros or b.keep_zeros)

    def __sub__(self, other):
        other = _as_tensor(other)
        a, b = _compatible(self, other)
        return SparseMultiArray(_coef_concat([a.coefs, -b.coefs]),
                                np.vstack([a.ixs, b.ixs]), a.indexnames,
                                keep_zeros=a.keep_zeros or b.keep_zeros)

    def __mul__(self, other):
        other = _as_tensor(other)
        common = [n for n in self.indexnames if n in other.indexnames]
        rows = []
        vals = []
        out_names = [n for n in self.indexnames if n not in common] + [
            n for n in other.indexnames if n not in common
        ]
        a_common = [self.indexnames.index(n) for n in common]
        b_common = [other.indexnames.index(n) for n in common]
        a_keep = [i for i, n in enumerate(self.indexnames) if n not in common]
        b_keep = [i for i, n in enumerate(other.indexnames) if n not in common]
        for ia in range(_coef_size(self.coefs)):
            va = _coef_take(self.coefs, ia)
            if common:
                match = np.all(other.ixs[:, b_common] == self.ixs[ia, a_common],
                               axis=1)
                js = np.flatnonzero(match)
            else:
                js = np.arange(_coef_size(other.coefs))
            for jb in js:
                rows.append(np.concatenate([
                    self.ixs[ia, a_keep], other.ixs[jb, b_keep]
                ]))
                vals.append(va * _coef_take(other.coefs, jb))
        if not rows:
            return SparseMultiArray([], np.zeros((0, len(out_names)), dtype=int),
                                    out_names,
                                    keep_zeros=self.keep_zeros
                                    or other.keep_zeros)
        return SparseMultiArray(_coef_concat(vals), np.vstack(rows), out_names,
                                keep_zeros=self.keep_zeros
                                or other.keep_zeros)

    def semiproduct(self, other) -> "SparseMultiArray":
        """Tensor product without contraction, equivalent to MRST ``^`` for
        the non-overlapping-index utility cases used here."""
        other = _as_tensor(other)
        if set(self.indexnames) & set(other.indexnames):
            raise NotImplementedError("semiproduct with duplicate index names")
        vals = [
            _coef_take(self.coefs, ia) * _coef_take(other.coefs, ib)
            for ia in range(_coef_size(self.coefs))
            for ib in range(_coef_size(other.coefs))
        ]
        left = np.repeat(self.ixs, _coef_size(other.coefs), axis=0)
        right = np.tile(other.ixs, (_coef_size(self.coefs), 1))
        return SparseMultiArray(_coef_concat(vals), np.column_stack([left, right]),
                                self.indexnames + other.indexnames,
                                keep_zeros=self.keep_zeros
                                or other.keep_zeros)

    def as_vector(self, ixnames: Sequence[str], shape=None):
        ixnames = _names(ixnames)
        t = self.sort_indices(ixnames)
        if shape is None:
            shape = t.shape
        ix = _compute_1d_index(t.ixs, shape)
        if isinstance(t.coefs, ADI):
            selector = sp.csr_array(
                (np.ones(ix.size), (ix, np.arange(ix.size))),
                shape=(int(np.prod(shape)), ix.size),
            )
            return ad_matmul(selector, t.coefs)
        return sp.csr_array((t.coefs, (ix, np.zeros(ix.size, dtype=int))),
                            shape=(int(np.prod(shape)), 1))

    def as_full_vector(self, ixnames: Sequence[str], shape=None) -> np.ndarray:
        vec = self.as_vector(ixnames, shape)
        if isinstance(vec, ADI):
            return vec
        return np.asarray(vec.todense()).ravel()

    def as_matrix(self, ixnames, force_sparse: bool = False, shape=None):
        if isinstance(ixnames, str):
            ixnames = [[ixnames]]
        else:
            ixnames = [(_names(x) if not isinstance(x, str) else [x])
                       for x in ixnames]
        flat_names = [n for part in ixnames for n in part]
        t = self.sort_indices(flat_names)
        if shape is None:
            shape = t.shape
        shape = np.asarray(shape, dtype=np.int64)
        if len(ixnames) == 1:
            return self.as_vector(flat_names, shape)
        nrow = len(ixnames[0])
        row_shape = shape[:nrow]
        col_shape = shape[nrow:]
        row = _compute_1d_index(t.ixs[:, :nrow], row_shape)
        col = _compute_1d_index(t.ixs[:, nrow:], col_shape)
        mat = sp.csr_array((_coef_values(t.coefs), (row, col)),
                           shape=(int(np.prod(row_shape)),
                                  int(np.prod(col_shape))))
        if not force_sparse and mat.size < 200:
            return np.asarray(mat.todense())
        return mat


@dataclass(frozen=True)
class LazySparseTensor:
    """Pure-Python lazy expression graph for sparse tensor operations.

    MRST's ``tbinaryop``/``tcontract`` machinery stores tensor operations in a
    compiled lazy backend. This class records the same high-level operation
    graph in Python and evaluates to ``SparseMultiArray`` only at realization
    boundaries.
    """

    op: str
    args: tuple
    indexnames: tuple[str, ...] | None = None

    @staticmethod
    def leaf(tensor) -> "LazySparseTensor":
        tensor = _as_tensor(tensor)
        return LazySparseTensor("leaf", (tensor,),
                                tuple(tensor.indexnames))

    def evaluate(self) -> SparseMultiArray:
        if self.op == "leaf":
            return self.args[0].copy()
        if self.op == "add":
            return self.args[0].evaluate() + self.args[1].evaluate()
        if self.op == "sub":
            return self.args[0].evaluate() - self.args[1].evaluate()
        if self.op == "mul":
            return self.args[0].evaluate() * self.args[1].evaluate()
        if self.op == "semiproduct":
            return self.args[0].evaluate().semiproduct(self.args[1].evaluate())
        if self.op == "sort":
            return self.args[0].evaluate().sort_indices(self.args[1])
        if self.op == "subindex":
            return self.args[0].evaluate().sub(self.args[1], self.args[2])
        if self.op == "indicator":
            return self.args[0].evaluate().to_indicator()
        raise ValueError(f"unknown lazy tensor op {self.op!r}")

    def materialize(self) -> SparseMultiArray:
        return self.evaluate()

    def index_names(self) -> list[str]:
        return list(self.indexnames or self.evaluate().indexnames)

    def graph(self):
        if self.op == "leaf":
            tensor = self.args[0]
            return {
                "op": "leaf",
                "indexnames": list(tensor.indexnames),
                "nnz": _coef_size(tensor.coefs),
            }
        return {
            "op": self.op,
            "indexnames": self.index_names(),
            "args": [
                arg.graph() if isinstance(arg, LazySparseTensor) else arg
                for arg in self.args
            ],
        }

    def sort_indices(self, ixset_order: Sequence[str]) -> "LazySparseTensor":
        names = tuple(_names(ixset_order))
        return LazySparseTensor("sort", (self, names), names)

    def to_indicator(self) -> "LazySparseTensor":
        return LazySparseTensor("indicator", (self,), self.indexnames)

    def sub(self, ixname: str, ixval: int) -> "LazySparseTensor":
        names = tuple(n for n in self.index_names() if n != ixname)
        return LazySparseTensor("subindex", (self, ixname, ixval), names)

    def semiproduct(self, other) -> "LazySparseTensor":
        other = lazy_sparse_tensor(other)
        names = tuple(self.index_names() + other.index_names())
        return LazySparseTensor("semiproduct", (self, other), names)

    def as_vector(self, ixnames: Sequence[str], shape=None):
        return self.evaluate().as_vector(ixnames, shape)

    def as_full_vector(self, ixnames: Sequence[str], shape=None):
        return self.evaluate().as_full_vector(ixnames, shape)

    def as_matrix(self, ixnames, force_sparse: bool = False, shape=None):
        return self.evaluate().as_matrix(ixnames, force_sparse, shape)

    def __add__(self, other):
        other = lazy_sparse_tensor(other)
        return LazySparseTensor("add", (self, other),
                                tuple(_compatible_names(self, other)))

    def __sub__(self, other):
        other = lazy_sparse_tensor(other)
        return LazySparseTensor("sub", (self, other),
                                tuple(_compatible_names(self, other)))

    def __mul__(self, other):
        other = lazy_sparse_tensor(other)
        common = [n for n in self.index_names() if n in other.index_names()]
        names = [n for n in self.index_names() if n not in common] + [
            n for n in other.index_names() if n not in common
        ]
        return LazySparseTensor("mul", (self, other), tuple(names))


def lazy_sparse_tensor(tensor) -> LazySparseTensor:
    """Return a lazy tensor expression for a tensor-like input."""
    return tensor if isinstance(tensor, LazySparseTensor) else LazySparseTensor.leaf(tensor)


def tbinaryop(op: str, a, b) -> LazySparseTensor:
    """Pure-Python counterpart of MRST's lazy tensor binary operation."""
    if op in ("+", "plus", "add"):
        return lazy_sparse_tensor(a) + b
    if op in ("-", "minus", "sub"):
        return lazy_sparse_tensor(a) - b
    if op in ("*", "times", "mtimes", "contract"):
        return lazy_sparse_tensor(a) * b
    if op in ("^", "semiproduct", "kron"):
        return lazy_sparse_tensor(a).semiproduct(b)
    raise ValueError(f"unsupported tensor binary op {op!r}")


def tcontract(a, b=None) -> LazySparseTensor:
    """Lazy contraction helper.

    With one argument, this realizes duplicate-index coalescing lazily by
    wrapping the tensor as a leaf. With two arguments, it records multiplication
    and contraction over common index names.
    """
    if b is None:
        return lazy_sparse_tensor(a)
    return lazy_sparse_tensor(a) * b


def _as_tensor(x) -> SparseMultiArray:
    return x if isinstance(x, SparseMultiArray) else SparseMultiArray(x)


def _compatible(a: SparseMultiArray, b: SparseMultiArray):
    if set(a.indexnames) != set(b.indexnames):
        raise ValueError("tensor index names are not compatible")
    order = [b.indexnames.index(n) for n in a.indexnames]
    return a, SparseMultiArray(b.coefs, b.ixs[:, order], a.indexnames,
                               keep_zeros=b.keep_zeros)


def _compatible_names(a: LazySparseTensor, b: LazySparseTensor):
    names = a.index_names()
    if set(names) != set(b.index_names()):
        raise ValueError("tensor index names are not compatible")
    return names


def kron_adi(u, v):
    """Eager counterpart of MRST ``kron_ADI`` for numeric or ADI vectors."""
    if not isinstance(u, ADI) and not isinstance(v, ADI):
        return np.kron(np.asarray(u, dtype=float).ravel(),
                       np.asarray(v, dtype=float).ravel())
    vals = [
        _coef_take(u, i) * _coef_take(v, j)
        for i in range(_coef_size(u))
        for j in range(_coef_size(v))
    ]
    return _coef_concat(vals)


def face_node_tensor(g, fname: str, nname: str, values=None,
                     boundary_only: bool = False) -> SparseMultiArray:
    """Construct the face-node indicator tensor for a grid."""
    face_no = np.repeat(np.arange(g.faces.num) + 1, np.diff(g.faces.nodePos))
    nodes = g.faces.nodes + 1
    ixs = np.column_stack([face_no, nodes])
    vals = None if values is None else _coef_array(values)
    if boundary_only:
        boundary = np.any(g.faces.neighbors < 0, axis=1)
        keep = np.repeat(boundary, np.diff(g.faces.nodePos))
        ixs = ixs[keep]
        if vals is not None:
            vals = _coef_take(vals, keep)
    return SparseMultiArray(vals, ixs, [fname, nname])


def enu2c_ad(E, nu, g, ixnames=("i", "j", "c")) -> SparseMultiArray:
    """Eager tensor counterpart of MRST ``Enu2C_AD``.

    Numeric inputs return numeric coefficients. ADI Young's modulus and/or
    Poisson ratio inputs preserve first-order Jacobians in the tensor
    coefficients, matching the coefficient-level behavior MRST's lazy tensor
    path provides before sparse matrix realization.
    """
    if g.griddim != 3:
        raise ValueError("Enu2C_AD only supports 3D grids in MRST")
    vals = []
    ixs = []
    for c in range(g.cells.num):
        Ec = _coef_take(E, c)
        nuc = _coef_take(nu, c)
        fac = Ec / ((1.0 + nuc) * (1.0 - 2.0 * nuc))
        normal_diag = fac * (1.0 - nuc)
        normal_offdiag = fac * nuc
        shear = fac * ((1.0 - 2.0 * nuc) / 2.0)
        for i in range(3):
            for j in range(3):
                vals.append(normal_diag if i == j else normal_offdiag)
                ixs.append([i + 1, j + 1, c + 1])
        for i in range(3, 6):
            vals.append(shear)
            ixs.append([i + 1, i + 1, c + 1])
    return SparseMultiArray(_coef_concat(vals), np.asarray(ixs, dtype=np.int64),
                            ixnames)


def _dense_vec(x):
    return x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()


def _value_vec(x):
    return np.asarray(value(x), dtype=float).ravel()


def _concat_jac(*args):
    ads = [x for x in args if isinstance(x, ADI)]
    if not ads:
        return []
    nblocks = len(ads[0].jac)
    sizes = [ads[0].jac[k].shape[1] for k in range(nblocks)]
    out = []
    for x in args:
        if isinstance(x, ADI):
            if len(x.jac) != nblocks:
                raise ValueError("ADI inputs must use the same Jacobian blocks")
            out.append(sp.hstack(x.jac, format="csr"))
        else:
            val = np.asarray(x, dtype=float).ravel()
            out.append(sp.csr_array((val.size, sum(sizes))))
    return out


def _ad_geometry_operators(g):
    from pymrst.modules.vemmech.vem import _cell_projections, _nlin, calculate_qc, calculate_qf

    dim = g.griddim
    nlin = _nlin(g)
    qc_all = calculate_qc(g)[0] if dim == 3 else calculate_qf(g)[0]
    I_blocks, vol_blocks, WC_blocks, NC_blocks, PP_blocks = [], [], [], [], []
    rows, cols, vals = [], [], []
    local_offset = 0
    for c in range(g.cells.num):
        NC, NR, WC, WR, dofs = _cell_projections(g, c, qc_all)
        PP = NR @ WR.T + NC @ WC.T
        nd = dofs.size
        I_blocks.append(sp.eye(nd, format="csr"))
        vol_blocks.append(sp.eye(nd, format="csr") * g.cells.volumes[c])
        WC_blocks.append(sp.csr_array(WC))
        NC_blocks.append(sp.csr_array(NC))
        PP_blocks.append(sp.csr_array(PP))
        local = local_offset + np.arange(nd)
        rows.append(dofs)
        cols.append(local)
        vals.append(np.ones(nd))
        local_offset += nd
    ndof = dim * g.nodes.num
    assemb = sp.csr_array((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(ndof, local_offset))
    return SimpleNamespace(
        I=sp.block_diag(I_blocks, format="csr"),
        volmap=sp.block_diag(vol_blocks, format="csr"),
        WC=sp.block_diag(WC_blocks, format="csr"),
        NC=sp.block_diag(NC_blocks, format="csr"),
        PP=sp.block_diag(PP_blocks, format="csr"),
        assemb=assemb,
        nlin=nlin,
    )


def _material_d_matrix(g, E, nu):
    from pymrst.modules.vemmech.vem import c2d, enu2c, enu2c_2d

    C = enu2c_2d(_value_vec(E), _value_vec(nu)) if g.griddim == 2 else enu2c(
        _value_vec(E), _value_vec(nu)
    )
    nlin = 3 if g.griddim == 2 else 6
    D_all = c2d(C).reshape(g.cells.num, nlin, nlin)
    return sp.block_diag([D_all[c] for c in range(g.cells.num)], format="csr")


def compute_ax_derivs(g, u, extra, E, nu, gamma: float = 1.0,
                      dirdofs=None):
    """Port of MRST ``compute_Ax_derivs`` for material AD variables.

    The returned matrix is ``d(S(E, nu) * u) / dcontrols`` restricted to free
    displacement rows. ``E`` and/or ``nu`` may be PyMRST ``ADI`` objects; for
    plain numeric inputs this returns an empty derivative matrix.
    """
    E_val = _value_vec(E)
    nu_val = _value_vec(nu)
    jac = _concat_jac(E, nu)
    if not jac:
        nfree = g.griddim * g.nodes.num if dirdofs is None else np.count_nonzero(~dirdofs)
        return sp.csr_array((nfree, 0))
    ejac, nujac = jac
    if not isinstance(E, ADI):
        ejac = sp.csr_array((E_val.size, nujac.shape[1]))
    if not isinstance(nu, ADI):
        nujac = sp.csr_array((nu_val.size, ejac.shape[1]))
    num_ders = ejac.shape[1]
    if num_ders == 0:
        nfree = g.griddim * g.nodes.num if dirdofs is None else np.count_nonzero(~dirdofs)
        return sp.csr_array((nfree, 0))

    N = g.cells.num
    nlin = 3 if g.griddim == 2 else 6
    dim = g.griddim
    vols = np.asarray(g.cells.volumes, dtype=float).ravel()
    repNlin = sp.csr_array(
        (np.ones(nlin * N),
         (np.arange(nlin * N),
          np.repeat(np.arange(N), nlin))),
        shape=(nlin * N, N),
    )
    nlc = np.diff(g.cells.nodePos)
    nlocal = dim * int(np.sum(nlc))
    repX = sp.csr_array(
        (np.ones(nlocal),
         (np.arange(nlocal),
          np.repeat(np.arange(N), nlc * dim))),
        shape=(nlocal, N),
    )

    repE = repNlin @ E_val
    D_dE = extra.D @ sp.diags(1.0 / repE, format="csr")

    Ds = E_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    repDs = repNlin @ Ds
    Dm = extra.D @ sp.diags(1.0 / repDs, format="csr")
    Ds_dnu = E_val * (1.0 + 4.0 * nu_val) / (
        ((1.0 + nu_val) * (1.0 - 2.0 * nu_val)) ** 2
    )
    Dm_dnu = extra.D.copy().astype(float)
    Dm_dnu.eliminate_zeros()
    Dm_dnu.data = np.ones_like(Dm_dnu.data)
    diag_pattern = [-1.0, -1.0, -4.0] if dim == 2 else [
        -1.0, -1.0, -1.0, -4.0, -4.0, -4.0
    ]
    diag = np.tile(diag_pattern, N)
    Dm_dnu = Dm_dnu.tolil()
    Dm_dnu[np.arange(diag.size), np.arange(diag.size)] = diag
    Dm_dnu = Dm_dnu.tocsr()
    D_dnu = (
        Dm @ sp.diags(repNlin @ Ds_dnu, format="csr")
        + Dm_dnu @ sp.diags(repNlin @ Ds, format="csr")
    )

    DNC = np.asarray((extra.NC.T @ extra.NC).diagonal(), dtype=float)
    trDNC = DNC.reshape(N, nlin).sum(axis=1)
    c = gamma / trDNC * vols
    trD = np.asarray(extra.D.diagonal()).reshape(N, nlin).sum(axis=1)
    trD_dnu = np.asarray(D_dnu.diagonal()).reshape(N, nlin).sum(axis=1)
    alpha_dE = c * trD / E_val
    alpha_dnu = c * trD_dnu
    S_dE = sp.diags(repX @ alpha_dE, format="csr")
    S_dnu = sp.diags(repX @ alpha_dnu, format="csr")
    ImPP = extra.I - extra.PP
    local_u = extra.assemb.T @ np.asarray(u, dtype=float).ravel()

    EJ = ejac.multiply(vols[:, None])
    NJ = nujac.multiply(vols[:, None])
    ax_E = extra.assemb @ (extra.WC @ D_dE @ extra.WC.T) @ (
        (repX @ EJ).multiply(local_u[:, None])
    )
    ax_nu = extra.assemb @ (extra.WC @ D_dnu @ extra.WC.T) @ (
        (repX @ NJ).multiply(local_u[:, None])
    )
    ax_stab_E = extra.assemb @ (ImPP.T @ S_dE @ ImPP) @ (
        (repX @ ejac).multiply(local_u[:, None])
    )
    ax_stab_nu = extra.assemb @ (ImPP.T @ S_dnu @ ImPP) @ (
        (repX @ nujac).multiply(local_u[:, None])
    )
    ax = (ax_E + ax_nu + ax_stab_E + ax_stab_nu).tocsr()
    if dirdofs is None:
        return ax
    return ax[~np.asarray(dirdofs, dtype=bool), :]


def vem_assemble_ad(g, E, nu, extra=None, alpha_scaling: float = 1.0):
    """Numeric matrix subset of MRST ``VEM_assemble_AD``.

    Accepts numeric or ADI Young's modulus and Poisson ratio arrays and returns
    the stiffness matrix evaluated at their numeric values plus
    AD-version-style geometry/material operators. Matrix-level lazy AD is
    represented explicitly elsewhere through ``compute_ax_derivs`` rather than
    through MRST's MEX-backed sparse tensor factor graph.
    """
    from pymrst.modules.vemmech.vem import enu2c, enu2c_2d, vem_assemble

    E_val = _value_vec(E)
    nu_val = _value_vec(nu)
    C = enu2c_2d(E_val, nu_val) if g.griddim == 2 else enu2c(E_val, nu_val)
    S = vem_assemble(g, C, alpha_scaling=alpha_scaling)
    op = _ad_geometry_operators(g) if extra is None else SimpleNamespace(
        I=extra.I, volmap=extra.volmap, WC=extra.WC, NC=extra.NC,
        PP=extra.PP, assemb=extra.assemb,
        nlin=getattr(extra, "nlin", 3 if g.griddim == 2 else 6),
    )
    op.D = _material_d_matrix(g, E_val, nu_val)
    op.S = S
    op.Ax_derivs = 0
    return S, op


def _bc_vector_from_tensor(g, disp_bc: SparseMultiArray):
    vec = np.zeros(g.griddim * g.nodes.num, dtype=float)
    isdir = np.zeros(vec.size, dtype=bool)
    names = disp_bc.index_names()
    d_col = names.index("d")
    n_col = names.index("n")
    ixs = np.asarray(disp_bc.ixs, dtype=np.int64)
    vals = np.asarray(disp_bc.coefs, dtype=float)
    dofs = (ixs[:, d_col] - 1) + g.griddim * (ixs[:, n_col] - 1)
    keep = ~np.isnan(vals)
    vec[dofs[keep]] = vals[keep]
    isdir[dofs[keep]] = True
    return vec, isdir


def _load_vector_dual_grad(g, load: SparseMultiArray, qc_all):
    from pymrst.utils import mcolon

    cells = np.arange(g.cells.num)
    inodes = mcolon(g.cells.nodePos[cells], g.cells.nodePos[cells + 1])
    nodes = g.cells.nodes[inodes]
    nlc = np.diff(g.cells.nodePos)
    X = np.repeat(g.cells.centroids[cells, :g.griddim], nlc, axis=0)
    rel_vec = -(X - g.nodes.coords[nodes, :g.griddim])
    w = qc_all[:, :g.griddim] * rel_vec
    load_mat = load.as_matrix([["cn"], ["d"]], True,
                              [nodes.size, g.griddim])
    ll = np.asarray(load_mat.toarray() if hasattr(load_mat, "toarray")
                    else load_mat) * w
    ndof = g.griddim * g.nodes.num
    dofs = (g.griddim * nodes[:, None] + np.arange(g.griddim)).ravel()
    return np.bincount(dofs, weights=ll.ravel(), minlength=ndof)


def _force_bc_vector(g, force_bc: SparseMultiArray, qf_all):
    from pymrst.utils import mcolon

    mat = force_bc.as_matrix([["f"], ["d"]], True,
                             [g.faces.num, g.griddim])
    forces = np.asarray(mat.toarray() if hasattr(mat, "toarray") else mat)
    faces = np.flatnonzero(np.any(forces != 0, axis=1))
    if faces.size == 0:
        return np.zeros(g.griddim * g.nodes.num)
    lnn = np.diff(g.faces.nodePos)[faces]
    inodes = mcolon(g.faces.nodePos[faces], g.faces.nodePos[faces + 1])
    nodes = g.faces.nodes[inodes]
    if g.griddim == 2:
        weights = np.repeat(qf_all[faces], lnn)
    else:
        weights = qf_all[inodes]
    vals = np.repeat(forces[faces, :g.griddim], lnn, axis=0) * weights[:, None]
    dofs = (g.griddim * nodes[:, None] + np.arange(g.griddim)).ravel()
    return np.bincount(dofs, weights=vals.ravel(),
                       minlength=g.griddim * g.nodes.num)


def vem_reduced_system_residual(
    g,
    E,
    nu,
    el_bc,
    load,
    uu,
    alpha_scaling: float = 1.0,
    pressure=None,
    extra=None,
    background_forces=None,
):
    """Free-row residual of the AD VEM mechanics system for fixed ``uu``.

    This is the residual used by the eliminated calibration objective:
    ``F(u, x) = (S(u) * U - f(u))_free`` with prescribed displacement entries
    overwritten from ``el_bc``. It mirrors the reduced system solved by
    ``vem_lin_elast_ad`` and is useful for adjoint/finite-difference
    derivative checks.
    """
    from pymrst.modules.vemmech.vem import calculate_qc, calculate_qf, vem_div

    dim = g.griddim
    S, op = vem_assemble_ad(g, E, nu, extra=extra,
                            alpha_scaling=alpha_scaling)
    if dim == 2:
        qc_all, qcvol = calculate_qf(g)
        qf_all = g.faces.areas / 2.0
    else:
        qc_all, qf_all, qcvol = calculate_qc(g)

    u_bc, isdir = _bc_vector_from_tensor(g, el_bc.disp_bc)
    full = np.asarray(uu, dtype=float).reshape(-1).copy()
    if full.size != dim * g.nodes.num:
        raise ValueError("uu must contain one displacement vector per node")
    full[isdir] = u_bc[isdir]

    f = _load_vector_dual_grad(g, load, qc_all)
    if pressure is not None and np.asarray(pressure).size:
        f += vem_div(g).T @ np.asarray(pressure, dtype=float).ravel()
    if background_forces is not None and np.asarray(background_forces).size:
        f += np.asarray(background_forces, dtype=float).ravel()
    if getattr(el_bc, "force_bc", None) is not None:
        f += _force_bc_vector(g, el_bc.force_bc, qf_all)

    op.S = S
    return (S @ full - f)[~isdir], isdir, op


def vem_lin_elast_ad(
    g,
    E,
    nu,
    el_bc,
    load,
    linsolve=None,
    alpha_scaling: float = 1.0,
    pressure=None,
    extra=None,
    background_forces=None,
    no_solve: bool = False,
):
    """Numeric subset of MRST ``VEM_linElast_AD``.

    Uses AD-version inputs (`disp_bc`/`force_bc`/`load` as
    ``SparseMultiArray`` objects) and returns the solved displacement plus the
    MRST-shaped ``extra`` fields. Material AD inputs populate
    ``extra.Ax_derivs`` through an explicit eager derivative block;
    ``extra.Ax_derivs`` is zero for numeric inputs.
    """
    from pymrst.modules.vemmech.vem import calculate_qc, calculate_qf, vem_div

    dim = g.griddim
    S, op = vem_assemble_ad(g, E, nu, extra=extra,
                            alpha_scaling=alpha_scaling)
    if dim == 2:
        qc_all, qcvol = calculate_qf(g)
        qf_all = g.faces.areas / 2.0
    else:
        qc_all, qf_all, qcvol = calculate_qc(g)

    u_bc, isdir = _bc_vector_from_tensor(g, el_bc.disp_bc)
    rhs = -(S @ u_bc)
    rhs += _load_vector_dual_grad(g, load, qc_all)
    if pressure is not None and np.asarray(pressure).size:
        rhs += vem_div(g).T @ np.asarray(pressure, dtype=float).ravel()
    if background_forces is not None and np.asarray(background_forces).size:
        rhs += np.asarray(background_forces, dtype=float).ravel()
    if getattr(el_bc, "force_bc", None) is not None:
        rhs += _force_bc_vector(g, el_bc.force_bc, qf_all)

    free = ~isdir
    A = S[free][:, free]
    A = (A + A.T) / 2.0
    red_rhs = rhs[free]
    if no_solve:
        x = np.full(np.count_nonzero(free), np.nan)
    else:
        solver = linsolve or (lambda AA, bb: spla.spsolve(AA.tocsc(), bb))
        x = solver(A, red_rhs)

    u = np.full(dim * g.nodes.num, np.nan)
    u[isdir] = u_bc[isdir]
    u[free] = x
    uu = u.reshape(g.nodes.num, dim)
    vdiv = vem_div(g)
    op.A = A
    op.S = S
    op.rhs = red_rhs
    if isinstance(E, ADI) or isinstance(nu, ADI):
        op.Ax_derivs = compute_ax_derivs(g, u, op, E, nu, alpha_scaling, isdir)
    else:
        op.Ax_derivs = 0
    op.disc = {
        "isdirdofs": isdir,
        "rhs": red_rhs,
        "V_dir": u_bc,
        "ovol_div": vdiv,
        "gradP": vdiv[:, free].T,
        "div": vdiv[:, free],
        "divrhs": vdiv @ u_bc,
    }
    return uu, op


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_sparse_multiarray_matrix():
        t = SparseMultiArray([2, 3], [[1, 2], [2, 1]], ["i", "j"])
        m = t.as_matrix([["i"], ["j"]])
        assert np.allclose(m, [[0, 2], [3, 0]])
        print("ok test_sparse_multiarray_matrix")

    def test_product_contracts_common_index():
        a = SparseMultiArray([2, 3], [[1, 1], [1, 2]], ["i", "j"])
        b = SparseMultiArray([5, 7], [[1, 1], [2, 1]], ["j", "k"])
        c = a * b
        assert c.indexnames == ["i", "k"]
        assert np.allclose(c.as_full_vector(["i", "k"], [1, 1]), [31])
        print("ok test_product_contracts_common_index")

    def test_face_node_tensor_boundary():
        g = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        t = face_node_tensor(g, "f", "n", boundary_only=True)
        assert t.ixs.shape[0] == g.faces.nodes.size
        print("ok test_face_node_tensor_boundary")

    def test_keep_zeros_dense_constructor():
        t = SparseMultiArray(np.array([[0.0, 2.0], [3.0, 0.0]]),
                             ["i", "j"], keep_zeros=True)
        assert t.coefs.size == 4
        mat = t.as_matrix([["i"], ["j"]], True, [2, 2]).toarray()
        assert np.allclose(mat, [[0.0, 2.0], [3.0, 0.0]])
        print("ok test_keep_zeros_dense_constructor")

    def test_vem_ad_numeric_path():
        from pymrst.modules.vemmech.vem import create_augmented_grid

        g = create_augmented_grid(
            compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        )
        E = np.full(g.cells.num, 1e9)
        nu = np.full(g.cells.num, 0.25)
        bc = SimpleNamespace(
            disp_bc=SparseMultiArray(
                [0.0, 0.0, 0.0],
                [[1, 1], [1, 2], [1, 3]],
                ["n", "d"],
                keep_zeros=True,
            ),
            force_bc=None,
        )
        load = SparseMultiArray(np.zeros((g.cells.nodes.size, g.griddim)),
                                ["cn", "d"], keep_zeros=True)
        uu, extra = vem_lin_elast_ad(g, E, nu, bc, load, no_solve=True)
        assert np.allclose(uu[0], 0.0)
        assert np.isnan(uu[1:]).any()
        assert extra.disc["isdirdofs"].sum() == 3
        print("ok test_vem_ad_numeric_path")

    def test_compute_ax_derivs_matches_finite_difference():
        from pymrst.modules.vemmech.vem import create_augmented_grid
        from pymrst.utils.adi import init_variables_adi

        g = create_augmented_grid(
            compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        )
        E0 = np.full(g.cells.num, 1.1e9)
        nu0 = np.full(g.cells.num, 0.22)
        E, nu = init_variables_adi(E0, nu0)
        bc = SimpleNamespace(
            disp_bc=SparseMultiArray(
                [0.0, 0.0, 0.0, 0.02],
                [[1, 1], [1, 2], [1, 3], [8, 3]],
                ["n", "d"],
                keep_zeros=True,
            ),
            force_bc=None,
        )
        load = SparseMultiArray(np.zeros((g.cells.nodes.size, g.griddim)),
                                ["cn", "d"], keep_zeros=True)
        uu, extra = vem_lin_elast_ad(g, E, nu, bc, load)
        full_u = uu.reshape(-1)
        free = ~extra.disc["isdirdofs"]
        hE = 1e-5 * E0[0]
        hnu = 1e-6
        Sp, _ = vem_assemble_ad(g, E0 + hE, nu0)
        Sm, _ = vem_assemble_ad(g, E0 - hE, nu0)
        fd_E = ((Sp - Sm) @ full_u)[free] / (2.0 * hE)
        Sp, _ = vem_assemble_ad(g, E0, nu0 + hnu)
        Sm, _ = vem_assemble_ad(g, E0, nu0 - hnu)
        fd_nu = ((Sp - Sm) @ full_u)[free] / (2.0 * hnu)
        got = extra.Ax_derivs.toarray()
        assert np.allclose(got[:, 0], fd_E, rtol=1e-5, atol=1e-9)
        assert np.allclose(got[:, 1], fd_nu, rtol=1e-5, atol=1e-3)
        print("ok test_compute_ax_derivs_matches_finite_difference")

    test_sparse_multiarray_matrix()
    test_product_contracts_common_index()
    test_face_node_tensor_boundary()
    test_keep_zeros_dense_constructor()
    test_vem_ad_numeric_path()
    test_compute_ax_derivs_matches_finite_difference()
    print("\nAll tests passed!")
