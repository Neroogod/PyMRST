"""Small array backend facade for NumPy and optional Torch tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class ArrayBackend:
    """Numerical backend used by experimental PyMRST GPU/tensor code."""

    name: str = "numpy"
    device: str | None = None
    dtype: str | Any = "float64"
    linear_solver: str = "auto"

    @property
    def is_torch(self) -> bool:
        return self.name.lower() == "torch"

    @property
    def xp(self):
        if self.is_torch:
            return self.torch
        return np

    @property
    def torch(self):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install torch or `pip install -e .[ml]` for backend='torch'") from exc
        return torch

    def asarray(self, value, *, dtype=None):
        if self.is_torch:
            torch = self.torch
            tdtype = self.torch_dtype(dtype or self.dtype)
            if isinstance(value, np.ndarray) and any(s < 0 for s in value.strides):
                value = value.copy()
            if np.isscalar(value) and isinstance(value, (float, complex, np.floating, np.complexfloating)):
                return torch.as_tensor(value, device=self.device, dtype=tdtype)
            tensor = torch.as_tensor(value, device=self.device)
            if tensor.is_floating_point() or tensor.is_complex():
                tensor = tensor.to(dtype=tdtype)
            return tensor
        return np.asarray(value, dtype=np.dtype(dtype or self.dtype))

    def as_int(self, value):
        if self.is_torch:
            return self.torch.as_tensor(value, device=self.device, dtype=self.torch.long)
        return np.asarray(value, dtype=np.int64)

    def zeros(self, shape, *, dtype=None):
        if self.is_torch:
            return self.torch.zeros(shape, device=self.device, dtype=self.torch_dtype(dtype or self.dtype))
        return np.zeros(shape, dtype=np.dtype(dtype or self.dtype))

    def to_numpy(self, value) -> np.ndarray:
        if self.is_torch and hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def solve(self, matrix, rhs):
        """Solve a linear system on the selected backend.

        Torch sparse direct solves are still limited, so this first backend path
        converts small sparse matrices to dense tensors. This is intended for
        validation and GPU prototyping, not field-scale production yet.
        """
        solver = self.linear_solver.lower()
        if solver == "cupy":
            return self._solve_cupy(matrix, rhs)
        if solver == "scipy":
            out = spla.spsolve(matrix.tocsc(), np.asarray(rhs, dtype=np.dtype(self.dtype))) if sp.issparse(matrix) else np.linalg.solve(np.asarray(matrix, dtype=np.dtype(self.dtype)), rhs)
            return self.asarray(out) if self.is_torch else out
        if self.is_torch:
            torch = self.torch
            dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
            a = torch.as_tensor(dense, device=self.device, dtype=self.torch_dtype())
            b = torch.as_tensor(rhs, device=self.device, dtype=self.torch_dtype())
            return torch.linalg.solve(a, b)
        if sp.issparse(matrix):
            return spla.spsolve(matrix.tocsc(), np.asarray(rhs, dtype=np.dtype(self.dtype)))
        return np.linalg.solve(np.asarray(matrix, dtype=np.dtype(self.dtype)), rhs)

    def _solve_cupy(self, matrix, rhs):
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpsp
            import cupyx.scipy.sparse.linalg as cpspla
        except ImportError as exc:
            raise ImportError("Install cupy to use linear_solver='cupy'") from exc
        if not sp.issparse(matrix):
            x = cp.linalg.solve(cp.asarray(matrix), cp.asarray(rhs))
        else:
            csr = matrix.tocsr()
            a = cpsp.csr_matrix((cp.asarray(csr.data), cp.asarray(csr.indices), cp.asarray(csr.indptr)), shape=csr.shape)
            x = cpspla.spsolve(a, cp.asarray(rhs))
        if self.is_torch:
            return self.asarray(cp.asnumpy(x))
        return cp.asnumpy(x)

    def torch_dtype(self, dtype=None):
        torch = self.torch
        dtype = dtype or self.dtype
        if dtype in ("float64", np.float64):
            return torch.float64
        if dtype in ("float32", np.float32):
            return torch.float32
        if dtype in ("float16", np.float16):
            return torch.float16
        return getattr(torch, str(np.dtype(dtype)), torch.float64)


def get_backend(
    backend: str | ArrayBackend = "numpy",
    *,
    device: str | None = None,
    dtype: str | Any = "float64",
    linear_solver: str = "auto",
) -> ArrayBackend:
    if isinstance(backend, ArrayBackend):
        return backend
    name = backend.lower()
    if name in ("numpy", "np"):
        return ArrayBackend("numpy", device=None, dtype=dtype, linear_solver=linear_solver)
    if name in ("torch", "pytorch"):
        return ArrayBackend("torch", device=device, dtype=dtype, linear_solver=linear_solver)
    raise ValueError(f"Unsupported backend {backend!r}")
