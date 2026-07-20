"""Small datasets for saved PyMRST simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


class SavedStateDataset:
    """Consecutive state-pair dataset backed by a saved ``.npz`` file."""

    def __init__(
        self,
        path: str | Path,
        *,
        input_fields: Sequence[str] = ("pressure", "sW", "sG", "rs"),
        target_fields: Sequence[str] | None = None,
        backend: str = "numpy",
        device: str | None = None,
        dtype: str | Any = "float64",
    ):
        self.path = Path(path)
        self.input_fields = tuple(input_fields)
        self.target_fields = tuple(target_fields or input_fields)
        self.backend = backend
        self.device = device
        self.dtype = dtype
        with np.load(self.path, allow_pickle=False) as data:
            self.data = {key: data[key].copy() for key in data.files}
        self._check_fields()

    def __len__(self) -> int:
        return self.n_steps - 1

    def __getitem__(self, index: int):
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        x = self._stack(self.input_fields, index)
        y = self._stack(self.target_fields, index + 1)
        meta = {
            "step": index,
            "time": self.data.get("time", np.arange(self.n_steps))[index],
            "target_time": self.data.get("time", np.arange(self.n_steps))[index + 1],
        }
        return (
            _convert_array(x, self.backend, self.device, self.dtype),
            _convert_array(y, self.backend, self.device, self.dtype),
            meta,
        )

    @property
    def n_steps(self) -> int:
        return int(np.asarray(self.data[self.input_fields[0]]).shape[0])

    def grid_metadata(self) -> dict[str, np.ndarray]:
        prefixes = ("grid.", "rock.", "well.")
        return {k: v for k, v in self.data.items() if k.startswith(prefixes)}

    def _check_fields(self) -> None:
        for field in self.input_fields + self.target_fields:
            if field not in self.data:
                raise KeyError(f"{field!r} not found in {self.path}")
            arr = np.asarray(self.data[field])
            if arr.ndim != 2:
                raise ValueError(f"{field!r} must be a time-by-cell array")

    def _stack(self, fields: Sequence[str], index: int) -> np.ndarray:
        return np.stack([np.asarray(self.data[name][index], dtype=self.dtype) for name in fields])


class SPE1Dataset(SavedStateDataset):
    """Convenience alias for SPE1 saved-state files."""


def _convert_array(arr: np.ndarray, backend: str, device: str | None, dtype):
    backend = backend.lower()
    if backend in ("numpy", "np"):
        return arr
    if backend == "torch":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install torch to use backend='torch'") from exc
        tensor = torch.as_tensor(arr, device=device)
        return tensor.to(dtype=_torch_dtype(torch, dtype)) if tensor.is_floating_point() else tensor
    raise ValueError(f"Unsupported backend {backend!r}")


def _torch_dtype(torch, dtype):
    if dtype in ("float32", np.float32):
        return torch.float32
    if dtype in ("float64", np.float64):
        return torch.float64
    if dtype in ("float16", np.float16):
        return torch.float16
    return getattr(torch, str(np.dtype(dtype)), torch.float64)
