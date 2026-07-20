"""Tensor export helpers for grids, rocks, and simulator states."""

from __future__ import annotations

from typing import Any

import numpy as np


def grid_to_tensors(
    g,
    *,
    backend: str = "numpy",
    device: str | None = None,
    dtype: str | Any = "float64",
) -> dict[str, Any]:
    """Convert a PyMRST grid to array/tensor dictionaries.

    ``float64`` is the default to preserve MRST parity. Use ``dtype="float32"``
    explicitly for ML/GPU speed.
    """
    data = {
        "cells.num": np.asarray(g.cells.num, dtype=np.int64),
        "cells.facePos": np.asarray(g.cells.facePos, dtype=np.int64),
        "cells.faces": np.asarray(g.cells.faces, dtype=np.int64),
        "cells.indexMap": np.asarray(g.cells.indexMap, dtype=np.int64),
        "faces.num": np.asarray(g.faces.num, dtype=np.int64),
        "faces.nodePos": np.asarray(g.faces.nodePos, dtype=np.int64),
        "faces.nodes": np.asarray(g.faces.nodes, dtype=np.int64),
        "faces.neighbors": np.asarray(g.faces.neighbors, dtype=np.int64),
        "nodes.num": np.asarray(g.nodes.num, dtype=np.int64),
        "nodes.coords": _float_array(g.nodes.coords, dtype),
        "griddim": np.asarray(g.griddim, dtype=np.int64),
    }
    _maybe(data, "cells.tags", g.cells.tags, np.int64)
    _maybe(data, "cells.volumes", g.cells.volumes, dtype)
    _maybe(data, "cells.centroids", g.cells.centroids, dtype)
    _maybe(data, "cells.nodes", g.cells.nodes, np.int64)
    _maybe(data, "cells.nodePos", g.cells.nodePos, np.int64)
    _maybe(data, "faces.tag", g.faces.tag, np.int64)
    _maybe(data, "faces.areas", g.faces.areas, dtype)
    _maybe(data, "faces.normals", g.faces.normals, dtype)
    _maybe(data, "faces.centroids", g.faces.centroids, dtype)
    if g.cartDims is not None:
        data["cartDims"] = np.asarray(g.cartDims, dtype=np.int64)
    return _convert_backend(data, backend=backend, device=device, dtype=dtype)


def rock_to_tensors(
    rock,
    *,
    backend: str = "numpy",
    device: str | None = None,
    dtype: str | Any = "float64",
) -> dict[str, Any]:
    data = {
        "perm": _float_array(rock.perm, dtype),
        "poro": _float_array(rock.poro, dtype),
    }
    _maybe(data, "ntg", getattr(rock, "ntg", None), dtype)
    return _convert_backend(data, backend=backend, device=device, dtype=dtype)


def state_to_tensors(
    state,
    *,
    backend: str = "numpy",
    device: str | None = None,
    dtype: str | Any = "float64",
) -> dict[str, Any]:
    """Convert dict-like or object-like state fields to tensors."""
    out = {}
    keys = state.keys() if hasattr(state, "keys") else dir(state)
    for key in keys:
        if key.startswith("_"):
            continue
        try:
            value = state[key] if hasattr(state, "__getitem__") else getattr(state, key)
        except Exception:
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in "biufc" and arr.ndim <= 2:
            out[key] = _float_array(arr, dtype) if arr.dtype.kind in "fc" else arr.astype(np.int64)
    return _convert_backend(out, backend=backend, device=device, dtype=dtype)


def _float_array(value, dtype) -> np.ndarray:
    return np.asarray(value, dtype=np.dtype(dtype))


def _maybe(data: dict[str, Any], key: str, value, dtype) -> None:
    if value is not None:
        arr = np.asarray(value)
        data[key] = arr.astype(np.dtype(dtype) if arr.dtype.kind in "fc" else dtype)


def _convert_backend(
    data: dict[str, np.ndarray],
    *,
    backend: str,
    device: str | None,
    dtype,
) -> dict[str, Any]:
    backend = backend.lower()
    if backend in ("numpy", "np"):
        return data
    if backend == "torch":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install torch to use backend='torch'") from exc
        torch_dtype = _torch_dtype(torch, dtype)
        out = {}
        for key, arr in data.items():
            tensor = torch.as_tensor(arr, device=device)
            if tensor.is_floating_point() or tensor.is_complex():
                tensor = tensor.to(dtype=torch_dtype)
            out[key] = tensor
        return out
    raise ValueError(f"Unsupported tensor backend {backend!r}")


def _torch_dtype(torch, dtype):
    if dtype in ("float32", np.float32):
        return torch.float32
    if dtype in ("float64", np.float64):
        return torch.float64
    if dtype in ("float16", np.float16):
        return torch.float16
    return getattr(torch, str(np.dtype(dtype)), torch.float64)
