"""Load MATLAB/MRST ``.mat`` files into PyMRST-friendly objects.

The bridge has two layers:

* ``load_mrst_mat`` / ``normalize_mrst_mat`` preserve everything as plain
  Python dictionaries/lists/arrays so unknown MRST fields are not lost.
* Typed loaders convert common MRST objects into PyMRST objects where the
  mapping is well-defined: grids, state structs, well solution structs,
  schedules, and well arrays.

MATLAB uses one-based indexing; typed converters fix known index fields.
Raw normalized dictionaries keep MATLAB values as saved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from pymrst.gridprocessing.load_mrst_grid_mat import mrst_grid_struct_to_grid
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.params.wells_and_bc import Well


def load_mrst_mat(
    path: str | Path,
    *,
    squeeze: bool = True,
    normalize: bool = True,
) -> dict[str, Any]:
    """Load a MATLAB file and optionally normalize MATLAB structs/cells.

    This supports normal MATLAB v7 files through SciPy. For v7.3/HDF5 files,
    install ``h5py``; those files are loaded into raw nested arrays/dicts and
    then passed through the same normalizer.
    """
    path = Path(path)
    try:
        raw = scipy.io.loadmat(path, squeeze_me=squeeze, struct_as_record=False)
        raw = {k: v for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        raw = _load_hdf5_mat(path)
    return normalize_mrst_mat(raw) if normalize else raw


def normalize_mrst_mat(obj: Any) -> Any:
    """Recursively convert MATLAB structs/cells to dicts/lists/arrays."""
    if _is_mat_struct(obj):
        return {name: normalize_mrst_mat(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, np.void) and obj.dtype.names:
        return {name: normalize_mrst_mat(obj[name]) for name in obj.dtype.names}
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            if obj.shape == ():
                return normalize_mrst_mat(obj.item())
            return [normalize_mrst_mat(v) for v in obj.ravel()]
        if obj.dtype == object:
            if obj.shape == ():
                return normalize_mrst_mat(obj.item())
            return [normalize_mrst_mat(v) for v in obj.ravel()]
        return np.asarray(obj).squeeze()
    if isinstance(obj, dict):
        return {str(k): normalize_mrst_mat(v) for k, v in obj.items()}
    if isinstance(obj, bytes):
        return obj.decode()
    return _scalar_or_value(obj)


def load_mrst_grid_mat(path: str | Path, variable: str = "G"):
    """Load MRST ``G`` from a MATLAB file as a PyMRST ``Grid``."""
    mat = load_mrst_mat(path)
    if variable not in mat:
        raise KeyError(f"{variable!r} not found in {path}")
    return mrst_grid_struct_to_grid(mat[variable])


def load_mrst_states_mat(path: str | Path, variable: str = "states") -> list[State]:
    """Load MRST state structs/cell arrays as PyMRST ``State`` objects."""
    mat = load_mrst_mat(path)
    if variable not in mat:
        raise KeyError(f"{variable!r} not found in {path}")
    return [_state_from_dict(s) for s in _as_list(mat[variable])]


def load_mrst_well_sols_mat(path: str | Path, variable: str = "wellSols") -> list[Any]:
    """Load MRST ``wellSols`` while preserving nested well fields.

    MRST typically stores well solutions as a cell array over timesteps, each
    containing a struct array over wells. The return value mirrors that shape:
    a list of timesteps, where each timestep is a list of normalized dicts.
    """
    mat = load_mrst_mat(path)
    if variable not in mat:
        raise KeyError(f"{variable!r} not found in {path}")
    return [_well_sol_step(step) for step in _as_list(mat[variable])]


def load_mrst_schedule_mat(path: str | Path, variable: str = "schedule") -> dict[str, Any]:
    """Load MRST ``schedule`` as a normalized dictionary.

    Known well arrays inside controls are converted to PyMRST ``Well`` objects
    and stored as ``W_py`` alongside the raw ``W`` dictionary/list so no original
    fields are lost.
    """
    mat = load_mrst_mat(path)
    if variable not in mat:
        raise KeyError(f"{variable!r} not found in {path}")
    sched = _ensure_dict(mat[variable])
    controls = sched.get("control")
    if controls is not None:
        new_controls = []
        for ctrl in _as_list(controls):
            c = _ensure_dict(ctrl)
            if "W" in c:
                c = dict(c)
                c["W_py"] = _wells_from_any(c["W"])
            new_controls.append(c)
        sched = dict(sched)
        sched["control"] = new_controls
    return sched


def load_mrst_wells_mat(path: str | Path, variable: str = "W") -> list[Well]:
    """Load MRST well array ``W`` as PyMRST ``Well`` objects."""
    mat = load_mrst_mat(path)
    if variable not in mat:
        raise KeyError(f"{variable!r} not found in {path}")
    return _wells_from_any(mat[variable])


def _state_from_dict(value: Any) -> State:
    d = _ensure_dict(value)
    state = State()
    for key, val in d.items():
        state[key] = _convert_state_value(key, val)
    return state


def _convert_state_value(key: str, val: Any) -> Any:
    if key in {"wellSol", "wellSols"}:
        return _well_sol_step(val)
    if isinstance(val, dict):
        return {k: _convert_state_value(k, v) for k, v in val.items()}
    if isinstance(val, list):
        return [_convert_state_value(key, v) for v in val]
    return np.asarray(val).copy() if isinstance(val, np.ndarray) else val


def _well_sol_step(value: Any) -> list[dict[str, Any]]:
    return [_ensure_dict(v) for v in _as_list(value)]


def _wells_from_any(value: Any) -> list[Well]:
    return [_well_from_dict(_ensure_dict(w), i) for i, w in enumerate(_as_list(value))]


def _well_from_dict(w: dict[str, Any], index: int = 0) -> Well:
    cells = _zero_based_array(_first_present(w, "cells", "cell", default=[]))
    if cells.size == 0:
        # Some MRST well structs store completions in a nested cstatus/cells
        # layout. Preserve the raw fields elsewhere, but create a harmless empty
        # well rather than failing to import the file.
        cells = np.zeros(0, dtype=np.int64)
    wi = _array(_first_present(w, "WI", "wi", default=np.zeros(cells.size)), dtype=float)
    if wi.size == 1 and cells.size > 1:
        wi = np.full(cells.size, float(wi.item()))
    compi = _array(_first_present(w, "compi", "components", default=np.ones(1)), dtype=float)
    d_z = _array(_first_present(w, "dZ", "dz", default=np.zeros(cells.size)), dtype=float)
    if d_z.size == 1 and cells.size > 1:
        d_z = np.full(cells.size, float(d_z.item()))
    radius = _array(_first_present(w, "r", "radius", default=np.zeros(max(cells.size, 1))), dtype=float)
    if radius.size == 1 and cells.size > 1:
        radius = np.full(cells.size, float(radius.item()))
    name = str(_first_present(w, "name", default=f"W{index + 1}"))
    return Well(
        cells=cells,
        type=str(_first_present(w, "type", default="unknown")),
        val=float(_scalar(_first_present(w, "val", default=0.0))),
        r=radius,
        dir=str(_first_present(w, "dir", default="z")),
        WI=wi,
        dZ=d_z,
        name=name,
        compi=compi,
        refDepth=float(_scalar(_first_present(w, "refDepth", "ref_depth", default=0.0))),
        sign=int(_scalar(_first_present(w, "sign", default=0))),
        status=bool(_scalar(_first_present(w, "status", default=True))),
        components=_optional_array(_first_present(w, "components", default=None), dtype=float),
        thp=_optional_scalar(_first_present(w, "thp", "THP", default=None)),
        lims=_ensure_dict(w["lims"]) if isinstance(w.get("lims"), dict) else w.get("lims"),
        multisegment=w.get("multisegment"),
        vfp_table=w.get("vfp_table", w.get("VFPTable")),
        vfp_index=_optional_int(_first_present(w, "vfp_index", "VFPIndex", default=None)),
    )


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return normalize_mrst_mat(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [normalize_mrst_mat(v) for v in value.ravel()]
    if isinstance(value, np.ndarray) and value.dtype.names:
        return [normalize_mrst_mat(v) for v in value.ravel()]
    return [value]


def _first_present(d: dict[str, Any], *names: str, default=None):
    for name in names:
        if name in d:
            return d[name]
    return default


def _array(value: Any, dtype=float) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=dtype)
    return np.atleast_1d(np.asarray(value, dtype=dtype)).squeeze()


def _optional_array(value: Any, dtype=float):
    if value is None:
        return None
    arr = _array(value, dtype=dtype)
    return None if arr.size == 0 else arr


def _zero_based_array(value: Any) -> np.ndarray:
    arr = _array(value, dtype=np.int64).ravel()
    if arr.size == 0:
        return arr
    return np.where(arr > 0, arr - 1, arr).astype(np.int64)


def _optional_scalar(value: Any):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return float(arr.squeeze())


def _optional_int(value: Any):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    val = float(arr.squeeze())
    if not np.isfinite(val):
        return None
    return int(val)


def _scalar(value: Any):
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return arr.squeeze().item() if hasattr(arr.squeeze(), "item") else arr.squeeze()


def _scalar_or_value(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_mat_struct(obj: Any) -> bool:
    return hasattr(obj, "_fieldnames")


def _load_hdf5_mat(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            f"{path} appears to be a MATLAB v7.3/HDF5 file. Install h5py to load it."
        ) from exc

    def visit(obj):
        if isinstance(obj, h5py.Dataset):
            arr = np.array(obj)
            if arr.dtype.kind in {"S", "U"}:
                return arr.astype(str)
            return arr.T if arr.ndim > 1 else arr
        if isinstance(obj, h5py.Group):
            return {k: visit(v) for k, v in obj.items()}
        return obj

    with h5py.File(path, "r") as f:
        return {k: visit(v) for k, v in f.items() if not k.startswith("#")}


__all__ = [
    "load_mrst_grid_mat",
    "load_mrst_mat",
    "load_mrst_schedule_mat",
    "load_mrst_states_mat",
    "load_mrst_well_sols_mat",
    "load_mrst_wells_mat",
    "normalize_mrst_mat",
]
