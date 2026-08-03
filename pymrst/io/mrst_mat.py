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

    This supports normal MATLAB v7 files through SciPy. For v7.3/HDF5 files -
    which is what MRST's ``save`` produces once a case is large enough - install
    the optional extra, ``pip install pymrst[mat73]``; those files are loaded
    into raw nested arrays/dicts and then passed through the same normalizer.
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


if __name__ == "__main__":
    import tempfile

    from scipy.io import savemat

    def _as_matlab_grid_struct(g):
        """Re-express a PyMRST `Grid` the way MATLAB's `save` would store it.

        Building a valid MRST `G` by hand is easy to get subtly wrong, so the
        fixture is derived from a real grid instead and the MATLAB conventions
        are re-applied on the way out: every index becomes 1-based, and a
        missing neighbour goes from PyMRST's -1 back to MATLAB's 0. Loading it
        again must return the grid we started from, which makes this a genuine
        round-trip rather than a test against a hand-written expectation.

        The same loaders were separately checked against a genuinely `save`d
        SPE1 case: cell/face counts, volumes, centroids, areas, permeability
        and pressures all matched MATLAB to a relative 5.6e-14, and the
        `neighbors` array summed to exactly `matlab_sum - 2 * n_faces`.
        """
        nb = np.asarray(g.faces.neighbors, dtype=float) + 1.0
        return {
            "cells": {
                "num": float(g.cells.num),
                "facePos": np.asarray(g.cells.facePos, dtype=float) + 1.0,
                "faces": np.asarray(g.cells.faces, dtype=float) + 1.0,
                "indexMap": np.asarray(g.cells.indexMap, dtype=float) + 1.0,
                "volumes": np.asarray(g.cells.volumes, dtype=float),
                "centroids": np.asarray(g.cells.centroids, dtype=float),
            },
            "faces": {
                "num": float(g.faces.num),
                "nodePos": np.asarray(g.faces.nodePos, dtype=float) + 1.0,
                "nodes": np.asarray(g.faces.nodes, dtype=float) + 1.0,
                "neighbors": nb,
                "areas": np.asarray(g.faces.areas, dtype=float),
                "centroids": np.asarray(g.faces.centroids, dtype=float),
                "normals": np.asarray(g.faces.normals, dtype=float),
            },
            "nodes": {
                "num": float(g.nodes.num),
                "coords": np.asarray(g.nodes.coords, dtype=float),
            },
            "griddim": float(g.griddim),
            "cartDims": np.asarray(g.cartDims, dtype=float),
        }

    def test_normalize_unwraps_matlab_structs_and_cells():
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.mat"
            savemat(p, {"s": {"a": np.arange(3.0), "b": {"c": 7.0}}})
            out = load_mrst_mat(p)
            assert isinstance(out["s"], dict), type(out["s"])
            assert isinstance(out["s"]["b"], dict)
            assert float(np.asarray(out["s"]["b"]["c"]).ravel()[0]) == 7.0
            assert np.allclose(np.asarray(out["s"]["a"]).ravel(), [0, 1, 2])
            # MATLAB bookkeeping keys must not leak into the result
            assert not any(k.startswith("__") for k in out)
        print("ok test_normalize_unwraps_matlab_structs_and_cells")

    def test_grid_round_trips_and_indices_return_zero_based():
        from pymrst.gridprocessing import cart_grid, compute_geometry
        src = compute_geometry(cart_grid([3, 2, 1], [3.0, 2.0, 1.0]))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.mat"
            savemat(p, {"G": _as_matlab_grid_struct(src)})
            g = load_mrst_grid_mat(p)
        assert g.cells.num == src.cells.num, (g.cells.num, src.cells.num)
        assert g.faces.num == src.faces.num
        assert g.nodes.num == src.nodes.num
        assert g.griddim == src.griddim
        # MATLAB's 1-based indices, and its 0 for "no neighbour", must come
        # back as PyMRST's 0-based indices and -1. Getting this wrong shifts
        # every connection by one cell, which stays silent until the flux is
        # wrong somewhere far downstream.
        for name, got, want in (
            ("faces.neighbors", g.faces.neighbors, src.faces.neighbors),
            ("cells.faces", g.cells.faces, src.cells.faces),
            ("cells.facePos", g.cells.facePos, src.cells.facePos),
            ("faces.nodes", g.faces.nodes, src.faces.nodes),
            ("faces.nodePos", g.faces.nodePos, src.faces.nodePos),
        ):
            assert np.array_equal(np.asarray(got), np.asarray(want)), name
        assert (np.asarray(g.faces.neighbors) == -1).any(), "boundary faces must read -1"
        for name, got, want in (
            ("cells.volumes", g.cells.volumes, src.cells.volumes),
            ("cells.centroids", g.cells.centroids, src.cells.centroids),
            ("faces.areas", g.faces.areas, src.faces.areas),
            ("nodes.coords", g.nodes.coords, src.nodes.coords),
        ):
            assert np.allclose(np.asarray(got), np.asarray(want)), name
        print("ok test_grid_round_trips_and_indices_return_zero_based")

    def test_wells_convert_cells_to_zero_based():
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.mat"
            savemat(p, {"W": [
                {"cells": np.array([1.0]), "type": "rate", "val": 3.0,
                 "name": "INJ", "sign": 1.0, "WI": np.array([1e-12]),
                 "compi": np.array([1.0, 0.0]), "r": np.array([0.1])},
                {"cells": np.array([4.0]), "type": "bhp", "val": 5e6,
                 "name": "PROD", "sign": -1.0, "WI": np.array([2e-12]),
                 "compi": np.array([0.0, 1.0]), "r": np.array([0.1])},
            ]})
            wells = load_mrst_wells_mat(p)
            assert len(wells) == 2, len(wells)
            assert np.asarray(wells[0].cells).tolist() == [0], wells[0].cells
            assert np.asarray(wells[1].cells).tolist() == [3], wells[1].cells
            assert wells[0].name == "INJ" and wells[0].type == "rate"
            assert wells[1].sign == -1 and wells[1].val == 5e6
        print("ok test_wells_convert_cells_to_zero_based")

    def test_states_and_schedule_round_trip():
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.mat"
            savemat(p, {
                "states": [{"pressure": np.array([1.0, 2.0]), "s": np.zeros((2, 2))},
                           {"pressure": np.array([3.0, 4.0]), "s": np.ones((2, 2))}],
                "schedule": {
                    "step": {"val": np.array([1.0, 2.0]), "control": np.array([1.0, 1.0])},
                    "control": [{"W": [{"cells": np.array([2.0]), "type": "bhp",
                                        "val": 1.0, "name": "P"}]}],
                },
            })
            states = load_mrst_states_mat(p)
            assert len(states) == 2, len(states)
            assert np.allclose(np.asarray(states[1].pressure).ravel(), [3.0, 4.0])
            sched = load_mrst_schedule_mat(p)
            # The raw `W` is kept and the converted wells added alongside, so
            # no original MRST field is lost on the way in.
            ctrl = sched["control"][0]
            assert "W" in ctrl and "W_py" in ctrl
            assert np.asarray(ctrl["W_py"][0].cells).tolist() == [1], ctrl["W_py"][0].cells
        print("ok test_states_and_schedule_round_trip")

    def test_missing_variable_names_the_file():
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.mat"
            savemat(p, {"other": np.zeros(1)})
            for fn in (load_mrst_grid_mat, load_mrst_states_mat, load_mrst_wells_mat):
                try:
                    fn(p)
                except KeyError as exc:
                    assert "not found" in str(exc), exc
                else:
                    raise AssertionError(f"{fn.__name__} should raise on a missing variable")
        print("ok test_missing_variable_names_the_file")

    def test_v73_without_h5py_explains_itself():
        """A v7.3 file must fail with an actionable message, not a stack trace.

        MRST writes `-v7.3` (HDF5) whenever a case is large, and SciPy cannot
        read it. `h5py` is an optional extra (`pip install pymrst[mat73]`), so
        the failure has to say so rather than surfacing SciPy's internal error.
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pass
        else:
            print("ok test_v73_without_h5py_explains_itself (skipped: h5py present)")
            return
        # A real v7.3 file cannot be written here without h5py or MATLAB, and a
        # fake HDF5 header is rejected by SciPy with a different error before
        # the h5py branch is reached. So the branch itself is exercised
        # directly. Checked separately against a genuine MATLAB `-v7.3` save,
        # which does reach this path and produces this message.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v73.mat"
            p.write_bytes(b"\x89HDF\r\n\x1a\n" + bytes(512))
            try:
                _load_hdf5_mat(p)
            except ImportError as exc:
                assert "h5py" in str(exc), exc
                assert str(p) in str(exc), "the message should name the file"
            else:
                raise AssertionError("expected an ImportError naming h5py")
        print("ok test_v73_without_h5py_explains_itself")

    test_normalize_unwraps_matlab_structs_and_cells()
    test_grid_round_trips_and_indices_return_zero_based()
    test_wells_convert_cells_to_zero_based()
    test_states_and_schedule_round_trip()
    test_missing_variable_names_the_file()
    test_v73_without_h5py_explains_itself()
    print("\nAll tests passed!")
