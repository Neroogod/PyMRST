"""Restrict rock properties to a subset of cells (MRST ``compressRock.m``).

`processGRDECL` drops inactive cells, so a rock built from the full logical
Cartesian box has more rows than the resulting grid has cells. `compress_rock`
takes `G.cells.indexMap` and keeps only those rows, which is the standard step
between `grdecl2rock` and a usable model.

MRST walks every field of the rock struct and subsets any numeric array whose
first dimension matches the original cell count, recursing into nested structs.
The port keeps that rule rather than naming `perm`, `poro` and `ntg` explicitly,
so a rock carrying extra per-cell data survives the round trip intact.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np

from pymrst.params.rock.rock import Rock


def compress_rock(rock: Any, active) -> Any:
    """Keep only the rows of `rock` named by `active`.

    `active` is a 0-based cell index array, normally `G.cells.indexMap` from a
    grid built by `process_grdecl`. Fields whose first dimension does not match
    the original cell count are passed through untouched, which is what lets a
    scalar or a per-face array coexist with per-cell data.
    """
    active = np.asarray(active, dtype=np.int64).ravel()
    n_cells = _cell_count(rock)
    if n_cells is None:
        raise TypeError("rock must expose a per-cell 'poro' to compress against")
    if active.size and (active.min() < 0 or active.max() >= n_cells):
        raise IndexError(
            f"active indices must lie in [0, {n_cells}); "
            f"got [{active.min()}, {active.max()}]")
    return _compress(rock, n_cells, active)


def _cell_count(rock: Any):
    poro = getattr(rock, "poro", None)
    if poro is None and isinstance(rock, dict):
        poro = rock.get("poro")
    if poro is None:
        return None
    return int(np.asarray(poro).shape[0])


def _compress(obj: Any, n_cells: int, active: np.ndarray) -> Any:
    if isinstance(obj, dict):
        return {k: _compress(v, n_cells, active) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = copy.copy(obj)
        for f in dataclasses.fields(obj):
            setattr(out, f.name, _compress(getattr(obj, f.name), n_cells, active))
        return out
    return _subset(obj, n_cells, active)


def _subset(value: Any, n_cells: int, active: np.ndarray) -> Any:
    if value is None:
        return None
    arr = np.asarray(value)
    # Only per-cell data is subset. A scalar, or an array indexed by something
    # other than cells, keeps its full extent - matching MRST's `size(d, 1) ==
    # nc` test rather than compressing anything that happens to be numeric.
    if arr.ndim == 0 or arr.shape[0] != n_cells or not np.issubdtype(arr.dtype, np.number):
        return value
    return arr[active, ...].copy()


__all__ = ["compress_rock"]


if __name__ == "__main__":
    def test_subsets_per_cell_fields():
        rock = Rock(perm=np.arange(12.0).reshape(6, 2),
                    poro=np.linspace(0.1, 0.6, 6),
                    ntg=np.full(6, 0.9))
        out = compress_rock(rock, [0, 2, 4])
        assert out.poro.shape == (3,), out.poro.shape
        assert np.allclose(out.poro, [0.1, 0.3, 0.5])
        assert out.perm.shape == (3, 2), out.perm.shape
        assert np.allclose(out.perm, [[0, 1], [4, 5], [8, 9]])
        assert np.allclose(out.ntg, 0.9)
        print("ok test_subsets_per_cell_fields")

    def test_original_is_untouched():
        """Compression returns a new rock; the caller's is unchanged."""
        rock = Rock(perm=np.arange(6.0).reshape(3, 2), poro=np.array([.1, .2, .3]),
                    ntg=None)
        out = compress_rock(rock, [1])
        assert rock.poro.shape == (3,), "input was mutated"
        assert out.poro.shape == (1,)
        out.poro[0] = 99.0
        assert rock.poro[1] == 0.2, "result aliases the input"
        print("ok test_original_is_untouched")

    def test_none_and_scalar_fields_pass_through():
        rock = Rock(perm=np.arange(3.0), poro=np.array([.1, .2, .3]), ntg=None)
        out = compress_rock(rock, [0, 1])
        assert out.ntg is None
        print("ok test_none_and_scalar_fields_pass_through")

    def test_dict_rock_and_nested_structs():
        """MRST recurses into nested structs, so a dict rock must work too."""
        rock = {"poro": np.array([.1, .2, .3]),
                "perm": np.arange(9.0).reshape(3, 3),
                "regions": {"rocknum": np.array([1, 2, 3])},
                "label": "unchanged"}
        out = compress_rock(rock, [2, 0])
        assert np.allclose(out["poro"], [.3, .1])
        assert np.allclose(out["regions"]["rocknum"], [3, 1]), out["regions"]
        assert out["label"] == "unchanged"
        print("ok test_dict_rock_and_nested_structs")

    def test_arrays_not_indexed_by_cells_are_left_alone():
        """A field whose first dimension is not the cell count is not per-cell.

        MRST tests `size(d, 1) == nc` for exactly this reason: compressing an
        array indexed by faces or by PVT region would silently corrupt it.
        """
        rock = {"poro": np.array([.1, .2, .3]),
                "face_data": np.arange(7.0)}
        out = compress_rock(rock, [0, 1])
        assert out["face_data"].shape == (7,), out["face_data"].shape
        print("ok test_arrays_not_indexed_by_cells_are_left_alone")

    def test_out_of_range_indices_are_rejected():
        rock = Rock(perm=np.arange(3.0), poro=np.array([.1, .2, .3]), ntg=None)
        try:
            compress_rock(rock, [0, 5])
        except IndexError as exc:
            assert "must lie in" in str(exc), exc
        else:
            raise AssertionError("expected IndexError for an out-of-range index")
        print("ok test_out_of_range_indices_are_rejected")

    test_subsets_per_cell_fields()
    test_original_is_untouched()
    test_none_and_scalar_fields_pass_through()
    test_dict_rock_and_nested_structs()
    test_arrays_not_indexed_by_cells_are_left_alone()
    test_out_of_range_indices_are_rejected()
    print("\nAll tests passed!")
