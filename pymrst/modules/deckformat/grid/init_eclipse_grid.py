"""Build a grid from ECLIPSE deck keywords.

Port of MRST ``initEclipseGrid.m``.

A deck can describe its geometry in several mutually exclusive ways, and this
function picks the right constructor by looking at which keywords are present:

===========================  ==================================================
Keywords in ``GRID``         Construction
===========================  ==================================================
``COORD`` and ``ZCORN``      corner-point grid through ``process_grdecl``
``DXV``/``DYV``/``DZV``      tensor grid from the three spacing vectors
``DX``/``DY``/``DZ``         block-centred grid, collapsed to a tensor grid
``COORDX``/``COORDY``/...    not supported, as in MRST
===========================  ==================================================

After construction three optional steps follow, in MRST's order: non-neighbour
connections are merged, `MAPAXES` is applied when asked for, and cells with no
pore volume are removed when asked for.

The block-centred branch deserves a note. `DX`, `DY` and `DZ` give a value per
cell, which in general describes a grid that is not a tensor product at all.
MRST supports only the tensor case and proves it by run-length encoding each
array across the two directions it should not vary in, asserting the run covers
the whole extent. The port keeps those checks and reports which direction
failed, since a deck that violates them is not merely unsupported, it is
describing a different geometry from the one the caller expects.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.gridprocessing.grid_structure import extract_subgrid
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.gridprocessing.process_nnc import process_nnc
from pymrst.gridprocessing.tensor_grid import tensor_grid
from pymrst.modules.deckformat.map_axes import map_axes


def init_eclipse_grid(deck: dict, *, map_axes_enabled: bool = False,
                      remove_zero_pv: bool = False, **kwargs) -> Any:
    """Construct the grid a deck describes.

    ``map_axes_enabled`` and ``remove_zero_pv`` mirror MRST's ``'mapAxes'`` and
    ``'removeZeroPV'`` options, both off by default. Remaining keyword
    arguments are forwarded to ``process_grdecl`` for the corner-point branch.
    """
    grid_sec = _section(deck, "GRID")
    runspec = _section(deck, "RUNSPEC")

    if _has(grid_sec, "COORD", "ZCORN") and _has(runspec, "DIMENS"):
        g = process_grdecl(grid_sec, repair_zcorn=True, **kwargs)
    elif _has(grid_sec, "DXV", "DYV", "DZV"):
        g = _tensor_from_vectors(grid_sec, runspec)
    elif _has(grid_sec, "COORDX", "COORDY", "COORDZ"):
        raise NotImplementedError(
            "the COORDX/COORDY/COORDZ grid type is not supported, as in MRST")
    elif _has(grid_sec, "DX", "DY", "DZ"):
        g = _tensor_from_block_centred(grid_sec, runspec)
    else:
        raise NotImplementedError(
            "deck GRID section describes no supported grid type; expected "
            "COORD/ZCORN, DXV/DYV/DZV, or DX/DY/DZ")

    if _get(grid_sec, "NNC") is not None:
        g = process_nnc(g, np.asarray(_get(grid_sec, "NNC")))

    if map_axes_enabled and _get(grid_sec, "MAPAXES") is not None:
        for grid in _as_list(g):
            coords = np.asarray(grid.nodes.coords, dtype=float).copy()
            coords[:, 0:2] = map_axes(coords[:, 0:2], _get(grid_sec, "MAPAXES"))
            grid.nodes.coords = coords

    if remove_zero_pv:
        g = _drop_zero_pore_volume(g, grid_sec)
    return g


def _tensor_from_vectors(grid_sec, runspec):
    """`DXV`/`DYV`/`DZV`: one spacing vector per direction."""
    dxv = np.asarray(_get(grid_sec, "DXV"), dtype=float).ravel()
    dyv = np.asarray(_get(grid_sec, "DYV"), dtype=float).ravel()
    dzv = np.asarray(_get(grid_sec, "DZV"), dtype=float).ravel()
    depthz = _get(grid_sec, "DEPTHZ")
    if depthz is None:
        depthz = np.zeros((dxv.size + 1) * (dyv.size + 1))
    g = tensor_grid(_edges(dxv), _edges(dyv), _edges(dzv),
                    depthz=_depthz_grid(depthz, dxv.size, dyv.size))
    return _apply_actnum(g, grid_sec)


def _depthz_grid(depthz, nx: int, ny: int) -> np.ndarray:
    """`DEPTHZ` as the `(nx+1, ny+1)` array `tensor_grid` expects.

    ECLIPSE and MRST both carry this as a flat vector over the node columns.
    PyMRST's `tensor_grid` wants it shaped, so the reshape happens here rather
    than at each call site. Fortran order matches the deck's column ordering.
    """
    arr = np.asarray(depthz, dtype=float)
    if arr.shape == (nx + 1, ny + 1):
        return arr
    if arr.size != (nx + 1) * (ny + 1):
        raise ValueError(
            f"DEPTHZ must hold {(nx + 1) * (ny + 1)} values for a "
            f"{nx} x {ny} grid, got {arr.size}")
    return arr.reshape((nx + 1, ny + 1), order="F")


def _tensor_from_block_centred(grid_sec, runspec):
    """`DX`/`DY`/`DZ`: a value per cell, valid only when it is a tensor grid."""
    dims = np.asarray(_get(runspec, "DIMENS"), dtype=int).ravel()
    nx, ny, nz = (int(v) for v in dims[:3])

    dxv = _collapse(_deltas(grid_sec, dims, 0), keep=0, dims=(nx, ny, nz))
    dyv = _collapse(_deltas(grid_sec, dims, 1), keep=1, dims=(nx, ny, nz))
    dzv = _collapse(_deltas(grid_sec, dims, 2), keep=2, dims=(nx, ny, nz))

    tops = _get(grid_sec, "TOPS")
    if tops is not None:
        tops = np.asarray(tops, dtype=float).ravel()
        # MRST supports block-centred grids only for a flat top surface,
        # because a varying TOPS makes the grid non-tensor in z.
        if not np.allclose(tops[:nx * ny], tops[0]):
            raise NotImplementedError(
                "block-centred grids are supported only for constant TOPS")
        depthz = np.full((nx + 1) * (ny + 1), float(tops[0]))
    else:
        depthz = np.zeros((nx + 1) * (ny + 1))

    g = tensor_grid(_edges(dxv), _edges(dyv), _edges(dzv),
                    depthz=_depthz_grid(depthz, nx, ny))
    return _apply_actnum(g, grid_sec)


def _deltas(grid_sec, dims, index: int) -> np.ndarray:
    """Per-cell spacing for one direction, from `D?V` or `D?`."""
    name = "XYZ"[index]
    vec = _get(grid_sec, f"D{name}V")
    dims = [int(d) for d in np.asarray(dims).ravel()[:3]]
    if vec is not None:
        shape = [1, 1, 1]
        shape[index] = -1
        reps = list(dims)
        reps[index] = 1
        return np.tile(np.asarray(vec, dtype=float).reshape(shape), reps)
    return np.asarray(_get(grid_sec, f"D{name}"),
                      dtype=float).reshape(dims, order="F")


def _collapse(delta: np.ndarray, *, keep: int, dims) -> np.ndarray:
    """Reduce a per-cell spacing array to the one vector it must be.

    A tensor grid's `dx` varies only along x, so collapsing along the other two
    axes must be exact. Anything else describes a grid MRST does not build, and
    saying which direction failed is more useful than a bare assertion.
    """
    axes = [a for a in (0, 1, 2) if a != keep]
    for axis in axes:
        first = np.take(delta, 0, axis=axis)
        if not np.allclose(delta, np.expand_dims(first, axis)):
            raise NotImplementedError(
                f"only tensor grids are supported: D{'XYZ'[keep]} varies along "
                f"{'XYZ'[axis]}")
    out = delta
    for axis in sorted(axes, reverse=True):
        out = np.take(out, 0, axis=axis)
    return np.asarray(out, dtype=float).ravel()


def _edges(spacing: np.ndarray) -> np.ndarray:
    """Cell spacings to node coordinates: `[0, cumsum(spacing)]`."""
    return np.concatenate([[0.0], np.cumsum(np.asarray(spacing, dtype=float))])


def _apply_actnum(g, grid_sec):
    actnum = _get(grid_sec, "ACTNUM")
    if actnum is None:
        return g
    active = np.asarray(actnum).ravel() > 0
    return extract_subgrid(g, np.flatnonzero(active))


def _drop_zero_pore_volume(g, grid_sec):
    """Remove cells that cannot hold fluid.

    A cell is kept only if every pore-volume-related keyword present in the
    deck is positive there. MRST builds the mask over the full logical box and
    then indexes it with each grid's `indexMap`, which matters because the grid
    may already have been restricted by `ACTNUM`.
    """
    mask = None
    for key in ("PORO", "PORV", "NTG", "MULTPV"):
        value = _get(grid_sec, key)
        if value is None:
            continue
        positive = np.asarray(value, dtype=float).ravel() > 0.0
        mask = positive if mask is None else (mask & positive)
    if mask is None:
        return g

    out = []
    for grid in _as_list(g):
        index_map = np.asarray(grid.cells.indexMap, dtype=np.int64)
        out.append(extract_subgrid(grid, np.flatnonzero(mask[index_map])))
    return out[0] if len(out) == 1 else out


def _section(deck, name):
    if hasattr(deck, name):
        return getattr(deck, name)
    if isinstance(deck, dict):
        for key in (name, name.lower()):
            if key in deck:
                return deck[key]
    return {}


def _get(section, key):
    if section is None:
        return None
    for k in (key, key.lower()):
        if isinstance(section, dict) and k in section:
            return section[k]
        if hasattr(section, k):
            return getattr(section, k)
    return None


def _has(section, *keys) -> bool:
    return all(_get(section, k) is not None for k in keys)


def _as_list(g):
    return g if isinstance(g, (list, tuple)) else [g]


__all__ = ["init_eclipse_grid"]


if __name__ == "__main__":
    from pymrst.gridprocessing import compute_geometry

    def _dxv_deck(nx=3, ny=2, nz=2, **grid):
        base = {"DXV": np.full(nx, 10.0), "DYV": np.full(ny, 20.0),
                "DZV": np.full(nz, 5.0)}
        base.update(grid)
        return {"GRID": base, "RUNSPEC": {"DIMENS": np.array([nx, ny, nz])}}

    def test_dxv_branch_builds_a_tensor_grid():
        g = compute_geometry(init_eclipse_grid(_dxv_deck()))
        assert g.cells.num == 12, g.cells.num
        assert np.isclose(np.sum(g.cells.volumes), 30.0 * 40.0 * 10.0), (
            np.sum(g.cells.volumes))
        assert np.allclose(g.cells.volumes, 10.0 * 20.0 * 5.0)
        print("ok test_dxv_branch_builds_a_tensor_grid")

    def test_actnum_removes_inactive_cells():
        actnum = np.ones(12, dtype=int)
        actnum[[0, 5]] = 0
        g = compute_geometry(init_eclipse_grid(_dxv_deck(ACTNUM=actnum)))
        assert g.cells.num == 10, g.cells.num
        assert np.array_equal(np.asarray(g.cells.indexMap),
                              np.flatnonzero(actnum > 0))
        print("ok test_actnum_removes_inactive_cells")

    def test_block_centred_branch_matches_the_vector_branch():
        """`DX`/`DY`/`DZ` and `DXV`/`DYV`/`DZV` describe the same grid here.

        The block-centred form gives one value per cell; when those values are
        constant along the other two directions it collapses to exactly the
        vector form, so the two grids must agree cell for cell.
        """
        nx, ny, nz = 3, 2, 2
        deck = {"GRID": {"DX": np.full(nx * ny * nz, 10.0),
                         "DY": np.full(nx * ny * nz, 20.0),
                         "DZ": np.full(nx * ny * nz, 5.0)},
                "RUNSPEC": {"DIMENS": np.array([nx, ny, nz])}}
        a = compute_geometry(init_eclipse_grid(deck))
        b = compute_geometry(init_eclipse_grid(_dxv_deck(nx, ny, nz)))
        assert a.cells.num == b.cells.num
        assert np.allclose(a.cells.volumes, b.cells.volumes)
        assert np.allclose(a.nodes.coords, b.nodes.coords)
        print("ok test_block_centred_branch_matches_the_vector_branch")

    def test_non_tensor_block_centred_grid_is_rejected_by_direction():
        """A `DX` that varies along y is not a tensor grid, and the error says so."""
        nx, ny, nz = 3, 2, 2
        dx = np.full((nx, ny, nz), 10.0)
        dx[:, 1, :] = 11.0
        deck = {"GRID": {"DX": dx.ravel(order="F"),
                         "DY": np.full(nx * ny * nz, 20.0),
                         "DZ": np.full(nx * ny * nz, 5.0)},
                "RUNSPEC": {"DIMENS": np.array([nx, ny, nz])}}
        try:
            init_eclipse_grid(deck)
        except NotImplementedError as exc:
            assert "varies along Y" in str(exc), exc
        else:
            raise AssertionError("expected a non-tensor DX to be rejected")
        print("ok test_non_tensor_block_centred_grid_is_rejected_by_direction")

    def test_varying_tops_is_rejected():
        nx, ny, nz = 3, 2, 2
        tops = np.zeros(nx * ny); tops[2] = 5.0
        deck = _dxv_deck(nx, ny, nz)
        deck["GRID"]["DX"] = np.full(nx * ny * nz, 10.0)
        deck["GRID"]["DY"] = np.full(nx * ny * nz, 20.0)
        deck["GRID"]["DZ"] = np.full(nx * ny * nz, 5.0)
        for k in ("DXV", "DYV", "DZV"):
            deck["GRID"].pop(k)
        deck["GRID"]["TOPS"] = tops
        try:
            init_eclipse_grid(deck)
        except NotImplementedError as exc:
            assert "TOPS" in str(exc), exc
        else:
            raise AssertionError("expected varying TOPS to be rejected")
        print("ok test_varying_tops_is_rejected")

    def test_constant_tops_shifts_the_grid_down():
        nx, ny, nz = 3, 2, 2
        deck = {"GRID": {"DX": np.full(nx * ny * nz, 10.0),
                         "DY": np.full(nx * ny * nz, 20.0),
                         "DZ": np.full(nx * ny * nz, 5.0),
                         "TOPS": np.full(nx * ny, 1000.0)},
                "RUNSPEC": {"DIMENS": np.array([nx, ny, nz])}}
        g = compute_geometry(init_eclipse_grid(deck))
        z = np.asarray(g.nodes.coords, dtype=float)[:, 2]
        assert np.isclose(z.min(), 1000.0), z.min()
        print("ok test_constant_tops_shifts_the_grid_down")

    def test_remove_zero_pore_volume():
        """Cells with zero porosity are dropped only when asked for."""
        poro = np.full(12, 0.3)
        poro[[1, 7]] = 0.0
        deck = _dxv_deck(PORO=poro)
        kept = init_eclipse_grid(deck)
        assert kept.cells.num == 12, kept.cells.num
        dropped = init_eclipse_grid(deck, remove_zero_pv=True)
        assert dropped.cells.num == 10, dropped.cells.num
        assert np.array_equal(np.asarray(dropped.cells.indexMap),
                              np.flatnonzero(poro > 0.0))
        print("ok test_remove_zero_pore_volume")

    def test_unsupported_grid_types_are_named():
        try:
            init_eclipse_grid({"GRID": {"COORDX": [0], "COORDY": [0],
                                        "COORDZ": [0]}, "RUNSPEC": {}})
        except NotImplementedError as exc:
            assert "COORDX" in str(exc), exc
        else:
            raise AssertionError("expected COORDX/Y/Z to be rejected")
        try:
            init_eclipse_grid({"GRID": {}, "RUNSPEC": {}})
        except NotImplementedError as exc:
            assert "no supported grid type" in str(exc), exc
        else:
            raise AssertionError("expected an empty GRID section to be rejected")
        print("ok test_unsupported_grid_types_are_named")

    def test_map_axes_is_opt_in():
        """Off by default, as in MRST, and applied only when requested."""
        # Right-handed frame, so the mirroring warning does not fire.
        ma = [100.0, 201.0, 100.0, 200.0, 101.0, 200.0]
        deck = _dxv_deck(MAPAXES=ma)
        plain = init_eclipse_grid(deck)
        moved = init_eclipse_grid(deck, map_axes_enabled=True)
        a = np.asarray(plain.nodes.coords, dtype=float)
        b = np.asarray(moved.nodes.coords, dtype=float)
        assert not np.allclose(a[:, :2], b[:, :2]), "mapAxes had no effect"
        assert np.allclose(a[:, 2], b[:, 2]), "mapAxes touched z"
        print("ok test_map_axes_is_opt_in")

    test_dxv_branch_builds_a_tensor_grid()
    test_actnum_removes_inactive_cells()
    test_block_centred_branch_matches_the_vector_branch()
    test_non_tensor_block_centred_grid_is_rejected_by_direction()
    test_varying_tops_is_rejected()
    test_constant_tops_shifts_the_grid_down()
    test_remove_zero_pore_volume()
    test_unsupported_grid_types_are_named()
    test_map_axes_is_opt_in()
    print("\nAll tests passed!")
