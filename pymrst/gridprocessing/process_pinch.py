"""Vertical NNCs across inactive layers (MRST ``processPINCH``)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from pymrst.gridprocessing.grid_structure import BOTTOM, TOP


def _pinch_options(raw) -> tuple[float, str, str]:
    values = np.asarray(raw, dtype=object).ravel().tolist()
    if not values:
        raise ValueError("PINCH cannot be empty")
    threshold = float(values[0])
    trans_option = str(values[3] if len(values) > 3 else "TOPBOT")
    mult_option = str(values[4] if len(values) > 4 else "TOP")
    trans_option = trans_option.strip().strip("'\"").upper()
    mult_option = mult_option.strip().strip("'\"").upper()
    # MRST warns and resets unsupported transmissibility modes to TOPBOT.
    if trans_option != "TOPBOT":
        trans_option = "TOPBOT"
    if mult_option not in {"TOP", "ALL"}:
        raise ValueError("PINCH MULTZ option must be 'TOP' or 'ALL'")
    return threshold, trans_option, mult_option


def _cell_face(grid, cell: int, direction: int) -> int:
    lo, hi = grid.cells.facePos[cell:cell + 2]
    tags = np.asarray(grid.cells.tags[lo:hi], dtype=np.int64)
    faces = np.asarray(grid.cells.faces[lo:hi], dtype=np.int64)
    selected = faces[tags == direction]
    if selected.size != 1:
        name = "TOP" if direction == TOP else "BOTTOM"
        raise ValueError(f"cell {cell} must have exactly one {name} contact")
    return int(selected[0])


def _gap_thickness(
    zcorn: np.ndarray,
    dims: tuple[int, int, int],
    top_global: int,
    bottom_global: int,
) -> float:
    nx, ny, _nz = dims
    ti = top_global % nx
    tj = (top_global // nx) % ny
    tk = top_global // (nx * ny)
    bi = bottom_global % nx
    bj = (bottom_global // nx) % ny
    bk = bottom_global // (nx * ny)
    if (ti, tj) != (bi, bj):
        raise ValueError("pinch connection cells must be in one column")
    z = np.asarray(zcorn, dtype=float).reshape(
        (2 * nx, 2 * ny, 2 * dims[2]), order="F"
    )
    upper_bottom = z[
        2 * ti:2 * ti + 2,
        2 * tj:2 * tj + 2,
        2 * tk + 1,
    ]
    lower_top = z[
        2 * bi:2 * bi + 2,
        2 * bj:2 * bj + 2,
        2 * bk,
    ]
    return float(np.max(lower_top - upper_bottom))


def process_pinch(grdecl: Mapping, grid):
    """Create vertical NNCs across explicitly inactive Cartesian layers.

    Returns ``None`` when no connection survives. Otherwise the result is a
    dictionary with zero-based ``cells`` and ``faces`` arrays and, when
    ``MULTZ`` is present, a ``mult`` vector.
    """
    if "ACTNUM" not in grdecl or "PINCH" not in grdecl:
        raise KeyError("process_pinch requires ACTNUM and PINCH")
    dims = tuple(int(value) for value in grdecl["cartDims"])
    if len(dims) != 3 or tuple(grid.cartDims or ()) != dims:
        raise ValueError("grid and GRDECL Cartesian dimensions must match")
    if grid.cells.tags is None:
        raise ValueError("process_pinch requires cell-face direction tags")

    actnum = np.asarray(grdecl["ACTNUM"]).ravel(order="F")
    total = int(np.prod(dims))
    if actnum.size != total:
        raise ValueError("ACTNUM must contain one value per Cartesian cell")
    inactive = actnum == 0
    if not np.any(inactive):
        return None

    threshold, _trans_option, mult_option = _pinch_options(grdecl["PINCH"])
    index_map = np.asarray(grid.cells.indexMap, dtype=np.int64)
    inverse = np.full(total, -1, dtype=np.int64)
    inverse[index_map] = np.arange(index_map.size, dtype=np.int64)
    nx, ny, nz = dims

    cell_pairs = []
    face_pairs = []
    global_pairs = []
    for j in range(ny):
        for i in range(nx):
            column = i + nx * j + nx * ny * np.arange(nz)
            active_global = column[inverse[column] >= 0]
            for top_global, bottom_global in zip(
                active_global[:-1], active_global[1:]
            ):
                top_k = int(top_global // (nx * ny))
                bottom_k = int(bottom_global // (nx * ny))
                if bottom_k <= top_k + 1:
                    continue
                skipped = column[top_k + 1:bottom_k]
                if skipped.size == 0 or not np.any(inactive[skipped]):
                    continue
                top_cell = int(inverse[top_global])
                bottom_cell = int(inverse[bottom_global])
                top_face = _cell_face(grid, top_cell, BOTTOM)
                bottom_face = _cell_face(grid, bottom_cell, TOP)
                if top_face == bottom_face:
                    continue
                thickness = _gap_thickness(
                    grdecl["ZCORN"], dims, int(top_global), int(bottom_global)
                )
                if thickness <= threshold:
                    cell_pairs.append([top_cell, bottom_cell])
                    face_pairs.append([top_face, bottom_face])
                    global_pairs.append([int(top_global), int(bottom_global)])

    if not cell_pairs:
        return None
    result = {
        "cells": np.asarray(cell_pairs, dtype=np.int64),
        "faces": np.asarray(face_pairs, dtype=np.int64),
    }
    if "MULTZ" in grdecl:
        multz = np.asarray(grdecl["MULTZ"], dtype=float).ravel(order="F")
        if multz.size != total:
            raise ValueError("MULTZ must contain one value per Cartesian cell")
        multipliers = []
        for top_global, bottom_global in global_pairs:
            if mult_option == "TOP":
                multipliers.append(multz[top_global])
            else:
                step = nx * ny
                multipliers.append(
                    np.min(multz[np.arange(top_global, bottom_global, step)])
                )
        result["mult"] = np.asarray(multipliers, dtype=float)
    return result


processPINCH = process_pinch

__all__ = ["process_pinch", "processPINCH"]
