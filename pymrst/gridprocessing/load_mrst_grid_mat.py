"""Load MRST ``G`` grid structs saved in MATLAB ``.mat`` files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes


def load_mrst_grid_mat(path: str | Path, variable: str = "G") -> Grid:
    """Load an MRST grid struct from a MATLAB file into a PyMRST ``Grid``.

    MATLAB/MRST uses one-based indices. The returned PyMRST grid is converted
    to zero-based indices, with exterior face neighbors represented by ``-1``.
    """
    from pymrst.io.mrst_mat import load_mrst_mat

    mat = load_mrst_mat(path)
    if variable not in mat:
        raise KeyError(f"{variable!r} not found in {path}")
    return mrst_grid_struct_to_grid(mat[variable])


def mrst_grid_struct_to_grid(g: Any) -> Grid:
    cells = _field(g, "cells")
    faces = _field(g, "faces")
    nodes = _field(g, "nodes")

    cell_faces_raw = _array(_field(cells, "faces", required=False), dtype=np.int64)
    if cell_faces_raw.ndim == 2:
        cell_faces = cell_faces_raw[:, 0]
        cell_tags = cell_faces_raw[:, 1] if cell_faces_raw.shape[1] > 1 else None
    else:
        cell_faces = cell_faces_raw
        cell_tags = None
    cell_faces = _one_to_zero(cell_faces)

    neighbors = _array(_field(faces, "neighbors"), dtype=np.int64)
    neighbors = np.where(neighbors > 0, neighbors - 1, -1)

    grid = Grid(
        cells=Cells(
            num=int(_scalar(_field(cells, "num"))),
            facePos=_one_to_zero(_array(_field(cells, "facePos"), dtype=np.int64)),
            faces=cell_faces.astype(np.int64),
            indexMap=_one_to_zero(_array(_field(cells, "indexMap"), dtype=np.int64)),
            tags=None if cell_tags is None else np.asarray(cell_tags, dtype=np.int64),
            volumes=_optional_array(cells, "volumes", dtype=float),
            centroids=_optional_array(cells, "centroids", dtype=float),
            nodes=_optional_index_array(cells, "nodes"),
            nodePos=_optional_pos_array(cells, "nodePos"),
        ),
        faces=Faces(
            num=int(_scalar(_field(faces, "num"))),
            nodePos=_one_to_zero(_array(_field(faces, "nodePos"), dtype=np.int64)),
            nodes=_one_to_zero(_array(_field(faces, "nodes"), dtype=np.int64)),
            neighbors=neighbors.astype(np.int64),
            tag=_optional_array(faces, "tag", dtype=np.int64),
            areas=_optional_array(faces, "areas", dtype=float),
            normals=_optional_array(faces, "normals", dtype=float),
            centroids=_optional_array(faces, "centroids", dtype=float),
        ),
        nodes=Nodes(
            num=int(_scalar(_field(nodes, "num"))),
            coords=_array(_field(nodes, "coords"), dtype=float),
        ),
        griddim=int(_scalar(_field(g, "griddim", "gridDim", default=0))),
        type=_type_list(_field(g, "type", default=[])),
        cartDims=_cart_dims(_field(g, "cartDims", default=None)),
    )
    grid.check()
    return grid


def _field(obj: Any, *names: str, required: bool = True, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
            return obj[name]
    if required:
        raise AttributeError(f"missing MATLAB struct field {names!r}")
    return default


def _array(value, dtype=float) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    return np.atleast_1d(arr).squeeze()


def _optional_array(obj, name: str, dtype=float):
    value = _field(obj, name, required=False)
    if value is None or np.asarray(value).size == 0:
        return None
    return _array(value, dtype=dtype)


def _optional_index_array(obj, name: str):
    value = _optional_array(obj, name, dtype=np.int64)
    return None if value is None else _one_to_zero(value)


def _optional_pos_array(obj, name: str):
    value = _optional_array(obj, name, dtype=np.int64)
    return None if value is None else _one_to_zero(value)


def _one_to_zero(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).ravel() - 1


def _scalar(value) -> float:
    return float(np.asarray(value).squeeze())


def _cart_dims(value):
    if value is None or np.asarray(value).size == 0:
        return None
    return tuple(int(v) for v in np.asarray(value).ravel())


def _type_list(value) -> list[str]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=object).ravel()
    return [str(v.item() if hasattr(v, "item") else v) for v in arr]
