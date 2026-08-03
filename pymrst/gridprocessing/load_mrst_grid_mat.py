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
            indexMap=_index_map(cells, int(_scalar(_field(cells, "num")))),
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
        # These three carry defaults because MRST does not always write them,
        # and `_field` only returns a default when `required=False` - passing
        # one without it means the default can never be reached. `cartDims` is
        # the case that bites: MRST sets it only for grids built on a Cartesian
        # index space, so `triangleGrid`, PEBI and tetrahedral grids have no
        # such field and used to fail to load with a bare AttributeError.
        griddim=int(_scalar(_field(g, "griddim", "gridDim", required=False, default=0))),
        type=_type_list(_field(g, "type", required=False, default=[])),
        cartDims=_cart_dims(_field(g, "cartDims", required=False, default=None)),
    )
    grid.check()
    return grid


def _index_map(cells: Any, num: int) -> np.ndarray:
    """`cells.indexMap`, or the identity when MRST did not write one.

    `indexMap` records where each cell sits in an underlying logical Cartesian
    index space. Grids that have no such space - `triangleGrid`, PEBI,
    tetrahedral - simply omit the field, so requiring it rejected every
    unstructured MRST grid. For those the identity map is the correct reading:
    cell `i` is cell `i`.
    """
    value = _field(cells, "indexMap", required=False)
    if value is None or np.asarray(value).size == 0:
        return np.arange(num, dtype=np.int64)
    return _one_to_zero(_array(value, dtype=np.int64))


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


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from scipy.io import savemat

    from pymrst.gridprocessing import cart_grid, compute_geometry

    def _matlab_struct(g, *, include=("indexMap", "cartDims", "type", "griddim")):
        """A PyMRST grid re-expressed with MATLAB's conventions.

        Optional fields can be dropped to imitate the grids MRST writes for
        unstructured cases.
        """
        s = {
            "cells": {
                "num": float(g.cells.num),
                "facePos": np.asarray(g.cells.facePos, float) + 1.0,
                "faces": np.asarray(g.cells.faces, float) + 1.0,
                "volumes": np.asarray(g.cells.volumes, float),
                "centroids": np.asarray(g.cells.centroids, float),
            },
            "faces": {
                "num": float(g.faces.num),
                "nodePos": np.asarray(g.faces.nodePos, float) + 1.0,
                "nodes": np.asarray(g.faces.nodes, float) + 1.0,
                "neighbors": np.asarray(g.faces.neighbors, float) + 1.0,
                "areas": np.asarray(g.faces.areas, float),
                "centroids": np.asarray(g.faces.centroids, float),
                "normals": np.asarray(g.faces.normals, float),
            },
            "nodes": {"num": float(g.nodes.num),
                      "coords": np.asarray(g.nodes.coords, float)},
        }
        if "indexMap" in include:
            s["cells"]["indexMap"] = np.asarray(g.cells.indexMap, float) + 1.0
        if "cartDims" in include:
            s["cartDims"] = np.asarray(g.cartDims, float)
        if "type" in include:
            s["type"] = ["cartGrid"]
        if "griddim" in include:
            s["griddim"] = float(g.griddim)
        return s

    def _load(struct):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.mat"
            savemat(p, {"G": struct})
            from pymrst.io.mrst_mat import load_mrst_mat
            return mrst_grid_struct_to_grid(load_mrst_mat(p)["G"])

    def test_full_grid_struct_round_trips():
        src = compute_geometry(cart_grid([3, 2, 1], [3.0, 2.0, 1.0]))
        g = _load(_matlab_struct(src))
        assert g.cells.num == src.cells.num
        assert np.array_equal(np.asarray(g.faces.neighbors),
                              np.asarray(src.faces.neighbors))
        assert np.array_equal(np.asarray(g.cells.indexMap),
                              np.asarray(src.cells.indexMap))
        assert tuple(g.cartDims) == tuple(src.cartDims)
        print("ok test_full_grid_struct_round_trips")

    def test_unstructured_grid_without_optional_fields_loads():
        """MRST omits these for grids with no Cartesian index space.

        `triangleGrid`, PEBI and tetrahedral grids carry no `cartDims` and no
        `cells.indexMap` - checked against MATLAB, where `triangleGrid` reports
        only (faces, cells, nodes, type, griddim). Every one of those used to
        fail here with a bare `missing MATLAB struct field` error: the call
        sites passed a `default=` but left `required=True`, so the default
        could never be returned. Only Cartesian grids could be imported.
        """
        src = compute_geometry(cart_grid([3, 2, 1], [3.0, 2.0, 1.0]))
        g = _load(_matlab_struct(src, include=("type", "griddim")))
        assert g.cartDims is None, g.cartDims
        # With no index space to map into, cell i is cell i.
        assert np.array_equal(np.asarray(g.cells.indexMap),
                              np.arange(src.cells.num)), g.cells.indexMap
        assert g.cells.num == src.cells.num
        assert np.array_equal(np.asarray(g.faces.neighbors),
                              np.asarray(src.faces.neighbors))
        print("ok test_unstructured_grid_without_optional_fields_loads")

    def test_griddim_and_type_default_when_absent():
        src = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        g = _load(_matlab_struct(src, include=("indexMap", "cartDims")))
        assert g.type == [], g.type
        assert g.griddim == 0, g.griddim
        print("ok test_griddim_and_type_default_when_absent")

    test_full_grid_struct_round_trips()
    test_unstructured_grid_without_optional_fields_loads()
    test_griddim_and_type_default_when_absent()
    print("\nAll tests passed!")
