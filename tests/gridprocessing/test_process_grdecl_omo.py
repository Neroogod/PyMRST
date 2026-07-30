import numpy as np
import pytest

from pymrst.gridprocessing import (
    compute_geometry,
    process_grdecl_omo,
)


def _grdecl(nx=2, ny=2, nz=2):
    xyz = np.zeros((6, nx + 1, ny + 1))
    for i in range(nx + 1):
        for j in range(ny + 1):
            xyz[:, i, j] = [i, j, 0.0, i, j, nz]
    z = np.empty((2 * nx, 2 * ny, 2 * nz))
    for k in range(nz):
        z[:, :, 2 * k] = k
        z[:, :, 2 * k + 1] = k + 1
    return {
        "cartDims": (nx, ny, nz),
        "COORD": xyz.ravel(order="F"),
        "ZCORN": z.ravel(order="F"),
    }


def test_omo_compatibility_entry_point_builds_geometry():
    grid = compute_geometry(process_grdecl_omo(_grdecl()))
    assert grid.cells.num == 8
    assert np.isclose(grid.cells.volumes.sum(), 8.0)


def test_omo_repair_zcorn_option_repairs_before_processing():
    grdecl = _grdecl(2, 2, 1)
    z = grdecl["ZCORN"].reshape((4, 4, 2), order="F")
    z[0, 0, 0] = 2.0
    grid = compute_geometry(process_grdecl_omo(grdecl, repair_zcorn=True))
    assert grid.cells.num == 4
    assert np.all(grid.cells.volumes >= 0)


def test_preserve_cp_nodes_retains_original_cell_vertices():
    grid = process_grdecl_omo(_grdecl(), preserve_cp_nodes=True)
    assert grid.cells.cpnodes.shape == (grid.cells.num, 8)
    assert np.all(grid.cells.cpnodes >= 0)
    assert np.all(grid.cells.cpnodes < grid.nodes.num)
    first = grid.nodes.coords[grid.cells.cpnodes[0]]
    assert {tuple(point) for point in first} == {
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0), (1.0, 1.0, 1.0),
    }
