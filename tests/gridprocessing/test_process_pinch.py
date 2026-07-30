import numpy as np

from pymrst.gridprocessing import process_grdecl, process_pinch


def _regular_grdecl(nx=2, ny=2, nz=3):
    coord = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord.extend([float(i), float(j), 0.0,
                          float(i), float(j), 15.0])
    z = np.empty((2 * nx, 2 * ny, 2 * nz), dtype=float)
    for k in range(nz):
        z[:, :, 2 * k] = 5.0 * k
        z[:, :, 2 * k + 1] = 5.0 * (k + 1)
    act = np.ones((nx, ny, nz), dtype=np.int64)
    act[:, :, 1] = 0
    return {
        "cartDims": (nx, ny, nz),
        "COORD": np.asarray(coord),
        "ZCORN": z.ravel(order="F"),
        "ACTNUM": act.ravel(order="F"),
        "PINCH": [5.0, None, None, "TOPBOT", "TOP"],
        "MULTZ": np.array([
            0.8, 0.7, 0.6, 0.5,
            0.3, 0.4, 0.2, 0.1,
            0.9, 0.95, 0.85, 0.88,
        ]),
    }


def test_process_pinch_connects_across_inactive_middle_layer():
    grdecl = _regular_grdecl()
    grid = process_grdecl(grdecl)
    nnc = process_pinch(grdecl, grid)
    assert nnc["cells"].tolist() == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert np.all(nnc["faces"][:, 0] != nnc["faces"][:, 1])
    assert np.allclose(nnc["mult"], [0.8, 0.7, 0.6, 0.5])


def test_all_multiplier_takes_minimum_over_gap():
    grdecl = _regular_grdecl()
    grdecl["PINCH"][4] = "ALL"
    nnc = process_pinch(grdecl, process_grdecl(grdecl))
    assert np.allclose(nnc["mult"], [0.3, 0.4, 0.2, 0.1])


def test_threshold_excludes_thick_gap():
    grdecl = _regular_grdecl()
    grdecl["PINCH"][0] = 4.99
    assert process_pinch(grdecl, process_grdecl(grdecl)) is None


def test_no_inactive_cells_returns_none():
    grdecl = _regular_grdecl()
    grdecl["ACTNUM"][:] = 1
    assert process_pinch(grdecl, process_grdecl(grdecl)) is None
