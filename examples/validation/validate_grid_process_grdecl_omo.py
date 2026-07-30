"""Validate the processGRDECLOMO compatibility entry point against MRST."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import compute_geometry, process_grdecl_omo


HERE = Path(__file__).resolve().parent


def _grdecl():
    nx, ny, nz = 3, 2, 2
    xyz = np.zeros((6, nx + 1, ny + 1))
    for i in range(nx + 1):
        for j in range(ny + 1):
            xyz[:, i, j] = [i, j, 0.0, i, j, nz]
    z = np.empty((2 * nx, 2 * ny, 2 * nz))
    for k in range(nz):
        z[:, :, 2 * k] = k
        z[:, :, 2 * k + 1] = k + 1
    actnum = np.ones(nx * ny * nz, dtype=np.int64)
    actnum[5] = 0
    return {
        "cartDims": (nx, ny, nz),
        "COORD": xyz.ravel(order="F"),
        "ZCORN": z.ravel(order="F"),
        "ACTNUM": actnum,
    }


def main() -> int:
    path = HERE / "ref_grid_process_grdecl_omo.mat"
    if not path.exists():
        print(f"{path.name} missing (run the matching MATLAB dump)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    grid = compute_geometry(process_grdecl_omo(
        _grdecl(), preserve_cp_nodes=True,
    ))
    order = np.lexsort(tuple(
        grid.cells.centroids[:, column]
        for column in range(grid.cells.centroids.shape[1] - 1, -1, -1)
    ))
    # np.lexsort's last key is primary; use a structured row sort explicitly.
    order = np.array(sorted(
        range(grid.cells.num),
        key=lambda index: tuple(grid.cells.centroids[index]),
    ))
    checks = {
        "cell count": (grid.cells.num, ref["cell_count"]),
        "face count": (grid.faces.num, ref["face_count"]),
        "node count": (grid.nodes.num, ref["node_count"]),
        "index map": (grid.cells.indexMap + 1, ref["index_map"]),
        "sorted centroids": (
            grid.cells.centroids[order], ref["sorted_centroids"],
        ),
        "sorted volumes": (grid.cells.volumes[order], ref["sorted_volumes"]),
        "total volume": (grid.cells.volumes.sum(), ref["total_volume"]),
        "corner-point nodes": (grid.cells.cpnodes + 1, ref["cpnodes"]),
        "corner-point coordinates": (
            grid.nodes.coords[grid.cells.cpnodes],
            ref["cpnode_coords"],
        ),
    }
    failed = False
    for name, (actual, expected) in checks.items():
        actual, expected = np.asarray(actual).squeeze(), np.asarray(expected).squeeze()
        passed = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-12, atol=1e-12
        )
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
        if not passed:
            print(f"     PyMRST={actual!r}\n     MRST={expected!r}")
        failed |= not passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
