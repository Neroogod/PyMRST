"""Validate 3D hinge-node geometry against an MRST reference dump."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry


HERE = Path(__file__).resolve().parent


def main() -> int:
    path = HERE / "ref_grid_compute_geometry_hinge.mat"
    if not path.exists():
        print(f"{path.name} missing (run the matching MATLAB dump)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    source = cart_grid([1, 1, 1], [1, 1, 1])
    source.nodes.coords[:] = ref["node_coords"]
    face = int(ref["face0"])
    default = compute_geometry(deepcopy(source))
    hinged = compute_geometry(
        deepcopy(source),
        hinge_nodes={"faces": [face], "nodes": [ref["hinge"]]},
    )
    checks = {
        "face nodes": (source.face_nodes(face), ref["face_nodes0"]),
        "default face areas": (default.faces.areas, ref["default_face_areas"]),
        "default face normals": (default.faces.normals, ref["default_face_normals"]),
        "default face centroids": (default.faces.centroids, ref["default_face_centroids"]),
        "default cell volumes": (default.cells.volumes, ref["default_cell_volumes"]),
        "default cell centroids": (default.cells.centroids, ref["default_cell_centroids"]),
        "hinged face areas": (hinged.faces.areas, ref["hinged_face_areas"]),
        "hinged face normals": (hinged.faces.normals, ref["hinged_face_normals"]),
        "hinged face centroids": (hinged.faces.centroids, ref["hinged_face_centroids"]),
        "hinged cell volumes": (hinged.cells.volumes, ref["hinged_cell_volumes"]),
        "hinged cell centroids": (hinged.cells.centroids, ref["hinged_cell_centroids"]),
    }
    failed = False
    for name, (actual, expected) in checks.items():
        actual = np.asarray(actual).squeeze()
        expected = np.asarray(expected).squeeze()
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
