"""Validate extractSubgrid, removeCells, splitDisconnectedGrid,
makeInternalBoundary, and removeInternalBoundary against MATLAB/MRST.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import (
    cart_grid,
    compute_geometry,
    extract_subgrid,
    make_internal_boundary,
    remove_cells,
    remove_internal_boundary,
    split_disconnected_grid,
)


HERE = Path(__file__).resolve().parent


def _counts(g):
    return np.array([g.cells.num, g.faces.num, g.nodes.num])


def _sorted_neighbors(g):
    neighbors = np.sort(np.asarray(g.faces.neighbors) + 1, axis=1)
    return neighbors[np.lexsort((neighbors[:, 1], neighbors[:, 0]))]


def main() -> int:
    path = HERE / "ref_grid_editing.mat"
    if not path.exists():
        print(f"{path.name} missing (run the matching MATLAB dump)")
        return 1
    ref = loadmat(path, squeeze_me=True)

    g = compute_geometry(cart_grid([5, 1, 1], [5.0, 1.0, 1.0]))
    extracted = extract_subgrid(g, [0, 1, 3, 4])
    removed = remove_cells(g, np.array([2]))
    parts = split_disconnected_grid(removed)

    internal = np.flatnonzero(
        np.all(g.faces.neighbors >= 0, axis=1)
        & np.any(g.faces.neighbors == 1, axis=1)
        & np.any(g.faces.neighbors == 2, axis=1)
    )
    boundary, pairing = make_internal_boundary(g, internal[:1])
    welded, merged = remove_internal_boundary(boundary, pairing)

    split_maps = np.concatenate([part.cells.indexMap for part in parts])
    split_pos = np.concatenate([[1], 1 + np.cumsum([p.cells.num for p in parts])])
    checks = {
        "extract counts": (_counts(extracted), ref["extract_counts"]),
        "extract indexMap": (extracted.cells.indexMap + 1, ref["extract_index_map"]),
        "extract volumes": (extracted.cells.volumes, ref["extract_volumes"]),
        "extract centroids": (extracted.cells.centroids, ref["extract_centroids"]),
        "extract face areas": (extracted.faces.areas, ref["extract_face_areas"]),
        "extract face centroids": (
            extracted.faces.centroids, ref["extract_face_centroids"],
        ),
        "remove counts": (_counts(removed), ref["remove_counts"]),
        "remove indexMap": (removed.cells.indexMap + 1, ref["remove_index_map"]),
        "remove volumes": (removed.cells.volumes, ref["remove_volumes"]),
        "remove centroids": (removed.cells.centroids, ref["remove_centroids"]),
        "remove face areas": (removed.faces.areas, ref["remove_face_areas"]),
        "remove face centroids": (
            removed.faces.centroids, ref["remove_face_centroids"],
        ),
        "split counts": (
            np.array([part.cells.num for part in parts]), ref["split_counts"],
        ),
        "split positions": (split_pos, ref["split_pos"]),
        "split index maps": (split_maps + 1, ref["split_index_maps"]),
        "roundtrip counts": (_counts(welded), ref["roundtrip_counts"]),
        "roundtrip neighbors": (
            _sorted_neighbors(welded), ref["roundtrip_neighbors"],
        ),
        "roundtrip merged neighbors": (
            np.sort(welded.faces.neighbors[merged] + 1, axis=1),
            ref["roundtrip_merged_neighbors"],
        ),
    }

    failed = False
    for name, (actual, expected) in checks.items():
        actual = np.asarray(actual).squeeze()
        expected = np.asarray(expected).squeeze()
        passed = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-13, atol=1e-13
        )
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
        if not passed:
            print(f"     PyMRST={actual!r}\n     MRST={expected!r}")
        failed |= not passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
