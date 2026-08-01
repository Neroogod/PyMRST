"""Validate computeGeometry's MaxBlockSize option against an MRST reference.

MaxBlockSize is an operational control: it bounds how many cells the 3D cell
pass holds at once, and must not move a single number. This pins the block
boundaries against MRST's `partition_cells` arithmetic and the geometry
against MRST's own output on the same perturbed grid.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid
from pymrst.gridprocessing.compute_geometry import compute_geometry, _partition_cells

REF = Path(__file__).with_name("ref_grid_compute_geometry_blocks.mat")
BLOCK_SIZES = [1, 2, 3, 7, 13, 59, 60, 61]


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _grid(ref):
    grid = cart_grid([5, 4, 3], [5.0, 4.0, 3.0])
    grid.nodes.coords[:] = ref["node_coords"]
    return grid


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_grid_compute_geometry_blocks_reference.m")
        return 1
    ref = loadmat(REF, squeeze_me=True)
    failed = 0

    print("partition_cells block boundaries vs MRST:")
    cases = np.atleast_2d(ref["block_cases"])
    for index, (cells, size) in enumerate(cases):
        expected = list(zip(np.atleast_1d(ref["block_starts"][index]).tolist(),
                            np.atleast_1d(ref["block_stops"][index]).tolist()))
        actual = _partition_cells(int(cells), int(size))
        failed += _check(f"{int(cells)} cells in blocks of {int(size)}",
                         actual == expected,
                         "" if actual == expected else f"{actual} vs {expected}")

    print("geometry vs MRST, single pass:")
    single = compute_geometry(_grid(ref), max_block_size=None)
    checks = {
        "cell volumes": (single.cells.volumes, ref["single_volumes"]),
        "cell centroids": (single.cells.centroids, ref["single_centroids"]),
        "face areas": (single.faces.areas, ref["single_face_areas"]),
    }
    for name, (actual, expected) in checks.items():
        actual = np.asarray(actual).squeeze()
        expected = np.asarray(expected).squeeze()
        error = (np.max(np.abs(actual - expected))
                 if actual.shape == expected.shape else np.inf)
        failed += _check(name, error < 1e-12, f"max|d| = {error:.3e}")

    print("blocking changes nothing:")
    worst = 0.0
    for size in BLOCK_SIZES:
        blocked = compute_geometry(_grid(ref), max_block_size=size)
        worst = max(
            worst,
            float(np.max(np.abs(blocked.cells.volumes - single.cells.volumes))),
            float(np.max(np.abs(blocked.cells.centroids - single.cells.centroids))),
            float(np.max(np.abs(blocked.faces.areas - single.faces.areas))),
        )
    failed += _check(f"identical across block sizes {BLOCK_SIZES}", worst == 0.0,
                     f"max|d| = {worst:.3e}")
    failed += _check("MRST reports the same invariance",
                     float(ref["blocked_max_diff"]) == 0.0)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
