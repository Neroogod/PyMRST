"""Compare PyMRST Phase 1 grids field-by-field against MATLAB MRST exports.

Workflow: run dump_phase1_reference.m in MATLAB (with MRST started) to
produce ref_*.mat in this folder, then run this script. Index fields are
compared after shifting MATLAB's 1-based values to 0-based (boundary 0
becomes -1); geometry fields are compared to tight tolerances.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry, tensor_grid

HERE = Path(__file__).parent
ATOL = 1e-10
RTOL = 1e-12


def build_cases() -> Dict[str, Callable[[], Any]]:
    """PyMRST constructions mirroring dump_phase1_reference.m, by file name."""
    x = np.array([0.0, 1.0, 2.5, 5.0, 10.0])
    y = np.array([0.0, 2.0, 3.0, 7.0])
    z = np.array([0.0, 1.0, 4.0])
    depthz = 0.1 * np.arange(x.size * y.size, dtype=float).reshape(
        (x.size, y.size), order="F"
    )
    return {
        "ref_cart_3x4x5.mat":
            lambda: cart_grid([3, 4, 5], [300.0, 200.0, 100.0]),
        "ref_tensor_graded.mat":
            lambda: tensor_grid(x, y, z, depthz=depthz),
        "ref_cart_2d_5x7.mat":
            lambda: cart_grid([5, 7], [10.0, 21.0]),
        "ref_tensor_1d.mat":
            lambda: tensor_grid(np.array([0.0, 1.0, 2.5, 5.0])),
        "ref_tensor_nonmonotone.mat": lambda: tensor_grid(
            np.array([0.0, 2.0, 1.0, 3.0]),
            np.array([0.0, 1.0, 1.0, 4.0]),
            np.array([0.0, 5.0, 3.0, 8.0]),
        ),
    }


def load_matlab_grid(path: Path) -> Any:
    """Load the G struct from a -v7 .mat file as nested objects."""
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return mat["G"]


def col(a: np.ndarray) -> np.ndarray:
    """Flatten MATLAB column vectors to 1-D."""
    return np.asarray(a).ravel()


def idx(a: np.ndarray) -> np.ndarray:
    """Flatten MATLAB index arrays to signed ints (0 - 1 must give -1, not
    a wraparound, since MRST saves small grids with unsigned types)."""
    return np.asarray(a, dtype=np.int64).ravel()


def compare_case(name: str, build: Callable[[], Any]) -> Tuple[int, int]:
    """Compare one grid; return (n_checked, n_failed) and print a report."""
    ref = load_matlab_grid(HERE / name)
    g = compute_geometry(build())

    m_cell_faces = np.atleast_2d(np.asarray(ref.cells.faces, dtype=np.int64))
    m_neighbors = np.atleast_2d(np.asarray(ref.faces.neighbors, dtype=np.int64))

    checks = {
        "cells.num": (g.cells.num, int(ref.cells.num)),
        "faces.num": (g.faces.num, int(ref.faces.num)),
        "nodes.num": (g.nodes.num, int(ref.nodes.num)),
        "cartDims": (np.asarray(g.cartDims), idx(ref.cartDims)),
        "cells.facePos": (g.cells.facePos, idx(ref.cells.facePos) - 1),
        "cells.faces": (g.cells.faces, idx(m_cell_faces[:, 0]) - 1),
        "cells.indexMap": (g.cells.indexMap, idx(ref.cells.indexMap) - 1),
        "faces.nodePos": (g.faces.nodePos, idx(ref.faces.nodePos) - 1),
        "faces.nodes": (g.faces.nodes, idx(ref.faces.nodes) - 1),
        "faces.neighbors": (g.faces.neighbors, m_neighbors - 1),
        "nodes.coords": (g.nodes.coords, np.atleast_2d(ref.nodes.coords)),
        "faces.areas": (g.faces.areas, col(ref.faces.areas)),
        "faces.normals": (g.faces.normals, np.atleast_2d(ref.faces.normals)),
        "faces.centroids": (g.faces.centroids,
                            np.atleast_2d(ref.faces.centroids)),
        "cells.volumes": (g.cells.volumes, col(ref.cells.volumes)),
        "cells.centroids": (g.cells.centroids,
                            np.atleast_2d(ref.cells.centroids)),
    }
    if g.cells.tags is not None and m_cell_faces.shape[1] > 1:
        checks["cells.tags"] = (g.cells.tags, idx(m_cell_faces[:, 1]) - 1)

    failed = 0
    for field, (py, ml) in checks.items():
        py, ml = np.asarray(py, dtype=float), np.asarray(ml, dtype=float)
        ml = ml.reshape(py.shape)
        if not np.allclose(py, ml, rtol=RTOL, atol=ATOL):
            worst = float(np.max(np.abs(py - ml)))
            print(f"  FAIL {field}: max abs diff = {worst:.3e}")
            failed += 1
    status = "FAILED" if failed else "matches MATLAB"
    print(f"{name}: {len(checks)} fields checked, {status}")
    return len(checks), failed


def main() -> int:
    total_failed = 0
    missing = 0
    for name, build in build_cases().items():
        if not (HERE / name).exists():
            print(f"{name}: missing (run dump_phase1_reference.m in MATLAB)")
            missing += 1
            continue
        _, failed = compare_case(name, build)
        total_failed += failed
    if missing == len(build_cases()):
        print("\nNo reference files found - nothing validated.")
        return 1
    print("\nAll compared fields match!" if total_failed == 0
          else f"\n{total_failed} field comparisons FAILED")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
