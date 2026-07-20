"""Compare PyMRST Phase 23 unstructured grids field-by-field against MATLAB.

Workflow: run dump_phase23_reference.m in MATLAB (with MRST started) to
produce ref_p23_*.mat in this folder, then run this script. The point sets
and triangulations are read from the reference files, so both codes build
from the same connectivity and the comparison isolates the grid
construction (triangle_grid, tetrahedral_grid, pebi, make_layered_grid),
not the triangulator. Index fields are compared after shifting MATLAB's
1-based values to 0-based (boundary 0 becomes -1).
"""

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import scipy.io

from pymrst.gridprocessing import (
    compute_geometry,
    make_layered_grid,
    pebi,
    tetrahedral_grid,
    triangle_grid,
)

HERE = Path(__file__).parent
ATOL = 1e-10
RTOL = 1e-12


def load_case(path: Path) -> Any:
    """Load a -v7 .mat reference file as nested objects."""
    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)


def build_cases() -> Dict[str, Callable[[Any], Any]]:
    """PyMRST constructions mirroring dump_phase23_reference.m, by file name."""
    def tri(mat):
        return triangle_grid(mat["p"], np.asarray(mat["t"], np.int64) - 1)

    def dual(mat):
        return pebi(triangle_grid(mat["p"], np.asarray(mat["t"], np.int64) - 1))

    def dual_aux(mat):
        return pebi(triangle_grid(mat["pb"], np.asarray(mat["tb"], np.int64) - 1))

    def tet(mat):
        return tetrahedral_grid(mat["p3"], np.asarray(mat["t3"], np.int64) - 1)

    def layered(mat):
        areal = triangle_grid(mat["p"], np.asarray(mat["t"], np.int64) - 1)
        return make_layered_grid(areal, np.asarray(mat["thickness"], float))

    return {
        "ref_p23_tri.mat": tri,
        "ref_p23_pebi.mat": dual,
        "ref_p23_pebi_aux.mat": dual_aux,
        "ref_p23_tet.mat": tet,
        "ref_p23_layered.mat": layered,
    }


def idx(a: np.ndarray) -> np.ndarray:
    """Flatten MATLAB index arrays to signed ints."""
    return np.asarray(a, dtype=np.int64).ravel()


def col(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).ravel()


def compare_case(name: str, build: Callable[[Any], Any]) -> Tuple[int, int]:
    """Compare one grid; return (n_checked, n_failed) and print a report."""
    mat = load_case(HERE / name)
    ref = mat["G"]
    g = compute_geometry(build(mat))

    m_cell_faces = np.atleast_2d(np.asarray(ref.cells.faces, dtype=np.int64))
    if m_cell_faces.shape[0] == 1:
        m_cell_faces = m_cell_faces.T
    m_neighbors = np.atleast_2d(np.asarray(ref.faces.neighbors, dtype=np.int64))

    checks = {
        "cells.num": (g.cells.num, int(ref.cells.num)),
        "faces.num": (g.faces.num, int(ref.faces.num)),
        "nodes.num": (g.nodes.num, int(ref.nodes.num)),
        "cells.facePos": (g.cells.facePos, idx(ref.cells.facePos) - 1),
        "cells.faces": (g.cells.faces, idx(m_cell_faces[:, 0]) - 1),
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
    if hasattr(ref.cells, "indexMap"):
        checks["cells.indexMap"] = (g.cells.indexMap,
                                    idx(ref.cells.indexMap) - 1)
    if hasattr(ref, "numLayers"):
        checks["numLayers"] = (g.numLayers, int(ref.numLayers))
        checks["layerSize"] = (g.layerSize, int(ref.layerSize))

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
            print(f"{name}: missing (run dump_phase23_reference.m in MATLAB)")
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
