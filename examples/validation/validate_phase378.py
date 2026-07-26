"""Phase 378: repairNormals against MRST (Phase 25 core leftovers).

Pins `repair_normals` against MRST's `repairNormals` on an irregular two-cell 2D
grid. Cell areas ("volumes") are compared per cell (tessellation order matches)
and the face lengths ("areas") as a sorted set. This validates the geometry
`repairNormals` recomputes for a 2D grid that carried topology only.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing.repair_normals import repair_normals
from pymrst.gridprocessing.tessellation_grid import tessellation_grid

REF = Path(__file__).with_name("ref_p378_repair.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase378_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    p = np.asarray(ref["p"], dtype=float)
    t = np.asarray(ref["t"], dtype=np.int64) - 1        # 1-based -> 0-based

    g = repair_normals(tessellation_grid(p, t))

    failed = 0
    print("repairNormals (irregular 2-cell grid) vs MRST:")
    err_v = np.max(np.abs(np.asarray(g.cells.volumes) - ref["cell_volumes"].ravel()))
    failed += _check("cell areas match", err_v < 1e-12, f"max|dA| = {err_v:.3e}")
    areas = np.sort(np.asarray(g.faces.areas))
    err_a = np.max(np.abs(areas - ref["face_areas"].ravel()))
    failed += _check("face lengths match (sorted)", err_a < 1e-12,
                     f"max|dl| = {err_a:.3e}")
    # normal magnitude equals face length (internal consistency)
    failed += _check("normal magnitudes equal face lengths",
                     np.allclose(np.linalg.norm(g.faces.normals, axis=1),
                                 g.faces.areas))

    print("\nrepairNormals matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
