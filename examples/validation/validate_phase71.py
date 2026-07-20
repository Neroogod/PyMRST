"""Phase 71: vemmech grid utilities vs MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.grid_utils import (
    flip_grid,
    pad_grdecl,
    vertical_grdecl,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p71_vemmech_grid_utils.mat"


def _field(mat_struct, name):
    return getattr(mat_struct, name)


def _grdecl(ref_struct):
    return {
        "cartDims": np.asarray(_field(ref_struct, "cartDims"), dtype=np.int64)
        .ravel(),
        "COORD": np.asarray(_field(ref_struct, "COORD"), dtype=float).ravel(),
        "ZCORN": np.asarray(_field(ref_struct, "ZCORN"), dtype=float).ravel(),
    }


def _box_grdecl(nx=2, ny=2, nz=2):
    coord = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord.extend([i, j, 0.0, i + 0.1, j + 0.2, float(nz)])
    z = np.zeros((2 * nx, 2 * ny, 2 * nz))
    for k in range(nz):
        z[:, :, 2 * k] = k
        z[:, :, 2 * k + 1] = k + 1
    return {
        "cartDims": np.array([nx, ny, nz], dtype=np.int64),
        "COORD": np.asarray(coord, dtype=float),
        "ZCORN": z.ravel(order="F"),
    }


def compare(name, py, ml, rtol=1e-12, atol=1e-12) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ml = np.asarray(ml, dtype=float).ravel()
    scale = max(float(np.nanmax(np.abs(ml))), 1.0)
    diff = float(np.nanmax(np.abs(py - ml)))
    ok = diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def compare_grdecl(name, py, ml) -> int:
    failures = 0
    failures += compare(f"{name} cartDims", py["cartDims"],
                        ml["cartDims"], rtol=0, atol=0)
    failures += compare(f"{name} COORD", py["COORD"], ml["COORD"])
    failures += compare(f"{name} ZCORN", py["ZCORN"], ml["ZCORN"])
    return failures


def main() -> int:
    if not REF.exists():
        raise SystemExit(
            f"Missing {REF.name}. Run dump_phase71_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True, struct_as_record=False)
    gr = _box_grdecl()
    box = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

    failures = 0
    print("Phase 71: vemmech grid utilities vs MATLAB MRST")
    failures += compare_grdecl("vertical all", vertical_grdecl(gr, "all"),
                               _grdecl(ref["v_all"]))
    failures += compare_grdecl("vertical sides", vertical_grdecl(gr, "sides"),
                               _grdecl(ref["v_sides"]))
    failures += compare_grdecl("pad z", pad_grdecl(gr, [False, False, True],
                                                   box), _grdecl(ref["pad_z"]))
    failures += compare_grdecl("pad xz", pad_grdecl(gr, [True, False, True],
                                                    box),
                               _grdecl(ref["pad_xz"]))
    failures += compare_grdecl("pad yz", pad_grdecl(gr, [False, True, True],
                                                    box),
                               _grdecl(ref["pad_yz"]))
    failures += compare_grdecl("pad xyz", pad_grdecl(gr, [True, True, True],
                                                     box),
                               _grdecl(ref["pad_xyz"]))
    failures += compare_grdecl("pad xz relative",
                               pad_grdecl(gr, [True, False, True], box,
                                          relative=True),
                               _grdecl(ref["pad_xz_rel"]))

    g = compute_geometry(cart_grid([2, 2, 2], [2, 4, 6]))
    gf = flip_grid(g)
    failures += compare("flip coords", gf.nodes.coords, ref["flip_coords"])
    failures += compare("flip tags", gf.cells.tags + 1,
                        np.asarray(ref["flip_tags"], dtype=float),
                        rtol=0, atol=0)

    if failures:
        print(f"\nvemmech grid utility validation failed ({failures} checks).")
    else:
        print("\nPyMRST matches MRST vemmech grid utilities.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
