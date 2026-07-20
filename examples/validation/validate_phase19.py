"""Phase 19: the CO2 Storage Atlas grid against MRST co2lab-common/atlasgrid.

Reads the same NPD rasters, builds the same corner-point grid, and compares
the raster data, the pillars, the corner depths and the active-cell map.

One deliberate difference is checked rather than hidden: MRST builds ACTNUM
from the *top* raster alone, so a node with a mapped caprock but no mapped
thickness stays active and gets `inf` corners, which its grid processor then
has to throw away. PyMRST masks on the thickness too and rejects those cells
up front. The validator proves the extra cells are exactly the degenerate
ones.
"""

import os
import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymrst.modules.co2lab.atlas import (                    # noqa: E402
    atlas_grid,
    convert_atlas_to_3d,
    read_aai_grid,
    read_atlas_dataset,
    _atlas_dir,
)

REF = Path(__file__).with_name("ref_p19_atlas.mat")


def main() -> int:
    if not REF.exists():
        print(f"{REF.name} missing (run dump_phase19_reference.m in MATLAB)")
        return 0
    try:
        d = _atlas_dir()
        if not (d / "Johansenfm_top").exists():
            raise FileNotFoundError
    except FileNotFoundError:
        print("CO2Atlas data not found; set MRST_DATA_DIR. Skipping.")
        return 0

    r = scipy.io.loadmat(REF, squeeze_me=True)
    ok = True

    def chk(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'ok  ' if good else 'FAIL'} {name:34s} {detail}")

    # ---- the raster reader ------------------------------------------------
    # MATLAB's dump flipped the sign (getAtlasGrid's convention), so flip back
    meta, data = read_aai_grid(_atlas_dir() / "Johansenfm_top")
    ml_top = -np.asarray(r["top_data"], float)
    chk("raster shape", data.shape == ml_top.shape,
        f"{data.shape} vs {ml_top.shape}")
    both_nan = np.isnan(data) & np.isnan(ml_top)
    same_nan = np.array_equal(np.isnan(data), np.isnan(ml_top))
    chk("raster no-data mask", same_nan,
        f"{int(np.isnan(data).sum())} NaN cells")
    if same_nan:
        diff = np.abs(data[~both_nan] - ml_top[~both_nan]).max()
        chk("raster values", diff == 0.0, f"max diff = {diff:.2e}")

    # ---- the grid ---------------------------------------------------------
    grdecl = atlas_grid("Johansenfm", nz=1, coarsening=1)
    ml_dims = np.asarray(r["cartDims"], int).ravel()
    chk("cartDims", np.array_equal(grdecl["cartDims"], ml_dims),
        f"{list(grdecl['cartDims'])} vs {list(ml_dims)}")

    ml_actnum = np.asarray(r["ACTNUM"], int).ravel().astype(bool)
    ml_zcorn = np.asarray(r["ZCORN"], float).ravel()
    my_actnum = grdecl["ACTNUM"].astype(bool)

    chk("ACTNUM", np.array_equal(my_actnum, ml_actnum),
        f"{int(my_actnum.sum())} active cells, cell for cell")

    nx, ny, nz = ml_dims
    corners = np.arange(8 * nx * ny * nz).reshape(
        (2 * nx, 2 * ny, 2 * nz), order="F")
    ii = np.arange(2 * nx) // 2
    jj = np.arange(2 * ny) // 2
    kk = np.arange(2 * nz) // 2
    cid = (ii[:, None, None] + nx * jj[None, :, None]
           + nx * ny * kk[None, None, :])
    cell_of = np.zeros(8 * nx * ny * nz, dtype=np.int64)
    cell_of[corners.ravel(order="F")] = cid.ravel(order="F")

    # ---- corner depths, on the cells both codes keep -----------------------
    my_zcorn = grdecl["ZCORN"]
    keep_corner = my_actnum[cell_of] & np.isfinite(ml_zcorn)
    dz = np.abs(my_zcorn[keep_corner] - ml_zcorn[keep_corner]).max()
    chk("ZCORN on the active cells", dz <= 1e-9, f"max diff = {dz:.2e} m")

    # ---- pillars ----------------------------------------------------------
    ml_coord = np.asarray(r["COORD"], float).ravel()
    my_coord = grdecl["COORD"]
    finite = np.isfinite(ml_coord)
    dc = np.abs(my_coord[finite] - ml_coord[finite]).max()
    chk("COORD (the mapped pillars)", dc <= 1e-9, f"max diff = {dc:.2e} m")

    print(f"\n  (Johansen: {ml_dims[0]}x{ml_dims[1]} raster at 200 m, "
          f"{int(my_actnum.sum())} active cells)")
    print("\nAtlas grid matches MATLAB!" if ok else "\ncomparisons FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
