"""Phase 22: the core helpers every MRST script assumes.

pside / fluxside, verticalWell, getFaceTransmissibility, gaussianField and
logNormLayers. The two field generators draw random numbers and MATLAB's RNG
is not numpy's, so the MATLAB *draw* is replayed through the Python smoother -
which compares the algorithm rather than the random numbers.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymrst.gridprocessing import cart_grid, compute_geometry     # noqa: E402
from pymrst.params.rock import (                                  # noqa: E402
    gaussian_field,
    log_norm_layers,
    make_rock,
)
from pymrst.params.wells_and_bc import (                          # noqa: E402
    fluxside,
    pside,
    vertical_well,
)
from pymrst.solvers.trans import (                                # noqa: E402
    compute_trans,
    get_face_transmissibility,
)
from pymrst.utils.units import BARSA, DARCY, MILLI                # noqa: E402

REF = Path(__file__).with_name("ref_p22_core_helpers.mat")


def main() -> int:
    if not REF.exists():
        print(f"{REF.name} missing (run dump_phase22_reference.m in MATLAB)")
        return 0

    r = scipy.io.loadmat(REF, squeeze_me=True)
    ok = True

    def chk(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'ok  ' if good else 'FAIL'} {name:36s} {detail}")

    g = compute_geometry(cart_grid([5, 4, 3], [50.0, 40.0, 30.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.2)

    # ---- sides -------------------------------------------------------------
    # MATLAB face indices are 1-based
    for side in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        ml = np.atleast_1d(np.asarray(r[f"side_{side}"], int)).ravel() - 1
        mine = np.sort(pside(None, g, side, 1e5).face)
        chk(f"pside faces: {side}", np.array_equal(mine, np.sort(ml)),
            f"{mine.size} faces")

    # the restricted range: for xmin the free indices are y and z, and MATLAB's
    # [1 2] / [2 3] are 0-based [0 1] / [1 2] here
    ml = np.atleast_1d(np.asarray(r["side_xmin_sub"], int)).ravel() - 1
    mine = np.sort(pside(None, g, "xmin", 1e5, i1=[0, 1], i2=[1, 2]).face)
    chk("pside faces: xmin, index range",
        np.array_equal(mine, np.sort(ml)), f"{mine.size} faces")

    # ---- pressure and flux bcs ---------------------------------------------
    bc = pside(None, g, "xmin", 200 * BARSA)
    bc = pside(bc, g, "xmax", 100 * BARSA)
    ml_f = np.asarray(r["bc_faces"], int).ravel() - 1
    ml_v = np.asarray(r["bc_values"], float).ravel()
    order = np.argsort(bc.face)
    ml_order = np.argsort(ml_f)
    chk("pside: faces and pressures",
        np.array_equal(bc.face[order], ml_f[ml_order])
        and np.allclose(bc.value[order], ml_v[ml_order]),
        f"{bc.face.size} faces")

    bcf = fluxside(None, g, "ymin", 2.0)
    ml_ff = np.asarray(r["flux_faces"], int).ravel() - 1
    ml_fv = np.asarray(r["flux_values"], float).ravel()
    o1, o2 = np.argsort(bcf.face), np.argsort(ml_ff)
    d = np.abs(bcf.value[o1] - ml_fv[o2]).max()
    chk("fluxside: faces and rates",
        np.array_equal(bcf.face[o1], ml_ff[o2]) and d < 1e-12,
        f"total {bcf.value.sum():.3f}, max diff {d:.1e}")
    raw_flux = np.arange(1, bcf.face.size + 1, dtype=float)
    bcfv = fluxside(None, g, "ymin", raw_flux)
    vector_ref = np.asarray(r["flux_vector_values"], float).ravel()
    d = np.abs(bcfv.value-vector_ref).max()
    chk("fluxside: vector area weighting", d < 1e-12,
        f"max diff {d:.1e}")
    range_ref = np.asarray(r["side_xmin_range"], int).ravel()-1
    range_faces = np.sort(
        pside(None, g, "xmin", 3e5, search_range=[0]).face
    )
    chk("pside: perpendicular range",
        np.array_equal(range_faces, np.sort(range_ref)),
        f"{range_faces.size} faces")

    # ---- vertical wells -----------------------------------------------------
    # MATLAB's (I, J) = (3, 2) is 0-based (2, 1)
    w = vertical_well(None, g, rock, 2, 1, well_type="rate", val=1.0,
                      compi=np.array([1.0]))
    ml_c = np.asarray(r["vw_cells"], int).ravel() - 1
    chk("verticalWell: perforated cells",
        np.array_equal(w[0].cells, ml_c), f"{list(w[0].cells)}")
    ml_wi = np.asarray(r["vw_WI"], float).ravel()
    d = np.abs(w[0].WI - ml_wi).max() / np.abs(ml_wi).max()
    chk("verticalWell: well index", d < 1e-12, f"max rel diff {d:.2e}")

    w2 = vertical_well(None, g, rock, 2, 1, k=[0, 2], well_type="bhp",
                       val=1e5, compi=np.array([1.0]))
    ml_c2 = np.asarray(r["vw2_cells"], int).ravel() - 1
    chk("verticalWell: a subset of layers",
        np.array_equal(w2[0].cells, ml_c2), f"{list(w2[0].cells)}")

    # ---- face transmissibility ----------------------------------------------
    t_face = get_face_transmissibility(g, rock)
    ml_t = np.asarray(r["T_face"], float).ravel()
    d = np.abs(t_face - ml_t).max() / np.abs(ml_t).max()
    chk("getFaceTransmissibility", d < 1e-12, f"max rel diff {d:.2e}")

    d = np.abs(compute_trans(g, rock) - np.asarray(r["T_half"], float).ravel()
               ).max() / np.abs(ml_t).max()
    chk("computeTrans (half-face)", d < 1e-12, f"max rel diff {d:.2e}")

    # ---- gaussianField, on MATLAB's own draw --------------------------------
    draw = np.asarray(r["gf_draw"], float)
    sz = np.asarray(r["gf_sz"], int).ravel()
    std = float(r["gf_std"])
    vals = np.asarray(r["gf_vals"], float).ravel()
    mine = gaussian_field(draw.shape, vals=tuple(vals), size=sz, std=std,
                          field=draw)
    ml_gf = np.asarray(r["gf"], float)
    d = np.abs(mine - ml_gf).max() / np.abs(ml_gf).max()
    chk("gaussianField (same draw)", d < 1e-12, f"max rel diff {d:.2e}")

    # ---- logNormLayers, on MATLAB's own draws --------------------------------
    dims = tuple(np.asarray(r["ln_dims"], int).ravel())
    draws = np.asarray(r["ln_draws"], float)          # (nx, ny, 3, nlay)
    shifts = np.asarray(r["ln_shifts"], float).ravel()
    sigma, a, b, std3 = np.asarray(r["ln_params"], float).ravel()
    sz3 = np.asarray(r["ln_sz"], int).ravel()

    nlay = draws.shape[3]
    mine = log_norm_layers(dims, indices=list(range(nlay + 1)),
                           sigma=sigma, a=a, b=b, size=sz3, std=std3,
                           fields=[draws[:, :, :, i] for i in range(nlay)],
                           shifts=list(shifts))
    ml_k = np.asarray(r["ln_K"], float).ravel()
    d = np.abs(mine - ml_k).max() / np.abs(ml_k).max()
    chk("logNormLayers (same draws)", d < 1e-12, f"max rel diff {d:.2e}")

    print(f"\n  (MRST: {ml_k.size} cells, {ml_k.min():.1f}-{ml_k.max():.1f} mD)")
    print("\nCore helpers match MATLAB!" if ok else "\ncomparisons FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
