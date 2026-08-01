"""Check the remaining ad-props assign* functions against MRST.

Companion to `validate_ad_props_assign.py`, covering the three-column
saturation family, the solvent and miscibility tables, the small record
keywords and the three-phase `relPerm` dispatcher.

Run `dump_ad_props_assign2_reference.m` in MATLAB first.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_props.props.assign_ehystr import assign_ehystr
from pymrst.modules.ad_props.props.assign_msfn import assign_msfn
from pymrst.modules.ad_props.props.assign_pvds import assign_pvds
from pymrst.modules.ad_props.props.assign_relperm import assign_relperm
from pymrst.modules.ad_props.props.assign_rocktab import assign_rocktab
from pymrst.modules.ad_props.props.assign_rsconstt import assign_rsconstt
from pymrst.modules.ad_props.props.assign_sdensity import assign_sdensity
from pymrst.modules.ad_props.props.assign_sgfn import assign_sgfn
from pymrst.modules.ad_props.props.assign_sgof import assign_sgof
from pymrst.modules.ad_props.props.assign_sgwfn import assign_sgwfn
from pymrst.modules.ad_props.props.assign_slgof import assign_slgof
from pymrst.modules.ad_props.props.assign_sof2 import assign_sof2
from pymrst.modules.ad_props.props.assign_sof3 import assign_sof3
from pymrst.modules.ad_props.props.assign_ssfn import assign_ssfn
from pymrst.modules.ad_props.props.assign_swfn import assign_swfn
from pymrst.modules.ad_props.props.assign_swof import assign_swof
from pymrst.modules.ad_props.utils.handle_regions import Regions

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "ref_ad_props_assign2.mat"
TOLERANCE = 1e-10

SWFN = np.array([[0.15, 0.00, 3.0e4], [0.40, 0.10, 8.0e3],
                 [0.70, 0.45, 1.0e3], [0.85, 1.00, 0.00]])
SGFN = np.array([[0.00, 0.00, 0.0], [0.10, 0.00, 1.0e3],
                 [0.40, 0.25, 6.0e3], [0.80, 0.90, 2.0e4]])
SOF3 = np.array([[0.00, 0.00, 0.00], [0.20, 0.00, 0.00],
                 [0.50, 0.30, 0.25], [0.85, 1.00, 0.90]])
SWOF2 = np.array([[0.12, 0.00, 1.00, 0.0], [0.90, 1.00, 0.00, 0.0]])
SLGOF = np.array([[0.12, 1.00, 0.00, 0.0], [0.50, 0.40, 0.10, 0.0],
                  [0.80, 0.05, 0.55, 0.0], [1.00, 0.00, 1.00, 0.0]])
SGWFN = np.array([[0.00, 0.00, 1.00, 0.0], [0.30, 0.10, 0.50, 2.0e3],
                  [0.70, 0.60, 0.10, 8.0e3], [1.00, 1.00, 0.00, 2.0e4]])
SSFN = np.array([[0.00, 0.00, 1.00], [0.40, 0.35, 0.60], [1.00, 1.00, 0.00]])
MSFN = np.array([[0.00, 0.00, 1.00], [0.50, 0.45, 0.55], [1.00, 1.00, 0.00]])
PVDS = np.array([[1.0e7, 5.0e-3, 1.4e-5], [2.0e7, 3.0e-3, 1.7e-5],
                 [3.0e7, 2.5e-3, 2.0e-5]])
ROCKTAB = np.array([[1.0e6, 0.98, 0.90], [2.0e7, 1.00, 1.00],
                    [5.0e7, 1.05, 1.20]])
SWOF3 = np.array([[0.10, 0.00, 1.00, 0.0], [0.50, 0.20, 0.40, 0.0],
                  [0.90, 1.00, 0.00, 0.0]])
SGOF3 = np.array([[0.00, 0.00, 1.00, 0.0], [0.40, 0.30, 0.35, 0.0],
                  [0.90, 1.00, 0.00, 0.0]])


def check(failures, label, got, want) -> None:
    got = np.asarray(got, dtype=float).ravel()
    want = np.asarray(want, dtype=float).ravel()
    if got.shape != want.shape:
        failures.append(f"{label}: shape {got.shape} vs MRST {want.shape}")
        return
    if not np.allclose(got, want, rtol=TOLERANCE, atol=0, equal_nan=True):
        error = np.nanmax(np.abs(got - want))
        failures.append(f"{label}: max abs diff {error:.3e}")
        return
    print(f"  {label:20} {got.size:3d} value(s)  ok")


def main() -> int:
    if not REFERENCE.exists():
        print(f"missing {REFERENCE.name}; run "
              f"dump_ad_props_assign2_reference.m in MATLAB first")
        return 2
    ref = loadmat(REFERENCE)
    reg = Regions()
    failures = []
    s = ref["s"].ravel()
    p = ref["p"].ravel()

    print("three-column saturation family")
    f = assign_swfn({}, [SWFN], reg)
    check(failures, "SWFN krW", f["krW"][0](s), ref["SWFN_krW"])
    check(failures, "SWFN pcOW", f["pcOW"][0](s), ref["SWFN_pcOW"])
    check(failures, "SWFN pts", f["krPts"]["w"], ref["SWFN_pts"])

    f = assign_sgfn(f, [SGFN], reg)
    check(failures, "SGFN krG", f["krG"][0](s), ref["SGFN_krG"])
    check(failures, "SGFN pcOG", f["pcOG"][0](s), ref["SGFN_pcOG"])
    check(failures, "SGFN pts", f["krPts"]["g"], ref["SGFN_pts"])

    g = assign_sof3({}, [SOF3], reg)
    check(failures, "SOF3 krOW", g["krOW"][0](s), ref["SOF3_krOW"])
    check(failures, "SOF3 krOG", g["krOG"][0](s), ref["SOF3_krOG"])
    check(failures, "SOF3 pts ow", g["krPts"]["ow"], ref["SOF3_pts_ow"])
    check(failures, "SOF3 pts og", g["krPts"]["og"], ref["SOF3_pts_og"])

    h = assign_sof2({}, [SOF3[:, :2]], reg)
    check(failures, "SOF2 krO", h["krO"](s), ref["SOF2_krO"])

    print("SLGOF, SGWFN, SSFN, MSFN")
    ll = assign_swof({}, [SWOF2], reg)
    ll = assign_slgof(ll, [SLGOF], reg)
    check(failures, "SLGOF krG", ll["krG"][0](s), ref["SLGOF_krG"])
    check(failures, "SLGOF krOG", ll["krOG"][0](s), ref["SLGOF_krOG"])
    check(failures, "SLGOF pts g", ll["krPts"]["g"], ref["SLGOF_pts_g"])

    w = assign_sgwfn({}, [SGWFN], reg)
    check(failures, "SGWFN krG", w["krG"](s), ref["SGWFN_krG"])
    check(failures, "SGWFN krW", w["krW"](s), ref["SGWFN_krW"])
    check(failures, "SGWFN pcWG", w["pcWG"](s), ref["SGWFN_pcWG"])

    v = assign_ssfn({}, [SSFN], reg)
    check(failures, "SSFN krFG", v["krFG"](s), ref["SSFN_krFG"])
    check(failures, "SSFN krFS", v["krFS"](s), ref["SSFN_krFS"])

    m = assign_msfn({}, [MSFN], reg)
    check(failures, "MSFN MkrG", m["MkrG"](s), ref["MSFN_MkrG"])
    check(failures, "MSFN MkrO", m["MkrO"](s), ref["MSFN_MkrO"])

    print("solvent PVT, rock table, records")
    d = assign_pvds({}, [PVDS], reg)
    check(failures, "PVDS bS", d["bS"](p), ref["PVDS_bS"])
    check(failures, "PVDS muS", d["muS"](p), ref["PVDS_muS"])

    r = assign_rocktab({}, [ROCKTAB], reg)
    check(failures, "ROCKTAB pvMultR", r["pvMultR"](p), ref["ROCKTAB_pvMultR"])
    check(failures, "ROCKTAB tranMultR", r["tranMultR"](p),
          ref["ROCKTAB_tranMultR"])

    check(failures, "SDENSITY", assign_sdensity({}, [[700.0]], reg)["rhoSS"],
          ref["SDENSITY_rhoSS"])
    rc = assign_rsconstt({}, [[25.0, 1.5e7]], reg)
    check(failures, "RSCONSTT", [rc["rs"], rc["pBub"]], ref["RSCONSTT"])
    eh = assign_ehystr({}, [0.1, 2, 1.0, 0.1, "BOTH", "RETR", "DRAIN", "OIL",
                            "NO", "NO", "NO", 0.0, 0])
    check(failures, "EHYSTR flags",
          [float("krHyst" in eh), float("pcHyst" in eh)], ref["EHYSTR_flags"])

    print("three-phase relPerm")
    rp = assign_swof({}, [SWOF3], reg)
    rp = assign_sgof(rp, [SGOF3], reg)
    for name in ("krW", "krOW", "krG", "krOG"):
        rp[name] = rp[name][0]
    rp["sWcon"] = 0.10
    rp = assign_relperm(rp)
    kr_w, kr_o, kr_g = rp["relPerm"](ref["relPerm_sw"].ravel(),
                                     ref["relPerm_sg"].ravel())
    check(failures, "relPerm krW", kr_w, ref["relPerm_krW"])
    check(failures, "relPerm krO", kr_o, ref["relPerm_krO"])
    check(failures, "relPerm krG", kr_g, ref["relPerm_krG"])

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("All remaining assign* values match MRST.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
