"""Check the ad-props assign* functions against MRST.

The assign functions turn deck tables into property closures. Most are simple
enough to reason about, but the undersaturated PVT branches are not: oil and
gas use *different* interpolation strategies ('parallel' and 'linshift'), and
both return plausible numbers when given the wrong one. Hand-derived
expectations are unreliable here, so the probe values come from MRST.

Run `dump_ad_props_assign_reference.m` in MATLAB first.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_props.props.assign_density import assign_density
from pymrst.modules.ad_props.props.assign_gravity import assign_gravity
from pymrst.modules.ad_props.props.assign_pvcdo import assign_pvcdo
from pymrst.modules.ad_props.props.assign_pvdg import assign_pvdg
from pymrst.modules.ad_props.props.assign_pvdo import assign_pvdo
from pymrst.modules.ad_props.props.assign_pvtg import assign_pvtg
from pymrst.modules.ad_props.props.assign_pvto import assign_pvto
from pymrst.modules.ad_props.props.assign_pvtw import assign_pvtw
from pymrst.modules.ad_props.props.assign_rock import assign_rock
from pymrst.modules.ad_props.props.assign_sgof import assign_sgof
from pymrst.modules.ad_props.props.assign_swof import assign_swof
from pymrst.modules.ad_props.utils.handle_regions import Regions
from pymrst.modules.ad_props.utils.preprocess_table_pvt import PackedPVT

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "ref_ad_props_assign.mat"
TOLERANCE = 1e-10

# The fixtures below must match dump_ad_props_assign_reference.m exactly.
PVTO = PackedPVT(
    key=np.array([10.0, 50.0]),
    pos=np.array([0, 2, 4]),
    data=np.array([[1.0e7, 1.20, 1.0e-3],
                   [2.0e7, 1.18, 1.1e-3],
                   [2.0e7, 1.40, 0.8e-3],
                   [3.0e7, 1.37, 0.9e-3]]))

PVTG = PackedPVT(
    key=np.array([1.0e7, 2.0e7]),
    pos=np.array([0, 2, 4]),
    data=np.array([[1.0e-4, 1.0e-2, 1.4e-5],
                   [0.0, 1.1e-2, 1.5e-5],
                   [3.0e-4, 5.0e-3, 1.8e-5],
                   [0.0, 5.5e-3, 1.9e-5]]))

PVDO = np.array([[1.0e7, 1.20, 1.0e-3],
                 [2.0e7, 1.15, 1.1e-3],
                 [3.0e7, 1.10, 1.2e-3]])
PVDG = np.array([[1.0e7, 5.0e-3, 1.4e-5],
                 [2.0e7, 3.0e-3, 1.7e-5],
                 [3.0e7, 2.5e-3, 2.0e-5]])
PVTW = [[2.7680382e7, 1.029, 4.5396812e-10, 3.1e-4, 0.0]]
PVCDO = [[2.0e7, 1.20, 1.0e-9, 1.0e-3, 2.0e-10]]
ROCK = [[1.0132929e5, 4.3511321e-10]]


def check(failures, name, got, want) -> None:
    got = np.asarray(got, dtype=float).ravel()
    want = np.asarray(want, dtype=float).ravel()
    if got.shape != want.shape:
        failures.append(f"{name}: shape {got.shape} vs MRST {want.shape}")
        return
    if not np.allclose(got, want, rtol=TOLERANCE, atol=0, equal_nan=True):
        error = np.nanmax(np.abs(got - want))
        failures.append(f"{name}: max abs diff {error:.3e}")
        return
    print(f"  {name:18} {got.size} value(s)  ok")


def main() -> int:
    if not REFERENCE.exists():
        print(f"missing {REFERENCE.name}; "
              f"run dump_ad_props_assign_reference.m in MATLAB first")
        return 2
    ref = loadmat(REFERENCE)
    reg = Regions()
    failures = []

    print("PVTO (live oil, 'parallel' undersaturated branches)")
    f = assign_pvto({}, [PVTO], reg)
    flag = ref["PVTO_flag"].ravel().astype(bool)
    p, rs = ref["PVTO_p"].ravel(), ref["PVTO_rs"].ravel()
    check(failures, "PVTO bO", f["bO"][0](p, rs, flag), ref["PVTO_bO"])
    check(failures, "PVTO muO", f["muO"][0](p, rs, flag), ref["PVTO_muO"])
    check(failures, "PVTO rsSat",
          f["rsSat"][0](ref["PVTO_rsSat_p"].ravel()), ref["PVTO_rsSat"])
    check(failures, "PVTO pb",
          f["pb"][0](ref["PVTO_pb_rs"].ravel()), ref["PVTO_pb"])

    print("PVTG (wet gas, 'linshift' undersaturated branches)")
    g = assign_pvtg({}, [PVTG], reg)
    gflag = ref["PVTG_flag"].ravel().astype(bool)
    pg, rv = ref["PVTG_p"].ravel(), ref["PVTG_rv"].ravel()
    check(failures, "PVTG bG", g["bG"][0](pg, rv, gflag), ref["PVTG_bG"])
    check(failures, "PVTG muG", g["muG"][0](pg, rv, gflag), ref["PVTG_muG"])
    check(failures, "PVTG rvSat",
          g["rvSat"][0](ref["PVTG_rvSat_p"].ravel()), ref["PVTG_rvSat"])

    print("dead oil, dry gas, water, rock")
    d = assign_pvdo({}, [PVDO], reg)
    check(failures, "PVDO bO", d["bO"][0](ref["PVDO_p"].ravel()), ref["PVDO_bO"])
    check(failures, "PVDO muO",
          d["muO"][0](ref["PVDO_p"].ravel()), ref["PVDO_muO"])

    d = assign_pvdg({}, [PVDG], reg)
    check(failures, "PVDG bG", d["bG"][0](ref["PVDG_p"].ravel()), ref["PVDG_bG"])
    check(failures, "PVDG muG",
          d["muG"][0](ref["PVDG_p"].ravel()), ref["PVDG_muG"])

    w = assign_pvtw({}, PVTW, reg)
    check(failures, "PVTW bW", w["bW"][0](ref["PVTW_p"].ravel()), ref["PVTW_bW"])
    check(failures, "PVTW muW",
          w["muW"][0](ref["PVTW_p"].ravel()), ref["PVTW_muW"])

    c = assign_pvcdo({}, PVCDO, reg)
    check(failures, "PVCDO bO",
          c["bO"][0](ref["PVCDO_p"].ravel()), ref["PVCDO_bO"])
    check(failures, "PVCDO muO",
          c["muO"][0](ref["PVCDO_p"].ravel()), ref["PVCDO_muO"])

    r = assign_rock({}, ROCK, reg)
    check(failures, "ROCK pvMultR",
          r["pvMultR"](ref["ROCK_p"].ravel()), ref["ROCK_pvMultR"])

    print("densities")
    dens = assign_density({}, [[49.1, 64.79, 0.06054]], reg)
    for phase in ("rhoOS", "rhoWS", "rhoGS"):
        check(failures, f"DENSITY {phase}", dens[phase], ref[f"DENSITY_{phase}"])
    grav = assign_gravity({}, [[45.5, 1.0, 0.7773]], reg)
    for phase in ("rhoOS", "rhoWS", "rhoGS"):
        check(failures, f"GRAVITY {phase}", grav[phase], ref[f"GRAVITY_{phase}"])

    print("saturation functions")
    swof = np.array([[0.12, 0.00, 1.00, 0.0],
                     [0.30, 0.05, 0.60, 0.0],
                     [0.60, 0.30, 0.15, 0.0],
                     [0.90, 1.00, 0.00, 0.0]])
    sgof = np.array([[0.00, 0.00, 1.00, 0.0],
                     [0.20, 0.05, 0.55, 0.0],
                     [0.50, 0.40, 0.10, 0.0],
                     [0.88, 1.00, 0.00, 0.0]])

    s = assign_swof({}, [swof], reg)
    check(failures, "SWOF krW",
          s["krW"][0](ref["SWOF_sw"].ravel()), ref["SWOF_krW"])
    check(failures, "SWOF krOW",
          s["krOW"][0](ref["SWOF_so"].ravel()), ref["SWOF_krOW"])
    check(failures, "SWOF pts w", s["krPts"]["w"], ref["SWOF_pts_w"])
    check(failures, "SWOF pts ow", s["krPts"]["ow"], ref["SWOF_pts_ow"])
    if bool(ref["SWOF_hasPC"].ravel()[0]) != ("pcOW" in s):
        failures.append("SWOF: pcOW presence differs from MRST")

    s = assign_sgof(s, [sgof], reg)
    check(failures, "SGOF krG",
          s["krG"][0](ref["SGOF_sg"].ravel()), ref["SGOF_krG"])
    check(failures, "SGOF krOG",
          s["krOG"][0](ref["SGOF_so"].ravel()), ref["SGOF_krOG"])
    check(failures, "SGOF pts g", s["krPts"]["g"], ref["SGOF_pts_g"])
    check(failures, "SGOF pts og", s["krPts"]["og"], ref["SGOF_pts_og"])

    swof_pc = swof.copy()
    swof_pc[:, 3] = [3.0e4, 1.0e4, 2.0e3, 0.0]
    sp = assign_swof({}, [swof_pc], reg)
    check(failures, "SWOF pcOW",
          sp["pcOW"][0](ref["SWOF_pc_sw"].ravel()), ref["SWOF_pcOW"])

    print("EOR and thermal")
    # Imported by name rather than looked up dynamically: the parity auditor
    # credits a file as validated when a validator names it, and a dynamic
    # __import__ hides that coverage from it.
    from pymrst.modules.ad_props.props.assign_misc import assign_misc
    from pymrst.modules.ad_props.props.assign_plyads import assign_plyads
    from pymrst.modules.ad_props.props.assign_plyshear import assign_plyshear
    from pymrst.modules.ad_props.props.assign_plyvisc import assign_plyvisc
    from pymrst.modules.ad_props.props.assign_pmisc import assign_pmisc
    from pymrst.modules.ad_props.props.assign_specrock import assign_specrock
    from pymrst.modules.ad_props.props.assign_surfads import assign_surfads
    from pymrst.modules.ad_props.props.assign_surfcapd import assign_surfcapd
    from pymrst.modules.ad_props.props.assign_surfst import assign_surfst
    from pymrst.modules.ad_props.props.assign_surfvisc import assign_surfvisc

    for name, assign, table, field, probe_key, ref_key, listed in [
        ("PLYVISC", assign_plyvisc, [[0.0, 1.0], [1.0, 5.0], [3.0, 20.0]],
         "muWMult", "PLYVISC_c", "PLYVISC_muWMult", True),
        ("PLYADS", assign_plyads, [[0.0, 0.0], [1.0, 0.002], [3.0, 0.003]],
         "ads", "PLYADS_c", "PLYADS_ads", True),
        ("PLYSHEAR", assign_plyshear,
         [[0.0, 1.0], [1.0e-5, 0.9], [1.0e-4, 0.5]],
         "plyshearMult", "PLYSHEAR_v", "PLYSHEAR_mult", True),
        ("SURFST", assign_surfst, [[0.0, 0.05], [1.0, 0.001], [5.0, 1.0e-5]],
         "ift", "SURFST_c", "SURFST_ift", True),
        ("SURFVISC", assign_surfvisc,
         [[0.0, 5.0e-4], [1.0, 8.0e-4], [5.0, 2.0e-3]],
         "muWSft", "SURFVISC_c", "SURFVISC_muWSft", True),
        ("SURFADS", assign_surfads, [[0.0, 0.0], [1.0, 0.0005], [5.0, 0.001]],
         "surfads", "SURFADS_c", "SURFADS_ads", True),
        ("SURFCAPD", assign_surfcapd, [[-10.0, 0.0], [-5.0, 0.5], [-2.0, 1.0]],
         "miscfact", "SURFCAPD_nc", "SURFCAPD_miscfact", True),
        ("MISC", assign_misc, [[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]],
         "Ms", "MISC_sg", "MISC_Ms", False),
        ("PMISC", assign_pmisc, [[1.0e7, 0.0], [2.0e7, 0.5], [3.0e7, 1.0]],
         "Mp", "PMISC_p", "PMISC_Mp", False),
        ("SPECROCK", assign_specrock,
         [[300.0, 2.0e6], [350.0, 2.1e6], [400.0, 2.2e6]],
         "eR", "SPECROCK_T", "SPECROCK_eR", False),
    ]:
        out = assign({}, [np.array(table)], reg)[field]
        curve = out[0] if listed else out
        check(failures, name, curve(ref[probe_key].ravel()), ref[ref_key])

    from pymrst.modules.ad_props.props.assign_specheat import assign_specheat
    specheat = np.array([[300.0, 1000.0, 4180.0, 1900.0],
                         [350.0, 1100.0, 4200.0, 2000.0],
                         [400.0, 1200.0, 4260.0, 2100.0]])
    h = assign_specheat({}, [specheat], reg)
    for phase in ("uG", "uW", "uO"):
        check(failures, f"SPECHEAT {phase}",
              h[phase](ref["SPECHEAT_T"].ravel()), ref[f"SPECHEAT_{phase}"])

    from pymrst.modules.ad_props.props.assign_plymax import assign_plymax
    from pymrst.modules.ad_props.props.assign_plyrock import assign_plyrock
    from pymrst.modules.ad_props.props.assign_plmixpar import assign_plmixpar
    from pymrst.modules.ad_props.props.assign_shrate import assign_shrate
    from pymrst.modules.ad_props.props.assign_surfrock import assign_surfrock
    from pymrst.modules.ad_props.props.assign_tlmixpar import assign_tlmixpar
    from pymrst.modules.ad_props.props.assign_viscref import assign_viscref

    pr = assign_plyrock({}, [[0.05, 1.3, 2650.0, 1.0, 2.5e-5]], reg)
    check(failures, "PLYROCK",
          [pr["dps"], pr["rrf"], pr["rhoR"], pr["adsInx"], pr["adsMax"]],
          ref["PLYROCK"])
    sr = assign_surfrock({}, [[2.0, 2650.0]], reg)
    check(failures, "SURFROCK", [sr["adsInxSft"], sr["rhoRSft"]],
          ref["SURFROCK"])
    check(failures, "PLYMAX", assign_plymax({}, [3.0, 0.0])["cpmax"],
          ref["PLYMAX_cpmax"])
    check(failures, "SHRATE", assign_shrate({}, [4.8])["shrate"],
          ref["SHRATE_shrate"])
    check(failures, "PLMIXPAR", assign_plmixpar({}, [1.0])["mixPar"],
          ref["PLMIXPAR_mixPar"])
    tl = assign_tlmixpar({}, [0.7, 0.3])
    check(failures, "TLMIXPAR", [tl["mixPar"], tl["mixParRho"]],
          ref["TLMIXPAR"])
    tl1 = assign_tlmixpar({}, [0.7])
    check(failures, "TLMIXPAR one", [tl1["mixPar"], tl1["mixParRho"]],
          ref["TLMIXPAR_one"])
    # VISCREF is a known divergence, not a match: MRST guards on
    # numel(reg.SATNUM) == 1, so an unpartitioned deck (SATNUM = []) falls to
    # val([], :) and MRST returns an empty refVisc. Assert that is still what
    # MRST does, and that this port returns the record instead.
    assert np.asarray(ref["VISCREF"]).size == 0, (
        "MRST's VISCREF defect appears to be fixed; revisit assign_viscref")
    ours = assign_viscref({}, [[2.0e7, 50.0]], reg)["refVisc"]
    if not np.allclose(ours, [2.0e7, 50.0]):
        failures.append(f"VISCREF: expected the record, got {ours}")
    else:
        print("  VISCREF            divergence by design: MRST returns empty")

    print("  WATERVIST / OILVIST  no MRST reference: both index reg.SATKINX,")
    print("                       which handleRegions never creates, so they")
    print("                       raise 'Unrecognized field name' in MRST.")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("All assign* values match MRST.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
