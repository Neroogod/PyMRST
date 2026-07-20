"""Phase 65: MRST run2DCase oil-water fixed-stress branch."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case

HERE = Path(__file__).parent
REF = HERE / "ref_p65_run2d_oilwater_fixedstress.mat"


def _arr(ref, key):
    return np.asarray(ref[key], dtype=float)


def compare(name, py, ml, rtol, atol=0.0) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ml = np.asarray(ml, dtype=float).ravel()
    scale = max(float(np.max(np.abs(ml))), 1.0)
    diff = float(np.max(np.abs(py - ml)))
    ok = diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        raise SystemExit(
            f"Missing {REF.name}. Run dump_phase65_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    nx, ny = np.asarray(ref["cartDim"], dtype=int).ravel()
    L = tuple(np.asarray(ref["L"], dtype=float).ravel())
    dt = np.asarray(ref["dt"], dtype=float).ravel()

    result = run_2d_case(
        fluid_model="oil water",
        method="fixed stress splitting",
        nx=int(nx),
        ny=int(ny),
        physdims=L,
        nsteps=dt.size,
        dt=dt,
        p_ref=float(ref["pRef"]),
    )
    assert all(r["converged"] for r in result.reports), result.reports
    states = result.states[1:]
    diagnostics = result.diagnostics[1:]

    p = np.vstack([st["pressure"] for st in states])
    sw = np.vstack([st["sW"] for st in states])
    xd = np.vstack([st["xd"] for st in states])
    u = np.vstack([st["u"] for st in states])
    vdiv = np.vstack([d["volumetric_strain"] for d in diagnostics])
    strain = np.stack([d["strain"] for d in diagnostics], axis=0)
    stress = np.stack([d["effective_stress"] for d in diagnostics], axis=0)
    qws = np.array([st["qWs"][0] for st in states])
    qos = np.array([st["qOs"][0] for st in states])
    bhp = np.array([st["bhp"][0] for st in states])

    failures = 0
    print("Phase 65: run2DCase oil-water fixed-stress vs MATLAB MRST")
    failures += compare("pressure", p, _arr(ref, "p_hist"), rtol=2e-6)
    failures += compare("water saturation", sw, _arr(ref, "sw_hist"), rtol=2e-6)
    failures += compare("free displacement xd", xd, _arr(ref, "xd_hist"),
                        rtol=2e-6)
    failures += compare("full displacement u", u, _arr(ref, "u_hist"),
                        rtol=2e-6)
    failures += compare("volumetric strain", vdiv, _arr(ref, "vdiv_hist"),
                        rtol=2e-6)
    failures += compare("strain", strain, _arr(ref, "strain_hist"),
                        rtol=2e-6)
    failures += compare("effective stress", stress, _arr(ref, "stress_hist"),
                        rtol=2e-6, atol=1.0)
    failures += compare("well qWs", qws, _arr(ref, "qws_hist"), rtol=2e-6)
    failures += compare("well qOs", qos, _arr(ref, "qos_hist"), rtol=2e-6)
    failures += compare("well bhp", bhp, _arr(ref, "bhp_hist"), rtol=2e-6)

    if failures:
        print(f"\nPhase 65 failed ({failures} checks).")
    else:
        print("\nPyMRST matches MRST run2DCase oil-water fixed-stress.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
