"""Phase 111: SPE1 ``convertRSRVtoKValue`` tables vs MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import setup_spe1_tutorial
from pymrst.modules.compositional.fluids.blackoil import convert_rs_rv_to_k_value
from pymrst.utils.units import BARSA

HERE = Path(__file__).parent
REF = HERE / "ref_p111_spe1_kvalue.mat"


def compare(name, py, ref, rtol=1e-8, atol=1e-12, scale=1.0, unit="") -> int:
    py = np.asarray(py, dtype=float)
    ref = np.asarray(ref, dtype=float)
    mask = np.isfinite(py) & np.isfinite(ref)
    if py.shape != ref.shape:
        diff = np.inf
        ok = False
    elif np.any(mask):
        diff = float(np.max(np.abs(py[mask] - ref[mask])))
        ref_scale = max(float(np.max(np.abs(ref[mask]))), 1.0)
        ok = diff <= atol + rtol * ref_scale
    else:
        diff = 0.0
        ok = True
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff / scale:.3e} {unit} "
        f"(rtol {rtol:.1e}, atol {atol / scale:.1e} {unit})"
    )
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p111_spe1_kvalue.mat missing (run dump_phase111_reference.m)")
        return 1

    ref = scipy.io.loadmat(REF, squeeze_me=True)
    _, model, _ = setup_spe1_tutorial()
    props = convert_rs_rv_to_k_value(
        model,
        np_=12,
        nz=16,
        interpolation="linear",
        min_pressure=300.0 * BARSA,
        max_pressure=500.0 * BARSA,
    )
    kv = props.k_values
    pr = props.properties

    print("\nPhase 111: SPE1 convertRSRVtoKValue")
    failures = 0
    failures += compare("K_o", kv.K_o, ref["k_K_o"], rtol=2e-5, atol=1e-10)
    failures += compare("K_g", kv.K_g, ref["k_K_g"], rtol=2e-5, atol=1e-10)
    bounded_free_gas = np.isfinite(kv.freeGas) & (kv.freeGas >= 0.0) & (kv.freeGas <= 1.0)
    failures += compare("freeGas bounded diagnostic", bounded_free_gas, np.ones_like(kv.freeGas, dtype=bool), rtol=0, atol=0)
    failures += compare("k z_g grid", kv.z_g, ref["k_z_g"], rtol=0, atol=0)
    failures += compare("k z_o grid", kv.z_o, ref["k_z_o"], rtol=0, atol=0)
    failures += compare("k pressure grid", kv.pressure, ref["k_pressure"], rtol=0, atol=1e-9 * BARSA, scale=BARSA, unit="bar")
    failures += compare("rhoO", pr.rhoO, ref["p_rhoO"], rtol=2e-5, atol=1e-8)
    failures += compare("rhoG", pr.rhoG, ref["p_rhoG"], rtol=2e-5, atol=1e-8)
    failures += compare("muO", pr.muO, ref["p_muO"], rtol=2e-5, atol=1e-12)
    failures += compare("muG", pr.muG, ref["p_muG"], rtol=2e-5, atol=1e-12)
    failures += compare("property pressure grid", pr.pressure, ref["p_pressure"], rtol=0, atol=1e-9 * BARSA, scale=BARSA, unit="bar")
    failures += compare("property x_o grid", pr.x_o, ref["p_x_o"], rtol=0, atol=0)
    failures += compare("property y_g grid", pr.y_g, ref["p_y_g"], rtol=0, atol=0)

    if failures:
        print(f"\nSPE1 K-value conversion validation failed ({failures} checks).")
    else:
        print("\nSPE1 K-value conversion tables match MATLAB.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
