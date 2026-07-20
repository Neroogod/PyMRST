"""Compare PyMRST's Peng-Robinson flash against MATLAB's compositional module.

A methane / n-butane / n-decane mixture flashed across a pressure sweep at
350 K: Wilson K-values, vapour fraction, phase compositions, and the
liquid/vapour compressibility factors.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.compositional import Mixture, flash, wilson_k

HERE = Path(__file__).parent


def main() -> int:
    path = HERE / "ref_p13_flash.mat"
    if not path.exists():
        print("ref_p13_flash.mat missing (run dump_phase13_reference.m)")
        return 1
    ref = scipy.io.loadmat(path, squeeze_me=True)
    failed = 0

    def chk(name, a, b, tol):
        nonlocal failed
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
        diff = float(np.abs(a - b).max()) if a.shape == b.shape else np.inf
        ok = diff <= tol
        print(f"  {'ok  ' if ok else 'FAIL'} {name:26s} max abs diff = "
              f"{diff:.2e}  (tol {tol:.0e})")
        failed += 0 if ok else 1

    mix = Mixture(
        names=("Methane", "n-Butane", "n-Decane"),
        Tcrit=np.asarray(ref["Tcrit"], float).ravel(),
        Pcrit=np.asarray(ref["Pcrit"], float).ravel(),
        Vcrit=np.asarray(ref["Vcrit"], float).ravel(),
        acentric=np.asarray(ref["acf"], float).ravel(),
        molarMass=np.asarray(ref["mw"], float).ravel(),
    )
    p = np.asarray(ref["p"], float).ravel()
    T = np.asarray(ref["T"], float).ravel()
    z = np.atleast_2d(np.asarray(ref["z"], float))

    chk("Wilson K-values", wilson_k(mix, p, T),
        np.atleast_2d(ref["K_wilson"]), 1e-10)

    r = flash(mix, p, T, z)
    chk("vapour fraction V", r.V, ref["V"], 1e-8)
    chk("liquid composition x", r.x, np.atleast_2d(ref["x"]), 1e-8)
    chk("vapour composition y", r.y, np.atleast_2d(ref["y"]), 1e-8)
    chk("liquid Z-factor", r.Z_L, ref["Zl"], 1e-8)
    chk("vapour Z-factor", r.Z_V, ref["Zv"], 1e-8)

    print("\nPeng-Robinson flash matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
