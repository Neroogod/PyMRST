"""Phase 363: 1D Buckley-Leverett AD profile against MRST (ad-core example).

Ports and pins MRST's introductory AD example `adBuckleyLeverett1D`. The PyMRST
example (`ad_core/examples/ad_buckley_leverett_1d.py`) runs the same
incompressible 1D waterflood - flux BC in, pressure BC out, one pore volume
injected in 25 steps - and this compares its final water-saturation profile,
cell for cell, against MRST's own (`dump_phase363_reference.m`).

The two codes assemble the flux and step the nonlinear solver independently, so
agreement across all 50 cells is a real end-to-end check of the BC-driven
transient, not just a spot value.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.examples.ad_buckley_leverett_1d import (
    run_ad_buckley_leverett_1d,
)

REF = Path(__file__).with_name("ref_p363_bl.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase363_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    sw_ref = ref["sW"].ravel()
    n = int(ref["n"].ravel()[0])

    _, states = run_ad_buckley_leverett_1d(n=n)
    sw = np.asarray(states[-1]["sW"]).ravel()

    failed = 0
    print(f"Buckley-Leverett water saturation after {n} steps (1 PVI) vs MRST:")
    failed += _check("same number of cells", sw.size == sw_ref.size,
                     f"{sw.size} vs {sw_ref.size}")
    err = np.max(np.abs(sw - sw_ref))
    failed += _check("full 50-cell profile matches", err < 1e-6,
                     f"max|dsW| = {err:.3e}")
    failed += _check(f"inlet saturation ({sw[0]:.4f})",
                     np.isclose(sw[0], sw_ref[0], atol=1e-6))
    failed += _check(f"outlet (breakthrough) saturation ({sw[-1]:.4f})",
                     np.isclose(sw[-1], sw_ref[-1], atol=1e-6))

    print("\nBuckley-Leverett profile matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
