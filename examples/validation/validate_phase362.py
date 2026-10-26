"""Phase 362: AD-side upscaling against MRST (AC25).

AC25 ports `upscaleModelTPFA`, `upscaleState` and `upscaleSchedule` under
`ad_core/upscale/`. `dump_phase362_reference.m` runs MRST's own routines on a
deterministic 6x6 fine model coarsened to 3x3, and dumps the coarse rock (perm,
poro) and the coarse state (saturation, pressure). Here PyMRST rebuilds the
identical fine model and partition and must reproduce them number for number:

- coarse **rock**: harmonic-mean permeability and pore-volume-conserving porosity
  per block (`upscaleModelTPFA`),
- coarse **state**: pore-volume-weighted saturation and pressure (`upscaleState`).

The partition is compared first, to confirm both codes number the coarse blocks
the same way (so the coarse arrays line up cell for cell).
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
)
from pymrst.modules.ad_core.upscale.upscale_model_tpfa import upscale_model_tpfa_ad as upscale_model_tpfa
from pymrst.modules.ad_core.upscale.upscale_state import upscale_state_ad as upscale_state
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.modules.upscaling.coarsegrid.partition_cart_grid import (
    partition_cart_grid,
)
from pymrst.params.rock import make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

REF = Path(__file__).with_name("ref_p362_upscale.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase362_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    part_mrst = ref["partition"].ravel().astype(np.int64) - 1     # 1- -> 0-based
    perm = ref["perm"].ravel()
    poro = ref["poro"].ravel()
    sW = ref["sW"].ravel()

    nc = perm.size
    g = compute_geometry(cart_grid([6, 6, 1], [600.0, 600.0, 10.0]))
    rock = make_rock(g, perm.reshape(-1, 1), poro.reshape(-1, 1))
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 3.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[0.0, 0.0], p_ref=200 * BARSA)
    model = GenericBlackOilModelAD(g, rock, fluid, [])

    failed = 0
    part = partition_cart_grid([6, 6, 1], [3, 3, 1])
    print("Partition numbering:")
    failed += _check("PyMRST partition matches MRST partitionUI",
                     np.array_equal(part, part_mrst))

    coarse = upscale_model_tpfa(model, part)

    print("\nCoarse rock (upscaleModelTPFA) vs MRST:")
    perm_c = np.asarray(coarse.rock.perm).ravel()
    err_k = np.max(np.abs(perm_c - ref["perm_c"].ravel()))
    failed += _check("harmonic-mean permeability matches",
                     err_k < 1e-6 * ref["perm_c"].ravel().max(),
                     f"max|dk| = {err_k:.3e}")
    err_phi = np.max(np.abs(np.asarray(coarse.rock.poro).ravel()
                            - ref["poro_c"].ravel()))
    failed += _check("pore-volume-weighted porosity matches", err_phi < 1e-12,
                     f"max|dphi| = {err_phi:.3e}")

    print("\nCoarse state (upscaleState) vs MRST:")
    i = np.arange(1, nc + 1)
    state = {"pressure": (200.0 + i) * BARSA,
             "s": np.column_stack([sW, 1.0 - sW])}
    cstate = upscale_state(coarse, model, state)
    err_s = np.max(np.abs(cstate["s"] - ref["s_c"]))
    failed += _check("pv-weighted saturation matches", err_s < 1e-12,
                     f"max|ds| = {err_s:.3e}")
    err_p = np.max(np.abs(cstate["pressure"] - ref["p_c"].ravel()))
    failed += _check("pv-weighted pressure matches",
                     err_p < 1e-6 * ref["p_c"].ravel().max(),
                     f"max|dp| = {err_p:.3e}")

    print("\nAD-side upscaling matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
