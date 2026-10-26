"""Validate MRST's black-oil utilities `getbO_BO`, `getbG_BO`,
`getCapillaryPressureBO` and `getCellStatusVO` on SPE9's deck fluid.

The reference is `dump_ad_blackoil_bo_utils_reference.m` (MRST 2024a):
live-oil `bO` and dry-gas `bG` through a `ThreePhaseBlackOilModel` with
dissolved gas; both capillary pressures, and the empty-saturation case;
the cell status masks for water-only, oil-only, gas-only, three-phase and
near-boundary cells under all four `vapoil`/`disgas` combinations, and with
a given status. Values to `RTOL`, masks exactly.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
    ThreePhaseBlackOilModel,
)
from pymrst.modules.ad_blackoil.utils.get_b_g_bo import getbG_BO
from pymrst.modules.ad_blackoil.utils.get_b_o_bo import getbO_BO
from pymrst.modules.ad_blackoil.utils.get_capillary_pressure_bo import (
    getCapillaryPressureBO,
)
from pymrst.modules.ad_blackoil.utils.get_cell_status_vo import getCellStatusVO
from pymrst.modules.test_suite.dataset_functions.spe9 import setup_spe9
from pymrst.utils.adi import value

HERE = Path(__file__).resolve().parent
RTOL = 1e-14


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(value(a), dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def main() -> int:
    ref = loadmat(HERE/"ref_ad_blackoil_bo_utils.mat", squeeze_me=True)
    G, rock, fluid, _, _ = setup_spe9()
    model = ThreePhaseBlackOilModel(G, rock, fluid)
    model.disgas = True
    p, rs, sat = ref["p"], ref["rs"], ref["isSat"].astype(bool)
    ok = True
    err = _rel(getbO_BO(model, p, rs, sat), ref["bO_live"])
    ok &= report(err < RTOL, f"getbO_BO, live oil: rel. diff {err:.1e}")
    err = _rel(getbG_BO(model, p, rs, sat), ref["bG_dry"])
    ok &= report(err < RTOL, f"getbG_BO, dry gas: rel. diff {err:.1e}")
    pcow, pcog = getCapillaryPressureBO(fluid, ref["sW"], ref["sG"])
    for name, actual in (("pcOW", pcow), ("pcOG", pcog)):
        err = _rel(actual, ref[name])
        ok &= report(err < RTOL, f"getCapillaryPressureBO {name}: rel. diff {err:.1e}")
    empty = getCapillaryPressureBO(fluid, np.array([]), np.array([]))
    ok &= report(empty == (float(ref["pcOW_empty"]), float(ref["pcOG_empty"])),
                 f"getCapillaryPressureBO with empty saturations: {empty}")
    s_o, s_w, s_g = ref["sO"], ref["sWs"], ref["sGs"]
    columns = []
    for vapoil in (False, True):
        for disgas in (False, True):
            st = getCellStatusVO(s_o, s_w, s_g, vapoil=vapoil, disgas=disgas)
            columns += [np.broadcast_to(np.asarray(v, dtype=float), s_o.shape) for v in st]
    ok &= report(np.array_equal(np.column_stack(columns), ref["status"]),
                 "getCellStatusVO masks under all vapoil/disgas combinations")
    given = getCellStatusVO(s_o, s_w, s_g, status=np.array([1, 2, 3, 1, 2, 3, 0]),
                            vapoil=True, disgas=True)
    ok &= report(np.array_equal(np.column_stack(given).astype(float), ref["given"]),
                 "getCellStatusVO with a given status")
    print("\nThe black-oil utilities match MRST" if ok
          else "\nThe black-oil utilities do NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
