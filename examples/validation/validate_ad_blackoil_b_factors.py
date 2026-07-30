"""Validate get_b_o_bo/get_b_g_bo dispatch against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.utils import get_b_g_bo, get_b_o_bo


HERE = Path(__file__).resolve().parent


class Fluid:
    @staticmethod
    def b_o_dead(p):
        return 2 + 0.1*p

    @staticmethod
    def b_o_live(p, rs, saturated):
        return 2 + 0.1*p + 0.2*rs + 0.3*saturated

    @staticmethod
    def b_g_dry(p):
        return 3 + 0.4*p

    @staticmethod
    def b_g_wet(p, rv, liquid):
        return 3 + 0.4*p + 0.5*rv + 0.6*liquid


def main() -> int:
    path = HERE / "ref_ad_blackoil_b_factors.mat"
    if not path.exists():
        print(f"{path.name} missing (run the matching MATLAB dump)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    p, rs, rv = ref["p"], ref["rs"], ref["rv"]
    gas = ref["gas_present"].astype(bool)
    oil = ref["oil_present"].astype(bool)
    fluid = SimpleNamespace(b_o=Fluid.b_o_dead, b_g=Fluid.b_g_dry)
    model = SimpleNamespace(fluid=fluid, disgas=False, vapoil=False)
    actual = {
        "bo_dead": get_b_o_bo(model, p, rs, gas),
        "bg_dry": get_b_g_bo(model, p, rv, oil),
    }
    model.disgas = True
    model.vapoil = True
    fluid.b_o = Fluid.b_o_live
    fluid.b_g = Fluid.b_g_wet
    actual["bo_live"] = get_b_o_bo(model, p, rs, gas)
    actual["bg_wet"] = get_b_g_bo(model, p, rv, oil)

    failed = False
    for name, value in actual.items():
        passed = np.allclose(value, ref[name], rtol=1e-14, atol=1e-14)
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
        failed |= not passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
