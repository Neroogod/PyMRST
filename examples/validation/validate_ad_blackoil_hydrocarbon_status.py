"""Validate calculate_hydrocarbons_from_status_bo against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.utils import calculate_hydrocarbons_from_status_bo


HERE = Path(__file__).resolve().parent


def main() -> int:
    path = HERE / "ref_ad_blackoil_hydrocarbon_status.mat"
    if not path.exists():
        print(f"{path.name} missing (run the matching MATLAB dump)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    fluid = SimpleNamespace(
        rs_sat=lambda p: 10 + 0.1*p,
        rv_sat=lambda p: 0.2 + 0.01*p,
    )
    args = (fluid, ref["encoded"], ref["oil_space"], ref["x"],
            ref["rs_input"], ref["rv_input"], ref["pressure"])
    live = calculate_hydrocarbons_from_status_bo(*args, True, True)
    dead = calculate_hydrocarbons_from_status_bo(*args, False, False)
    names = ("sG", "rs", "rv", "rsSat", "rvSat")
    failed = False
    for prefix, result in (("", live), ("_dead", dead)):
        for name, actual in zip(names, result):
            key = name + prefix
            passed = np.allclose(actual, ref[key], rtol=1e-14, atol=1e-14)
            print(f"{'ok  ' if passed else 'FAIL'} {key}")
            failed |= not passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
