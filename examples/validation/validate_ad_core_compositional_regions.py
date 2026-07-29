"""Validate ``getInitializationRegionsCompositional`` against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

from pymrst.modules.ad_core.initialization import get_initialization_regions_compositional
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.models.eos.equation_of_state_model import EquationOfStateModel


def main() -> int:
    here = Path(__file__).parent
    ref = scipy.io.loadmat(here / "ref_ad_core_compositional_regions.mat", squeeze_me=True)
    mix = Mixture(
        names=("Methane", "n-Butane", "n-Decane"),
        Tcrit=np.atleast_1d(ref["Tcrit"]), Pcrit=np.atleast_1d(ref["Pcrit"]),
        Vcrit=np.atleast_1d(ref["Vcrit"]), acentric=np.atleast_1d(ref["acf"]),
        molarMass=np.atleast_1d(ref["mw"]),
    )
    model = SimpleNamespace(
        G=SimpleNamespace(cells=SimpleNamespace(num=4)),
        fluid=SimpleNamespace(phases=("oil", "gas"), swcon=0.0),
        EOSModel=EquationOfStateModel(None, mix, "PR"),
        n_phases=2, getPhaseNames=lambda: ("O", "G"),
        getNumberOfPhases=lambda: 2,
    )
    region = get_initialization_regions_compositional(
        model, np.atleast_1d(ref["contacts"]),
        cells=np.atleast_1d(ref["cells"]).astype(int) - 1,
        T=350.0, x=np.atleast_1d(ref["x"]), y=np.atleast_1d(ref["y"]),
        datum_depth=float(ref["datum_depth"]),
        datum_pressure=float(ref["datum_pressure"]),
    )
    p, z = np.atleast_1d(ref["p"]), np.atleast_1d(ref["z"])
    checks = {
        "liquid density": (region.rho[0](p, z), ref["rho_l"]),
        "vapor density": (region.rho[1](p, z), ref["rho_v"]),
        "s_min": (region.s_min, ref["s_min"]),
        "s_max": (region.s_max, ref["s_max"]),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        ok = np.allclose(actual, expected, rtol=2e-10, atol=2e-8)
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    ref_index = int(ref["reference_index"]) - 1
    ok = region.reference_index == ref_index
    print(f"{'ok  ' if ok else 'FAIL'} reference index: {region.reference_index}")
    failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
