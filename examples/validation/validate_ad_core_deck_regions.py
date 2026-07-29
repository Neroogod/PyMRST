"""Direct MATLAB/MRST validation of getInitializationRegionsDeck."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.initialization import getInitializationRegionsDeck


def main() -> int:
    ref = loadmat(Path(__file__).with_name("ref_ad_core_deck_regions.mat"))
    model = SimpleNamespace(
        G=SimpleNamespace(cells=SimpleNamespace(num=6, indexMap=np.arange(6))),
        fluid=SimpleNamespace(rho_s=np.array([1000.0, 800.0])),
        raw_fluid=SimpleNamespace(swcon=0.0),
        n_phases=2, water=True, oil=True, gas=False,
    )
    deck = {
        "SOLUTION": {"EQUIL": np.array([
            [1000, 1e7, 1100, 5, 900, 0, 0, 0, 0, 0],
            [1200, 1.2e7, 1300, 7, 1000, 0, 0, 0, 0, 0],
        ])},
        "REGIONS": {
            "EQLNUM": [1, 1, 1, 2, 2, 2],
            "SATNUM": [1, 1, 2, 1, 1, 1],
            "PVTNUM": [1, 2, 1, 1, 1, 2],
        },
    }
    regions = getInitializationRegionsDeck(model, deck)
    failed = 0
    print("Deck initialization regions vs MATLAB/MRST:")
    for i, region in enumerate(regions):
        cells = np.asarray(ref["cells"][i, 0]).ravel().astype(int) - 1
        contacts = np.asarray(ref["contacts"][i, 0]).ravel()
        contacts_pc = np.asarray(ref["contacts_pc"][i, 0]).ravel()
        checks = [
            np.array_equal(region.cells, cells),
            np.allclose(region.contacts, contacts),
            np.allclose(region.contacts_pc, contacts_pc),
            np.isclose(region.datum_depth, ref["datum_depth"][i, 0]),
            np.isclose(region.datum_pressure, ref["datum_pressure"][i, 0]),
        ]
        ok = all(checks)
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} region {i + 1}")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
