"""Direct MATLAB/MRST validation of getInitializationRegionsDeck."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_core.initialization import get_initialization_regions_deck
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params.rock import make_rock


def main() -> int:
    ref = loadmat(Path(__file__).with_name("ref_ad_core_deck_regions.mat"))
    G = compute_geometry(cart_grid([6, 1]))
    rock = make_rock(G, 1, 0.2)
    fluid = init_simple_adi_fluid(phases="WO", n=[2, 2], mu=[1, 1], rho=[1000, 800])
    model = GenericBlackOilModel(G, rock, fluid, gas=False).validate_model()
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
    regions = get_initialization_regions_deck(model, deck)
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
