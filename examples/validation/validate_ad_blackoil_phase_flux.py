"""Validate black-oil phase flux/property utility tuples against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.utils import (
    get_flux_and_props_gas_bo, get_flux_and_props_oil_bo,
    get_flux_and_props_water_bo,
)
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.params.rock import make_rock


HERE = Path(__file__).resolve().parent


def main() -> int:
    path = HERE / "ref_ad_blackoil_phase_flux.mat"
    if not path.exists():
        print(f"{path.name} missing (run the matching MATLAB dump)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    grid = compute_geometry(cart_grid([3, 1, 1], [3, 1, 1]))
    rock = make_rock(grid, 1.0, 0.2)
    fluid = SimpleNamespace(
        rhoS=np.array([1000., 700., 2.]),
        pc_ow=lambda s: 2*s, pc_og=lambda s: 3*s,
        b_w=lambda p: 1+.01*p, mu_w=lambda p: 2+.001*p,
        b_o=lambda p,r,s: 1.2+.02*p+.03*r+.04*s,
        mu_o=lambda p,r,s: 3+.002*p+.01*r,
        b_g=lambda p,r,s: 4+.04*p+.05*r+.06*s,
        mu_g=lambda p,r,s: .5+.003*p+.02*r,
    )
    model = SimpleNamespace(G=grid, g=grid, rock=rock, raw_fluid=fluid,
                            fluid=fluid, operators=setup_operators_tpfa(grid, rock),
                            disgas=True, vapoil=True)
    water = get_flux_and_props_water_bo(
        model, ref['p'], ref['sw'], ref['krw'], ref['T'], ref['gdz'])
    oil = get_flux_and_props_oil_bo(
        model, ref['p'], ref['so'], ref['kro'], ref['T'], ref['gdz'],
        ref['rs'], ref['gas_sat'].astype(bool))
    gas = get_flux_and_props_gas_bo(
        model, ref['p'], ref['sg'], ref['krg'], ref['T'], ref['gdz'],
        ref['rv'], ref['oil_sat'].astype(bool))
    failed = False
    for phase, result in zip('WOG', (water, oil, gas)):
        for stem, actual in zip(('v','b','mob','rho','p','up','dp','mu'), result):
            key = stem + phase
            expected = np.asarray(ref[key]).squeeze()
            actual = np.asarray(actual).squeeze()
            passed = actual.shape == expected.shape and np.allclose(
                actual, expected, rtol=1e-12, atol=1e-12)
            print(f"{'ok  ' if passed else 'FAIL'} {key}")
            failed |= not passed
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
