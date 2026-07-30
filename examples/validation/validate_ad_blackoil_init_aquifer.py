"""Validate corrected-intent ``computeInitAquifer`` equilibrium behavior."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.utils import compute_init_aquifer
from pymrst.modules.ad_core.models.aquifers.aquifer import AquiferConnections


HERE = Path(__file__).resolve().parent


def main() -> int:
    path = HERE / 'ref_ad_blackoil_init_aquifer.mat'
    if not path.exists():
        print(f'{path.name} missing (run the matching MATLAB dump)')
        return 1
    ref = loadmat(path, squeeze_me=True)
    aquid = ref['aquid'].astype(int)-1
    conn = AquiferConnections(
        aquid=aquid, cells=np.arange(aquid.size), j=ref['j'],
        c=np.full(aquid.size, 1e-9), alpha=ref['alpha'],
        depth_conn=ref['depth_conn'], depth_aq=ref['depth_aq'],
        c_aq=np.full(int(aquid.max())+1, 1e-9),
    )
    aquifer = SimpleNamespace(
        connections=conn, state_arg_keys=('aquifer_p','aquifer_V'),
        _water_props=lambda model, state: (ref['pW'], ref['bW']),
        pressures=np.zeros(2), volumes=np.zeros(2),
    )
    model = SimpleNamespace(
        aquifer=aquifer,
        raw_fluid=SimpleNamespace(rhoS=np.array([ref['rhoWS'],700.,2.])),
        fluid=SimpleNamespace(rho_s=np.array([ref['rhoWS'],700.,2.])),
        gravity=float(ref['gravity']),
    )
    state = compute_init_aquifer(model, {}, ref['volumes'])
    checks = {
        'aquifer pressure': (state['aquifer_p'], ref['p_aq']),
        'aquifer volume': (state['aquifer_V'], ref['volumes']),
    }
    failed = False
    for name, (actual, expected) in checks.items():
        passed = np.allclose(actual, expected, rtol=1e-13, atol=1e-10)
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
        failed |= not passed
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
