"""Validate getWellPolymer/getPolymerShearMultiplier against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.utils import (
    get_polymer_shear_multiplier, get_well_polymer,
)


HERE = Path(__file__).resolve().parent


def main() -> int:
    path = HERE/'ref_ad_blackoil_polymer_utils.mat'
    if not path.exists():
        print(f'{path.name} missing (run the matching MATLAB dump)')
        return 1
    ref = loadmat(path, squeeze_me=True)
    wells = [
        SimpleNamespace(sign=1,cells=np.array([0,1]),cp=2.5,compi=[1,0,0]),
        SimpleNamespace(sign=-1,cells=np.array([2]),cp=None,compi=[0,1,0]),
        SimpleNamespace(sign=1,cells=np.array([3,4,5]),cp=None,compi=[0,0,1]),
    ]
    wpoly, wc, indices = get_well_polymer(wells)
    fluid = SimpleNamespace(plyshear_mult=lambda velocity: 1+0.5*velocity)
    mult, velocity, report = get_polymer_shear_multiplier(
        SimpleNamespace(fluid=fluid), ref['VW0'], ref['muWmult'])
    checks = {
        'well polymer': (wpoly, ref['wPoly']),
        'connection polymer': (wc, ref['wciPoly']),
        'water injector perforations': (indices, ref['iInxW0']),
        'shear multiplier': (mult, ref['mult']),
        'shear velocity': (velocity, ref['VW']),
        'iterations': (report['Iterations'], ref['iterations']),
        'converged': (report['Converged'], ref['converged']),
    }
    failed = False
    for name, (actual, expected) in checks.items():
        passed = np.asarray(actual).shape == np.asarray(expected).shape and np.allclose(
            actual, expected, rtol=1e-12, atol=1e-14)
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
        failed |= not passed
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
