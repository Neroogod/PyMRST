"""Validate corrected ``updateStateBlackOilGeneric`` behavior against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.utils import update_state_black_oil_generic
from pymrst.modules.ad_core.statefunctions.state_function import State


HERE = Path(__file__).resolve().parent


class Facility:
    n_wells = 0

    @staticmethod
    def update_switched_controls(state, rate_names):
        return False


def main() -> int:
    path = HERE / 'ref_ad_blackoil_update_state.mat'
    if not path.exists():
        print(f'{path.name} missing (run the matching MATLAB dump)')
        return 1
    ref = loadmat(path, squeeze_me=True)
    fluid = SimpleNamespace(rs_sat=lambda p: 10+0.01*p)
    model = SimpleNamespace(
        G=SimpleNamespace(cells=SimpleNamespace(num=3)),
        facility=Facility(), n_phases=3, dp_max_rel=.2, ds_max=.2,
        disgas=True, vapoil=False, raw_fluid=fluid,
        _status_masks=lambda status: (status == 1, status == 2, status == 3),
        _ms_well_indices=lambda: [],
    )
    state = State(
        pressure=np.array([100.,200.,300.]),
        sW=np.array([.2,.3,.25]), sO=np.array([.8,.5,.5]),
        sG=np.array([0.,.2,.25]), rs=np.array([5.,12.,13.]),
        status=np.array([1,3,3]), qWs=np.zeros(0), qOs=np.zeros(0),
        qGs=np.zeros(0), bhp=np.zeros(0),
    )
    increment = np.concatenate([
        [50.,-60.,30.], [.3,-.1,.05], [2.,-.4,.1],
    ])
    actual = update_state_black_oil_generic(model, state, increment)
    checks = {
        'pressure': (actual['pressure'], ref['pressure']),
        'saturations': (np.column_stack([actual['sW'],actual['sO'],actual['sG']]), ref['s']),
        'rs': (actual['rs'], ref['rs']),
        'status': (actual['status'], ref['status']),
    }
    failed = False
    for name, (got, expected) in checks.items():
        passed = np.allclose(got, expected, rtol=1e-13, atol=1e-13)
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
        failed |= not passed
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
