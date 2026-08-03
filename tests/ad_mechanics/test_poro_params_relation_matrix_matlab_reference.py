from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_mechanics.utils.poro_params import PARAMETER_NAMES, poro_params


FIXTURE = (Path(__file__).resolve().parents[2] / "scripts" / "matlab" /
           "ad_mechanics" / "ref_poro_params_relation_matrix.mat")


def _struct_dict(value):
    return {name: float(np.asarray(getattr(value, name)).reshape(-1)[0])
            for name in value._fieldnames}


def test_every_mrst_relation_argument_inversion_matches_matlab():
    raw = loadmat(FIXTURE, squeeze_me=True, struct_as_record=False)
    phi = float(raw["phi"])
    for case in np.atleast_1d(raw["cases"]):
        if str(case.error):
            continue
        inputs = _struct_dict(case.inputs)
        expected = case.result
        actual = poro_params(phi, True, **inputs)
        for name in PARAMETER_NAMES:
            target = float(getattr(expected, name))
            if np.isfinite(target):
                assert np.isclose(actual[name], target, rtol=5e-6, atol=1e-10), (
                    int(case.relation_index), str(case.unknown), name,
                    target, actual[name], inputs,
                )
