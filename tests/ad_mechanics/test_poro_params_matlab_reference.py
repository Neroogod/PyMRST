from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pymrst.modules.ad_mechanics.utils.poro_params import poro_params


FIXTURE = Path(__file__).resolve().parents[2] / "scripts" / "matlab" / "ad_mechanics" / "ref_poro_params.mat"
BASIS_FIXTURE = FIXTURE.with_name("ref_poro_params_basis.mat")


def _field(struct, name):
    value = struct[name]
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return float(value)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="MATLAB poroParams fixture has not been generated")
def test_representative_parameter_bases_match_mrst():
    raw = loadmat(FIXTURE, squeeze_me=True, struct_as_record=False)["cases"]
    references = np.atleast_1d(raw)
    actual = [
        poro_params(0.20, E=12e9, nu=0.25, alpha=0.8, M=5e9),
        poro_params(0.25, K=9e9, G=4e9, alpha=0.75, M=6e9),
        poro_params(0.20, True, K=8e9, G=4e9, alpha=0.8, M=5e9),
    ]
    fields = ("K", "G", "E", "nu", "K_u", "B", "nu_u", "K_v", "K_vu")
    for reference, result in zip(references, actual):
        for name in fields:
            expected = float(getattr(reference, name))
            if np.isfinite(expected):
                assert result[name] == pytest.approx(expected, rel=2e-6, abs=1e-10)


@pytest.mark.skipif(not BASIS_FIXTURE.is_file(), reason="MATLAB basis fixture has not been generated")
def test_every_elastic_input_basis_matches_mrst_branch_selection():
    raw = loadmat(BASIS_FIXTURE, squeeze_me=True, struct_as_record=False)
    phi = float(raw["phi"])
    fields = ("K", "E", "G", "lambda", "nu", "K_v",
              "K_u", "E_u", "lambda_u", "nu_u", "K_vu")
    for family in ("drained", "undrained"):
        for case in np.atleast_1d(raw[family]):
            left = str(case.left)
            right = str(case.right)
            reference = case.result
            actual = poro_params(
                phi,
                **{left: float(getattr(reference, left)),
                   right: float(getattr(reference, right))},
            )
            for name in fields:
                expected = float(getattr(reference, name))
                if np.isfinite(expected):
                    assert actual[name] == pytest.approx(
                        expected, rel=3e-6, abs=1e-10
                    ), (family, left, right, name)
