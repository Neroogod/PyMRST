"""Validate ``double2ADI``, ``getSampleAD``, ``structPropEvaluated``, and
``crossFlowMixtureDensity`` against MRST.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    cross_flow_mixture_density,
    double_to_adi,
    get_sample_ad,
    struct_prop_evaluated,
)
from pymrst.utils.adi import ADI


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_ad_sampling_crossflow.mat",
        squeeze_me=True,
    )
    sample = ADI(np.array([1.0, 2.0]), [sp.eye(2, format="csr")])
    converted = double_to_adi([4.0, 5.0, 6.0], sample)
    sample_ad, is_ad = get_sample_ad(np.array([2.0]), sample, np.array([3.0]))
    sample_numeric, is_numeric_ad = get_sample_ad(
        np.array([2.0]), np.array([3.0])
    )
    properties = {"a": np.array([1.0]), "b": np.array([])}
    mixture = cross_flow_mixture_density(
        np.array([[-2.0, -4.0], [1.0, 2.0], [0.0, 0.0]]),
        np.array([-1.0, 2.0, 3.0]),
        np.array([[10.0, 20.0], [0.0, 0.0]]),
        SimpleNamespace(perf2well=np.array([0, 0, 1]), W=[1, 2]),
    )
    checks = {
        "converted values": (converted.val, ref["converted_value"]),
        "converted Jacobian": (
            converted.jac[0].toarray(),
            ref["converted_jacobian"],
        ),
        "numeric conversion": (
            double_to_adi([7.0, 8.0], 1.0),
            ref["converted_double"],
        ),
        "sample AD values": (sample_ad.val, ref["sample_ad_value"]),
        "sample AD flag": (is_ad, ref["is_ad"]),
        "numeric sample": (sample_numeric, ref["sample_numeric"]),
        "numeric AD flag": (is_numeric_ad, ref["is_numeric_ad"]),
        "property a": (
            struct_prop_evaluated(properties, "a"),
            ref["evaluated_a"],
        ),
        "property b": (
            struct_prop_evaluated(properties, "b"),
            ref["evaluated_b"],
        ),
        "mixture density": (mixture, ref["mixture"]),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual = np.asarray(actual, dtype=float).squeeze()
        expected = np.asarray(expected, dtype=float).squeeze()
        error = float(np.max(np.abs(actual - expected)))
        ok = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-12, atol=1e-12
        )
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
