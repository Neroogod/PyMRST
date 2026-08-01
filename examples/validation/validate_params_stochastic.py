"""Validate deterministic stochastic-rock algorithms against MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.params.rock import gaussian_field, log_norm_layers


HERE = Path(__file__).parent


def main():
    path = HERE/"ref_params_stochastic.mat"
    if not path.exists():
        raise SystemExit("Run dump_params_stochastic_reference.m in MATLAB first")
    ref = scipy.io.loadmat(path, squeeze_me=True)

    dims_g = tuple(np.asarray(ref["dimsG"], int).ravel())
    gaussian = gaussian_field(
        dims_g, vals=np.asarray(ref["valsG"]).ravel(),
        size=np.asarray(ref["szG"], int).ravel(), std=float(ref["stdG"]),
        field=np.asarray(ref["drawG"], float),
    )
    np.testing.assert_allclose(gaussian, ref["gaussian3"], rtol=2e-14, atol=2e-15)
    print("  ok   deterministic 3-D Gaussian field")

    dims_l = tuple(np.asarray(ref["dimsL"], int).ravel())
    bounds = np.asarray(ref["boundsL"], int).ravel()-1
    layered, returned = log_norm_layers(
        dims_l, mean_perm=np.asarray(ref["meansL"]).ravel(), indices=bounds,
        sigma=float(ref["sigmaL"]), a=float(ref["aL"]), b=float(ref["bL"]),
        size=np.asarray(ref["szL"], int).ravel(), std=float(ref["stdL"]),
        fields=[ref["drawL1"], ref["drawL2"]],
        shifts=[float(ref["shiftL1"]), float(ref["shiftL2"])],
        return_indices=True,
    )
    np.testing.assert_array_equal(returned, bounds)
    np.testing.assert_allclose(layered, np.ravel(ref["logLayers"], order="F"),
                               rtol=2e-14, atol=2e-14)
    print("  ok   uneven mean-controlled lognormal layers")

    _, default_bounds = log_norm_layers(dims_l, rng=np.random.default_rng(1),
                                        return_indices=True)
    _, all_bounds = log_norm_layers(dims_l, mean_perm=dims_l[2],
                                    rng=np.random.default_rng(1), return_indices=True)
    _, three_bounds = log_norm_layers(dims_l, mean_perm=3,
                                      rng=np.random.default_rng(1), return_indices=True)
    assert default_bounds.size == np.ravel(ref["Ldefault"]).size
    assert all_bounds.size == np.ravel(ref["Lall"]).size
    assert three_bounds.size == np.ravel(ref["Lthree"]).size
    assert three_bounds[0] == 0 and three_bounds[-1] == dims_l[2]
    print("  ok   native scalar layer-count structure")
    print("\nStochastic rock algorithms match MATLAB MRST.")


if __name__ == "__main__":
    main()
