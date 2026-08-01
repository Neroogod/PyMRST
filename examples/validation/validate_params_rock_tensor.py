"""Compare make_rock and perm_tensor layouts with MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid
from pymrst.params import make_rock, perm_tensor


HERE = Path(__file__).parent


def _check(name, actual, expected, *, index=False):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if index:
        expected = expected-1
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    print(f"  ok   {name}")


def main():
    path = HERE/"ref_params_rock_tensor.mat"
    if not path.exists():
        raise SystemExit("Run dump_params_rock_tensor_reference.m in MATLAB first")
    ref = scipy.io.loadmat(path, squeeze_me=True)

    cases = [
        ("K1", cart_grid([3]), [[2.0]], .2, 1),
        ("K2s", cart_grid([2, 2]), [[2.0]], .2, 2),
        ("K2d", cart_grid([2, 2]), [[2.0, 3.0]], [.1, .2, .3, .4], 2),
        ("K2f", cart_grid([2, 2]), [[2.0, .5, 3.0]], .25, 2),
        ("K3s", cart_grid([2, 1, 2]), [[2.0]], .2, 3),
        ("K3d", cart_grid([2, 1, 2]), [[2.0, 3.0, 4.0]], .2, 3),
        ("K3f", cart_grid([2, 1, 2]), [[2.0, .1, .2, 3.0, .3, 4.0]], .2, 3),
    ]
    built = {}
    for name, grid, permeability, porosity, dim in cases:
        rock = make_rock(grid, permeability, porosity)
        built[name] = rock
        tensor, rows, cols = perm_tensor(rock, dim)
        _check(name, tensor, np.asarray(ref[name]).reshape(tensor.shape))
        if name in ("K1", "K2s", "K3s"):
            suffix = str(dim)
            _check(f"row{suffix}", rows, np.ravel(ref[f"row{suffix}"]), index=True)
            _check(f"col{suffix}", cols, np.ravel(ref[f"col{suffix}"]), index=True)

    _check("makeRock 2-D permeability", built["K2d"].perm, ref["perm2d"])
    _check("makeRock per-cell porosity", built["K2d"].poro, np.ravel(ref["poro2d"]))
    _check("makeRock full 3-D permeability", built["K3f"].perm, ref["perm3f"])
    ntg = make_rock(cart_grid([2, 2]), 2.0, .2, ntg=.8).ntg
    _check("makeRock scalar NTG", ntg, np.ravel(ref["ntg2s"]))
    print("\nmakeRock and permTensor layouts match MATLAB MRST.")


if __name__ == "__main__":
    main()
