"""Compare advanced addBC/addSource structures with MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.params import add_bc, add_source


HERE = Path(__file__).parent


def _flat(ref, key):
    return np.asarray(ref[key]).ravel()


def main():
    path = HERE/"ref_params_forcing.mat"
    if not path.exists():
        raise SystemExit("Run dump_params_forcing_reference.m in MATLAB first")
    ref = scipy.io.loadmat(path, squeeze_me=True)

    bc = add_bc(None, [1, 3], ["Pressure", "FLUX"], 2, sat=[.2, .8])
    bc = add_bc(bc, [5], "pressure", 5, sat=[.6, .4])
    np.testing.assert_array_equal(bc.face, _flat(ref, "bcFace")-1)
    np.testing.assert_allclose(bc.value, _flat(ref, "bcValue"))
    np.testing.assert_allclose(bc.sat, ref["bcSat"])
    np.testing.assert_array_equal(
        [kind == "pressure" for kind in bc.type], _flat(ref, "bcTypeCode")
    )
    print("  ok   per-face BC types, values, and saturation")

    empty = add_source(None, [0, 2], 3)
    np.testing.assert_array_equal(empty.cell, _flat(ref, "srcEmptyCell")-1)
    np.testing.assert_allclose(empty.rate, _flat(ref, "srcEmptyRate"))
    np.testing.assert_array_equal(empty.sat.shape, _flat(ref, "srcEmptySatSize"))
    print("  ok   source default empty saturation")

    src = add_source(None, [0, 2], [1, -1], sat=[.25, .75])
    src = add_source(src, [4], 2, sat=[.6, .4])
    np.testing.assert_array_equal(src.cell, _flat(ref, "srcCell")-1)
    np.testing.assert_allclose(src.rate, _flat(ref, "srcRate"))
    np.testing.assert_allclose(src.sat, ref["srcSat"])
    print("  ok   source rate and saturation append")
    print("\nBoundary and source structures match MATLAB MRST.")


if __name__ == "__main__":
    main()
