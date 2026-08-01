"""Compare PyMRST's stateful gravity facade with MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.params import gravity


HERE = Path(__file__).parent


def main():
    path = HERE/"ref_params_gravity.mat"
    if not path.exists():
        raise SystemExit("Run dump_params_gravity_reference.m in MATLAB first")
    ref = scipy.io.loadmat(path, squeeze_me=True)
    actual = []
    actual.append(gravity("reset"))
    actual.append(gravity("x", "on"))
    actual.append(gravity("off", "y"))
    actual.append(gravity("yes"))
    actual.append(gravity("reset", "true", "z"))
    gravity("reset", "on")
    actual.append(gravity(2.5))
    actual.append(gravity([1.0, 2.0]))
    actual.append(gravity([1.0, 2.0, -3.0]))
    gravity(False)
    actual.append(gravity([4.0, 5.0]))
    actual.append(gravity(True))
    actual.append(gravity("x", "on", "reset"))
    gravity("reset")
    for index, vector in enumerate(actual):
        expected = np.asarray(ref[f"g{index}"], dtype=float).ravel()
        np.testing.assert_allclose(vector, expected, rtol=0.0, atol=1e-14)
        print(f"  ok   gravity state transition {index}")
    print("\nGravity command semantics match MATLAB MRST.")


if __name__ == "__main__":
    main()
