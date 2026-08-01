"""Check `simple_equilibrium` against MRST (scripts/matlab/ref_equil.mat).

Regenerate from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_equil_reference.m')"

Saturations are compared entry for entry: cell numbering agrees between the two
`cartGrid`s, and the twisted case rebuilds MATLAB's grid from its dumped node
coordinates so both codes see the same distorted geometry. That case is the one
that matters - it is where a cell's face-centroid extent stops matching its node
extent, and where a port that measured the height differently would diverge.

**Two divergences are asserted rather than skipped:**

- MRST's axis argument is 1-based, PyMRST's is 0-based, so MRST's `1` is `0`.
- **MRST cannot run on a 2-D grid at all.** Its default direction is `[0 0 1]`,
  which cannot multiply a 2-D grid's `(n, 2)` centroids. The reference records
  whether MATLAB errored; this asserts that it did, and that the port does not.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry              # noqa: E402
from pymrst.utils.equil.simple_equilibrium import simple_equilibrium       # noqa: E402


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_equil.mat",
                        squeeze_me=True, struct_as_record=False)
    g = compute_geometry(cart_grid([1, 1, 10], [1.0, 1.0, 10.0]))
    gx = compute_geometry(cart_grid([10, 1, 1], [10.0, 1.0, 1.0]))

    twisted = cart_grid([3, 3, 4], [3.0, 3.0, 4.0])
    twisted.nodes.coords = np.asarray(reference["twistcoords"], dtype=float)
    twisted = compute_geometry(twisted)

    cases = [
        ("two", simple_equilibrium(g, [5.0])),
        ("three", simple_equilibrium(g, [3.0, 7.0])),
        ("strad", simple_equilibrium(g, [3.25])),
        ("above", simple_equilibrium(g, [-100.0])),
        ("vec", simple_equilibrium(g, [5.0], [0.0, 0.0, 1.0])),
        # MATLAB's axis 1 is PyMRST's axis 0.
        ("axis1", simple_equilibrium(gx, [5.0], 0)),
        ("twist", simple_equilibrium(twisted, [1.5, 2.5])),
    ]

    failures = []
    for name, mine in cases:
        theirs = np.atleast_2d(np.asarray(reference[name], dtype=float))
        if theirs.shape[0] == 1 and mine.shape[0] != 1:
            theirs = theirs.T
        ok = mine.shape == theirs.shape and np.allclose(mine, theirs, atol=1e-12)
        detail = f"{mine.shape}"
        if not ok and mine.shape == theirs.shape:
            detail += f", max diff {np.max(np.abs(mine - theirs)):.3e}"
        print(f"  {'ok ' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failures.append(name)

    # The 2-D divergence.
    ran = int(reference["twod_ran"])
    print(f"  {'ok ' if not ran else 'FAIL'} 2-D: MRST "
          f"{'errored, as expected' if not ran else 'unexpectedly succeeded'}"
          + (f" -> {reference['twod_msg']}" if not ran else ""))
    if ran:
        failures.append("2-D: expected MRST to fail")

    g2 = compute_geometry(cart_grid([1, 10], [1.0, 10.0]))
    mine = simple_equilibrium(g2, [5.0])
    ok = mine.shape == (10, 2) and np.allclose(mine.sum(axis=1), 1.0)
    print(f"  {'ok ' if ok else 'FAIL'} 2-D: the port returns {mine.shape} "
          f"summing to one")
    if not ok:
        failures.append("2-D: port")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("simple_equilibrium matches MRST's simpleEquilibrium, and works in "
          "2-D where MRST's cannot run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
