"""Check `crossindconstruct` against MRST (scripts/matlab/ref_crossind.mat).

Regenerate from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_crossind_reference.m')"

Compared entry for entry after converting MATLAB's 1-based output. **Two divergences are asserted rather than skipped**, both from MRST building
four difference arrays and `cumsum`ing them:

- a **trailing** group with `nb2 = 0` leaves those arrays mismatched, so MRST
  raises "Arrays have incompatible sizes";
- an **interior** group with `nb1 = 0` makes two of its write positions
  collide, so only the later survives and the running offset silently loses
  that group's slots. With `nb1 = [2,0,1]`, `nb2 = [3,5,2]` the second arrays
  laid end to end give group 2 the slots `[8, 9]`; MRST answers `[5, 6]`, which
  are inside group 1's block. That is a wrong answer, not a different
  convention, and it is checked in both directions.

The port computes the same thing with two `repeat`s and an `mcolon`, which has
neither failure mode.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.utils.tensor_assembly.utils.crossindconstruct import (  # noqa: E402
    crossindconstruct,
)

# Cases MRST computes correctly, compared entry for entry.
CASES = [([2, 1], [3, 2]), ([1], [1]), ([3, 2, 1], [2, 4, 3])]


def main() -> int:
    ref = loadmat(Path(__file__).parent / "matlab" / "ref_crossind.mat",
                  squeeze_me=True, struct_as_record=False)
    failures = []
    for k, (nb1, nb2) in enumerate(CASES, start=1):
        a, b = crossindconstruct(nb1, nb2)
        r = ref[f"c{k}"]
        ta = np.atleast_1d(np.asarray(r.a, dtype=np.int64)).ravel() - 1
        tb = np.atleast_1d(np.asarray(r.b, dtype=np.int64)).ravel() - 1
        ok = np.array_equal(a, ta) and np.array_equal(b, tb)
        print(f"  {'ok ' if ok else 'FAIL'} nb1={nb1} nb2={nb2} "
              f"({a.size} pairs)")
        if not ok:
            failures.append(f"case {k}")

    # An interior empty group: MRST computes it, and gets it wrong.
    a, b = crossindconstruct([2, 0, 1], [3, 5, 2])
    r = ref["c4"]
    theirs = np.asarray(r.b, dtype=np.int64).ravel() - 1
    # Second arrays laid end to end give group 2 the slots [8, 9]; MRST's
    # duplicate write positions collide across the empty group and lose its
    # five slots, so it points group 2 at [5, 6] - inside group 1's block.
    ok = b.tolist()[-2:] == [8, 9] and theirs.tolist()[-2:] == [5, 6]
    print(f"  {'ok ' if ok else 'FAIL'} interior nb1=0: port gives "
          f"{b.tolist()[-2:]}, MRST gives {theirs.tolist()[-2:]} (wrong)")
    if not ok:
        failures.append("interior zero")

    ran = int(ref["trailing_zero_ran"])
    print(f"  {'ok ' if not ran else 'FAIL'} trailing nb2=0: MRST "
          f"{'raises, as expected' if not ran else 'unexpectedly succeeded'}")
    if ran:
        failures.append("trailing zero: expected MRST to fail")

    a, _ = crossindconstruct([2, 3], [4, 0])
    ok = a.size == 8
    print(f"  {'ok ' if ok else 'FAIL'} trailing nb2=0: the port gives "
          f"{a.size} pairs (2*4 + 3*0)")
    if not ok:
        failures.append("trailing zero: port")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("crossindconstruct matches MRST, and handles a case MRST cannot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
