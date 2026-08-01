"""Check `TensorMap` against MRST (scripts/matlab/ref_tensormap.mat).

Regenerate from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_tensormap_reference.m')"

Both the **results** and the **dispatch index vectors** are compared. The
results alone would not be enough: several different pairings can sum to the
same answer on a symmetric example, and it is the dispatch vectors that other
code consumes through `tbldispatch1`/`tbldispatch2`.

The reference is built by assigning `m.fromTbl` and friends after construction
rather than passing them in, because MRST's own `TensorMap()` takes
name-value arguments its `merge_options` call does honour - but the sibling
`CrossIndexArrayGenerator` does not, and doing it the same way in both keeps
the dump scripts comparable.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.utils.tensor_assembly.index_array import IndexArray  # noqa: E402
from pymrst.utils.tensor_assembly.tensor_map import TensorMap    # noqa: E402


def main() -> int:
    ref = loadmat(Path(__file__).parent / "matlab" / "ref_tensormap.mat",
                  squeeze_me=True, struct_as_record=False)
    failures = []

    def check(label, mine, theirs, base=0):
        mine = np.atleast_1d(np.asarray(mine, dtype=float)).ravel()
        theirs = np.atleast_1d(np.asarray(theirs, dtype=float)).ravel() - base
        ok = mine.shape == theirs.shape and np.allclose(mine, theirs)
        print(f"  {'ok ' if ok else 'FAIL'} {label}")
        if not ok:
            failures.append(label)
            print(f"       mine {mine.tolist()}\n       mrst {theirs.tolist()}")

    cf = IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})
    c = IndexArray({"cells": [0, 1]})

    m = TensorMap(from_tbl=cf, to_tbl=c, mergefds=["cells"]).setup()
    check("sum: result", m.eval(np.array([1.0, 10.0, 100.0])), ref["sum"].v)
    check("sum: dispind1", m.dispind1, ref["sum"].d1, base=1)
    check("sum: dispind2", m.dispind2, ref["sum"].d2, base=1)

    m2 = TensorMap(from_tbl=c, to_tbl=cf, mergefds=["cells"]).setup()
    check("broadcast: result", m2.eval(np.array([5.0, 9.0])),
          ref["broadcast"].v)
    check("broadcast: dispind1", m2.dispind1, ref["broadcast"].d1, base=1)
    check("broadcast: dispind2", m2.dispind2, ref["broadcast"].d2, base=1)

    check("transpose", m.eval_transpose(np.array([2.0, 3.0])), ref["transpose"])

    a = IndexArray({"cells": [0, 1, 2]})
    b = IndexArray({"cells": [2, 0, 1]})
    m3 = TensorMap(from_tbl=a, to_tbl=b, mergefds=["cells"]).setup()
    check("permutation: result", m3.eval(np.array([10.0, 20.0, 30.0])),
          ref["perm"].v)
    check("permutation: dispatch ind", m3.get_dispatch_ind(),
          ref["perm"].disp, base=1)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("TensorMap matches MRST's TensorMap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
