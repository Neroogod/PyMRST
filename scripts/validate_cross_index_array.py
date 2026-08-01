"""Check `cross_index_array` against MRST (scripts/matlab/ref_crossindexarray.mat).

Regenerate from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_crossindexarray_reference.m')"

Both the **rows and their order** are compared, not just the row set. Order is
part of what this function promises - `indstruct` maps output rows back to the
inputs positionally, so a join that produced the right pairings in a different
order would silently corrupt anything using those maps.

Column names are compared too, since the join's output names are assembled from
both inputs and getting that wrong is easy.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.utils.tensor_assembly.cross_index_array import (  # noqa: E402
    cross_index_array,
)
from pymrst.utils.tensor_assembly.index_array import IndexArray  # noqa: E402


def names_of(entry):
    raw = entry.fdnames
    return [str(n) for n in (raw if isinstance(raw, (list, np.ndarray))
                             else [raw])]


def main() -> int:
    ref = loadmat(Path(__file__).parent / "matlab" / "ref_crossindexarray.mat",
                  squeeze_me=True, struct_as_record=False)
    failures = []

    def check(label, got, entry, extra=None):
        theirs = np.atleast_2d(np.asarray(entry.inds, dtype=np.int64)) - 1
        mine = np.atleast_2d(np.asarray(got.inds, dtype=np.int64))
        ok = mine.shape == theirs.shape and np.array_equal(mine, theirs)
        print(f"  {'ok ' if ok else 'FAIL'} {label}: rows {mine.shape} "
              f"vs {theirs.shape}")
        if not ok:
            failures.append(label)
            print(f"       mine {mine.tolist()}\n       mrst {theirs.tolist()}")
        same = got.fdnames == names_of(entry)
        print(f"  {'ok ' if same else 'FAIL'} {label}: names {got.fdnames}")
        if not same:
            failures.append(f"{label} names")
        if extra:
            extra()

    t1 = IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})
    t2 = IndexArray({"faces": [3, 7, 7], "nodes": [10, 20, 30]})
    joined, s = cross_index_array(t1, t2, ["faces"])

    def maps():
        for k, key in ((0, "i1"), (1, "i2")):
            mine = np.asarray(s[k]["inds"], dtype=np.int64)
            theirs = np.asarray(getattr(ref["join"], key), dtype=np.int64).ravel() - 1
            ok = np.array_equal(mine, theirs)
            print(f"  {'ok ' if ok else 'FAIL'} join: indstruct[{k}]")
            if not ok:
                failures.append(f"indstruct {k}")

    check("join", joined, ref["join"], maps)

    a = IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})
    b = IndexArray({"cells": [0, 1], "faces": [7, 7], "v": [5, 6]})
    check("two keys", cross_index_array(a, b, ["cells", "faces"])[0], ref["two"])

    d = IndexArray({"faces": [1, 2]})
    e = IndexArray({"faces": [2, 3], "nodes": [7, 8]})
    check("partial overlap", cross_index_array(d, e, ["faces"])[0],
          ref["partial"])

    p1 = IndexArray({"cells": [0, 1]})
    p2 = IndexArray({"nodes": [7, 8, 9]})
    check("pure product",
          cross_index_array(p1, p2, [], pure_product=True), ref["product"])

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("cross_index_array matches MRST's crossIndexArray.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
