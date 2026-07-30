"""Phase 430: compare obj2hash digests with MRST."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import obj2hash

REF = Path(__file__).with_name("ref_p430_obj2hash.mat")


def _string(value):
    array = np.asarray(value)
    return "".join(array.ravel().tolist()) if array.dtype.kind in "US" else str(
        array.item()
    )


def main():
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase430_reference.m in MATLAB first")
        return 1
    # Preserve MATLAB storage classes: scipy otherwise narrows integer-valued
    # doubles to uint8, which would intentionally produce a different digest.
    ref = loadmat(REF, mat_dtype=True)
    numeric = ref["numeric"]
    logical_value = ref["logical_value"].astype(bool)
    cell_value = ["yes", np.array([True, False]), np.array([2.5, -7.0])]
    struct_value = {
        "numeric": numeric,
        "cell": cell_value,
        "ignored": 99.0,
    }
    struct_hash, structure = obj2hash(struct_value, return_structure=True)
    actual = {
        "out_numeric": obj2hash(numeric),
        "out_logical": obj2hash(logical_value),
        "out_char": obj2hash("PyMRST"),
        "out_cell": obj2hash(cell_value),
        "out_struct": struct_hash,
        "out_struct_numeric": structure["numeric"],
        "out_struct_cell": structure["cell"],
        "out_skip": obj2hash(struct_value, skip=("ignored",)),
        "out_max_size": obj2hash(numeric, max_size=1),
        "out_empty": obj2hash(np.array([])),
        "out_sparse": obj2hash(sp.csc_array(
            ([2.5, -1.0, 7.0], ([0, 2, 1], [0, 1, 3])),
            shape=(3, 4),
        )),
    }
    failed = 0
    for name, digest in actual.items():
        expected = _string(ref[name])
        ok = digest == expected
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {digest}"
              + ("" if ok else f" != {expected}"))
    print("\nobj2hash matches MRST!" if not failed else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
